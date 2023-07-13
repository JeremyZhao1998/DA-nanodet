# Copyright 2021 RangiLyu.
# Modified by Zijing Zhao, 2023.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import warnings
from typing import Any, Dict, List
from tqdm import tqdm

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
import numpy as np

from nanodet.data.batch_process import stack_batch_img
from nanodet.optim import build_optimizer
from nanodet.util import convert_avg_params, gather_results, mkdir

from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager
from ..model.loss.iou_loss import bbox_overlaps


class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "NanoDet"
        self.weight_averager = None
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(
                cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)

    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        batch = self._preprocess_batch_input(batch)
        preds = self.forward(batch["img"])
        results = self.model.head.post_process(preds, batch)
        return results

    def log_training_losses(self, batch_idx, loss_states):
        # log train losses
        if self.global_step % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                self.trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        preds, loss, loss_states = self.model.forward_train(batch)
        self.log_training_losses(batch_idx, loss_states)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))

    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        if self.weight_averager is not None:
            preds, loss, loss_states = self.avg_model.forward_train(batch)
        else:
            if hasattr(self, "teacher"):
                preds, loss, loss_states = self.teacher.forward_train(batch)
            else:
                preds, loss, loss_states = self.model.forward_train(batch)

        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                sum(self.trainer.num_val_batches),
                memory,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        if hasattr(self, "teacher"):
            dets = self.teacher.head.post_process(preds, batch)
        else:
            dets = self.model.head.post_process(preds, batch)
        return dets

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the
        outputs of all validation steps.Evaluating results
        and save best model.
        Args:
            validation_step_outputs: A list of val outputs

        """
        results = {}
        for res in validation_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(best_save_path, "nanodet_model_best.pth")
                )
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch + 1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            self.logger.log_metrics(eval_results, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))

    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        return dets

    def test_epoch_end(self, test_step_outputs):
        results = {}
        for res in test_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, "results.json")
            json.dump(res_json, open(json_path, "w"))

            if self.cfg.test_mode == "val":
                eval_results = self.evaluator.evaluate(
                    all_results, self.cfg.save_dir, rank=self.local_rank
                )
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            self.logger.info("Skip test on rank {}".format(self.local_rank))

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        optimizer = build_optimizer(
            torch.nn.ModuleList([self.model, self.discriminators]) if hasattr(self, 'discriminators') else self.model,
            optimizer_cfg
        )

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        scheduler = {
            "scheduler": build_scheduler(optimizer=optimizer, **schedule_cfg),
            "interval": "epoch",
            "frequency": 1,
        }
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_idx=None,
            optimizer_closure=None,
            on_tpu=None,
            using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == "constant":
                k = self.cfg.schedule.warmup.ratio
            elif self.cfg.schedule.warmup.name == "linear":
                k = 1 - (
                        1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                ) * (1 - self.cfg.schedule.warmup.ratio)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (
                        1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                )
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * k

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)

    # ------------Hooks-----------------
    def on_fit_start(self) -> None:
        if "weight_averager" in self.cfg.model:
            self.logger.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                self.weight_averager.to(self.weight_averager.device)
                return
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.weight_averager.load_from(self.model)

    def on_train_epoch_start(self):
        self.model.set_epoch(self.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_test_epoch_start(self) -> None:
        if self.weight_averager:
            self.on_load_checkpoint({"state_dict": self.state_dict()})
            self.weight_averager.apply_to(self.model)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")


class GradReverse(torch.autograd.Function):

    def __init__(self):
        super(GradReverse, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GradReverseLayer(torch.nn.Module):
    def __init__(self, lambd=1):
        super(GradReverseLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)


class MultiConv2d(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Conv2d(n, k, kernel_size=(3, 3), padding=1) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TeachingTask(TrainingTask):

    def __init__(self, cfg, evaluator=None):
        super().__init__(cfg, evaluator)
        # Teacher model
        self.teacher = build_model(cfg.model)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.ema_alpha = cfg.ema_alpha
        self.tgt_loss_scale = cfg.tgt_loss_scale
        # Dynamic threshold
        self.thresholds = [cfg.threshold.start_value for _ in range(cfg.model.arch.head.num_classes)]
        self.source_logits = [0.0 for _ in range(cfg.model.arch.head.num_classes)]
        self.source_logits_cnt = [0 for _ in range(cfg.model.arch.head.num_classes)]
        self.gamma = cfg.threshold.gamma
        self.min_dt = cfg.threshold.min_value
        self.max_dt = cfg.threshold.max_value

    def _ema_update(self):
        state_dict, student_state_dict = self.teacher.state_dict(), self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = self.ema_alpha * value + (1 - self.ema_alpha) * student_state_dict[key].detach()
        self.teacher.load_state_dict(state_dict)

    def _select_pseudo_labels(self, dets_tgt_tch):
        img_ids, pseudo_boxes, pseudo_labels = [], [], []
        for img_id, value in dets_tgt_tch:
            img_ids.append(img_id)
            img_box, img_label = [], []
            for label, data in value.items():
                for proposal in data:
                    if proposal[4] > self.thresholds[label]:
                        img_box.append(proposal[:4])
                        img_label.append(label)
            pseudo_boxes.append(np.asarray(img_box, dtype=np.float32))
            pseudo_labels.append(np.asarray(img_label, dtype=np.int64))
        return pseudo_boxes, pseudo_labels

    def _record_source_logits(self, batch, preds):
        dets = self.model.head.post_process(preds[0], batch, pseudo=True)
        for idx, (img_id, value) in enumerate(dets):
            for label, data in value.items():
                if len(data) == 0:
                    continue
                proposal_boxes = torch.tensor(data, device=preds[0].device)[:, :4]
                gt_boxes = torch.tensor(batch['gt_bboxes'][idx], device=preds[0].device)
                gt_labels = batch['gt_labels'][idx]
                ious = bbox_overlaps(proposal_boxes, gt_boxes, mode='iou')
                selected = torch.gt(ious, 0.5).nonzero()
                for pair in selected:
                    if label == gt_labels[pair[1]]:
                        self.source_logits[label] += data[pair[0]][4]
                        self.source_logits_cnt[label] += 1

    def _preprocess_teaching_batch(self, batch):
        batch_tgt_stu, batch_tgt_tch = {}, {}
        original_keys = list(batch.keys())
        for key in original_keys:
            if key.endswith('tgt_stu'):
                batch_tgt_stu[key[:-8]] = batch.pop(key)
            if key.endswith('tgt_tch'):
                batch_tgt_tch[key[:-8]] = batch.pop(key)
        batch = self._preprocess_batch_input(batch)
        batch_tgt_tch = self._preprocess_batch_input(batch_tgt_tch)
        batch_tgt_stu = self._preprocess_batch_input(batch_tgt_stu)
        return batch, batch_tgt_stu, batch_tgt_tch

    def training_step(self, batch, batch_idx):
        # Pre-process batch
        batch, batch_tgt_stu, batch_tgt_tch = self._preprocess_teaching_batch(batch)
        # Teacher forward
        with torch.no_grad():
            # EMA
            self._ema_update()
            # Teacher forward
            preds_tgt_tch, _, _ = self.teacher.forward_train(batch_tgt_tch)
            dets_tgt_tch = self.teacher.head.post_process(preds_tgt_tch, batch_tgt_tch, pseudo=True)
            # Select pseudo labels
            pseudo_boxes, pseudo_labels = self._select_pseudo_labels(dets_tgt_tch)
            batch_tgt_stu['gt_bboxes'] = pseudo_boxes
            batch_tgt_stu['gt_labels'] = pseudo_labels
        # Student forward
        preds, loss, loss_states = self.model.forward_train(batch, return_features=True)
        preds_tgt_stu, loss_tgt_stu, loss_states_tgt_stu = self.model.forward_train(batch_tgt_stu, True, True)
        # Record source logits
        self._record_source_logits(batch, preds)
        # Loss
        loss += self.tgt_loss_scale * loss_tgt_stu
        loss_states['loss_tgt'] = self.tgt_loss_scale * loss_tgt_stu
        self.log_training_losses(batch_idx, loss_states)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))
        source_logits_mean = [logit / cnt if cnt > 0 else 0.0
                              for logit, cnt in zip(self.source_logits, self.source_logits_cnt)]
        print('Logits means: ', source_logits_mean)
        self.thresholds = [np.clip(mean * (1 - self.gamma) + self.gamma * threshold,
                                   self.min_dt, self.max_dt)
                           for mean, threshold in zip(source_logits_mean, self.thresholds)]
        print('New thresholds: ', self.thresholds)
        self.source_logits = [0.0 for _ in range(self.cfg.model.arch.head.num_classes)]
        self.source_logits_cnt = [0 for _ in range(self.cfg.model.arch.head.num_classes)]


class AlignmentTask(TrainingTask):

    def __init__(self, cfg, evaluator=None):
        super().__init__(cfg, evaluator)
        # Detection loss scale
        self.src_loss_scale = cfg.src_loss_scale
        self.tgt_loss_scale = cfg.tgt_loss_scale
        # Discriminator
        if cfg.discriminators.enable:
            self.dis_hidden_size = cfg.discriminators.dis_hidden_size
            self.discriminators = torch.nn.ModuleList(
                [MultiConv2d(self.model.fpn.out_channels, self.dis_hidden_size, 2, 2)
                 for _ in range(self.model.fpn.num_outs)]
            )
            self.dis_scale = cfg.discriminators.dis_scale
            self.grad_reverse = GradReverseLayer()
        # Feature projector
        if cfg.features_gt.enable:
            self.features_path = os.path.join(cfg.data.data_root, cfg.features_gt.features_path)
            print('Loading features from: ', self.features_path)
            self.features_gt = torch.from_numpy(np.load(self.features_path))
            self.projector = torch.nn.Conv2d(
                in_channels=cfg.model.arch.fpn.out_channels,
                out_channels=cfg.features_gt.size[0],
                kernel_size=1
            )
            self.features_scale = cfg.features_gt.scale

    def discriminator_forward(self, features, domain_label_value):
        features = [self.grad_reverse(f) for f in features]
        dis_preds = [dis(f).flatten(start_dim=2) for dis, f in zip(self.discriminators, features)]
        dis_preds = torch.cat(dis_preds, dim=2)
        domain_label = torch.zeros((dis_preds.shape[0], dis_preds.shape[2]), dtype=torch.long, device=dis_preds.device)
        torch.fill(domain_label, domain_label_value)
        loss = torch.nn.functional.cross_entropy(dis_preds, domain_label)
        return loss

    def _get_features(self, batch):
        features = self.features_gt[batch['img_info']['id']].to(batch['img'].device)
        return features

    def training_step(self, batch, batch_idx):
        # Pre-process batch
        batch_tgt = {}
        for key in list(batch.keys()):
            if key.endswith('tgt_tch'):
                batch_tgt[key[:-8]] = batch.pop(key)
        batch = self._preprocess_batch_input(batch)
        batch_tgt = self._preprocess_batch_input(batch_tgt)
        # Model forward
        preds, loss, loss_states = self.model.forward_train(batch, return_features=True)
        preds_tgt, loss_tgt, loss_states_tgt = self.model.forward_train(batch_tgt, return_features=True)
        loss *= self.src_loss_scale
        for key, value in loss_states.items():
            loss_states[key] *= self.src_loss_scale
        loss += self.tgt_loss_scale * loss_tgt
        loss_states['loss_tgt'] = self.tgt_loss_scale * loss_tgt
        # Discriminator forward
        if hasattr(self, 'discriminators'):
            loss_dis = self.discriminator_forward(preds[1], 0) + self.discriminator_forward(preds_tgt[1], 1)
            loss += loss_dis * self.dis_scale
            loss_states['loss_dis'] = loss_dis
        # Feature alignment
        if hasattr(self, 'projector'):
            features = preds_tgt[1][-1]
            features_gt = self._get_features(batch_tgt)
            loss_features = torch.nn.functional.mse_loss(self.projector(features), features_gt)
            loss += self.features_scale * loss_features
            loss_states['loss_features'] = loss_features
        self.log_training_losses(batch_idx, loss_states)
        return loss
