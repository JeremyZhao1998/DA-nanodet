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

import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from pytorch_lightning.callbacks import TQDMProgressBar

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask, TeachingTask
from nanodet.util import (
    NanoDetLightningLogger,
    set_print,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def set_env():
    torch.set_float32_matmul_precision('high')
    load_config(args.config)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError(
            "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
            "but got {} and {}".format(
                cfg.model.arch.head.num_classes, len(cfg.class_names)
            )
        )
    local_rank = int(args.local_rank)
    mkdir(local_rank, cfg.save_dir)
    return NanoDetLightningLogger(cfg.save_dir)


def set_random_seed():
    cudnn.enabled = True
    cudnn.benchmark = True
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)


def set_device():
    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy, precision = ("cpu", None, None, cfg.device.precision)
    else:
        accelerator, devices, strategy, precision = ("gpu", cfg.device.gpu_ids, None, cfg.device.precision)

    if devices and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)
        set_print()
    return accelerator, devices, strategy, precision


def load_weights(task):
    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model, map_location='cpu' if cfg.device.gpu_ids == -1 else 'cuda')
        if "pytorch-lightning_version" not in ckpt:
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        if hasattr(task, 'teacher'):
            load_model_weight(task.teacher, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))


def build_dataloader(dataset, is_train):
    return DataLoader(
        dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=is_train,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=is_train,
    )


def set_trainer():
    # Set device
    accelerator, devices, strategy, precision = set_device()
    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
        precision=precision,
    )
    return trainer


def source_only():
    # Setup data
    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train", cfg.data.data_root)
    val_dataset = build_dataset(cfg.data.val, "test", cfg.data.data_root)
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    train_dataloader = build_dataloader(train_dataset, is_train=True)
    val_dataloader = build_dataloader(val_dataset, is_train=False)
    # Create model
    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)
    load_weights(task)
    # Create trainer and start training
    trainer = set_trainer()
    trainer.fit(task, train_dataloader, val_dataloader)


def teaching():
    # Setup data
    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train", cfg.data.data_root, teaching=True)
    val_dataset = build_dataset(cfg.data.val, "test", cfg.data.data_root)
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    source_dataloader = build_dataloader(train_dataset, is_train=True)
    val_dataloader = build_dataloader(val_dataset, is_train=False)
    # Create model
    logger.info("Creating model...")
    task = TeachingTask(cfg, evaluator)
    load_weights(task)
    # Create trainer and start training
    trainer = set_trainer()
    trainer.fit(task, source_dataloader, val_dataloader)


def main():
    if cfg.mode == 'teaching':
        teaching()
    elif cfg.mode == 'source_only':
        source_only()


if __name__ == "__main__":
    args = parse_args()
    logger = set_env()
    logger.dump_cfg(cfg)
    set_random_seed()
    main()
