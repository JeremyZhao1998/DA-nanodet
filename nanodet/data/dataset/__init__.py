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

import os
import copy
import warnings

from .coco import CocoDataset, CocoDatasetCrossDomain


def build_dataset(cfg, mode, data_root=None, cross_domain=False):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")
    if data_root is not None:
        for key in dataset_cfg:
            if key.endswith('path'):
                dataset_cfg[key] = os.path.join(data_root, dataset_cfg[key])
    if name == "CocoDataset":
        if cross_domain:
            return CocoDatasetCrossDomain(mode=mode, **dataset_cfg)
        return CocoDataset(mode=mode, **dataset_cfg)
    else:
        warnings.warn(
            "DA-nanodet now only support COCO-style dataset"
        )
        raise NotImplementedError("Unknown dataset type!")
