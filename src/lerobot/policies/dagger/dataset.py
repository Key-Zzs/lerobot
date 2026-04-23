#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.dagger.configuration_dagger import DAggerDatasetConfig
from lerobot.utils.constants import HF_LEROBOT_HOME


def resolve_dataset_root(repo_id: str, root: str | Path | None) -> Path:
    return Path(root) if root is not None else HF_LEROBOT_HOME / repo_id


def is_lerobot_dataset_root(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def ensure_aggregated_dataset_ready(dataset_cfg: DAggerDatasetConfig) -> tuple[str, Path]:
    seed_root = resolve_dataset_root(dataset_cfg.seed_repo_id, dataset_cfg.seed_root)
    aggr_repo_id = dataset_cfg.aggregated_repo_id
    if aggr_repo_id is None:
        raise ValueError("dataset.aggregated_repo_id cannot be None after validation.")
    aggr_root = resolve_dataset_root(aggr_repo_id, dataset_cfg.aggregated_root)

    if is_lerobot_dataset_root(aggr_root):
        if not dataset_cfg.resume_aggregation:
            raise FileExistsError(
                f"Aggregated dataset already exists at {aggr_root} and resume_aggregation=False."
            )
        logging.info("Using existing aggregated dataset at %s", aggr_root)
        return aggr_repo_id, aggr_root

    if not dataset_cfg.copy_seed_if_missing:
        raise FileNotFoundError(
            f"Aggregated dataset does not exist at {aggr_root} and copy_seed_if_missing=False."
        )

    if not is_lerobot_dataset_root(seed_root):
        raise FileNotFoundError(
            f"Seed dataset root {seed_root} does not look like a LeRobot dataset "
            "(missing meta/info.json)."
        )

    logging.info(
        "Bootstrapping aggregated dataset by copying seed dataset:\n"
        "  seed: %s\n"
        "  dest: %s",
        seed_root,
        aggr_root,
    )

    aggregate_datasets(
        repo_ids=[dataset_cfg.seed_repo_id],
        aggr_repo_id=aggr_repo_id,
        roots=[seed_root],
        aggr_root=aggr_root,
    )

    return aggr_repo_id, aggr_root


def make_training_dataset(
    repo_id: str,
    root: Path,
    policy_cfg: PreTrainedConfig,
) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta_timestamps)
    return dataset
