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
from typing import Any

import numpy as np
import torch

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


def resolve_dagger_data_root(path: Path) -> Path | None:
    """Resolve dataset root, preferring `<path>/dagger_data` when present."""
    dagger_data_root = path / "dagger_data"
    if is_lerobot_dataset_root(dagger_data_root):
        return dagger_data_root
    if is_lerobot_dataset_root(path):
        return path
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return False
        return bool(value[0])
    return bool(value)


class ExpertOnlyLeRobotDataset(torch.utils.data.Dataset):
    """Torch dataset wrapper that keeps only rows with `is_expert == True`."""

    supports_episode_sampler = False

    def __init__(self, base_dataset: LeRobotDataset):
        self.base_dataset = base_dataset
        self.meta = base_dataset.meta
        self.features = base_dataset.features

        if "is_expert" not in base_dataset.hf_dataset.column_names:
            self.indices = list(range(len(base_dataset)))
            logging.warning(
                "[DAgger offline] column `is_expert` missing. Falling back to full dataset (%d frames).",
                len(self.indices),
            )
        else:
            is_expert_column = base_dataset.hf_dataset["is_expert"]
            self.indices = [idx for idx, value in enumerate(is_expert_column) if _to_bool(value)]
            logging.info(
                "[DAgger offline] expert filter kept %d / %d frames.",
                len(self.indices),
                len(base_dataset),
            )

        if len(self.indices) == 0:
            raise ValueError(
                "No expert frames available after filtering `is_expert == True`. "
                "Please collect DAgger expert data first."
            )

    @property
    def num_frames(self) -> int:
        return len(self.indices)

    @property
    def num_episodes(self) -> int:
        return self.base_dataset.num_episodes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)


def ensure_aggregated_dataset_ready(dataset_cfg: DAggerDatasetConfig) -> tuple[str, Path]:
    seed_root = resolve_dataset_root(dataset_cfg.seed_repo_id, dataset_cfg.seed_root)
    aggr_repo_id = dataset_cfg.aggregated_repo_id
    if aggr_repo_id is None:
        raise ValueError("dataset.aggregated_repo_id cannot be None after validation.")
    aggr_root = resolve_dataset_root(aggr_repo_id, dataset_cfg.aggregated_root)

    resolved_aggr_root = resolve_dagger_data_root(aggr_root)
    if resolved_aggr_root is not None:
        if is_lerobot_dataset_root(aggr_root) and not dataset_cfg.resume_aggregation:
            raise FileExistsError(
                f"Aggregated dataset already exists at {aggr_root} and resume_aggregation=False."
            )
        logging.info("Using existing aggregated dataset at %s", resolved_aggr_root)
        return aggr_repo_id, resolved_aggr_root

    resolved_seed_root = resolve_dagger_data_root(seed_root)
    if resolved_seed_root is not None:
        logging.info(
            "Using seed dataset directly for offline DAgger training at %s (aggregated dataset not found).",
            resolved_seed_root,
        )
        return dataset_cfg.seed_repo_id, resolved_seed_root

    if not dataset_cfg.copy_seed_if_missing:
        raise FileNotFoundError(
            f"Aggregated dataset does not exist at {aggr_root} and copy_seed_if_missing=False."
        )

    if resolved_seed_root is None:
        raise FileNotFoundError(
            f"Seed dataset root {seed_root} does not look like a LeRobot dataset "
            "(missing meta/info.json)."
        )

    logging.info(
        "Bootstrapping aggregated dataset by copying seed dataset:\n"
        "  seed: %s\n"
        "  dest: %s",
        resolved_seed_root,
        aggr_root,
    )

    aggregate_datasets(
        repo_ids=[dataset_cfg.seed_repo_id],
        aggr_repo_id=aggr_repo_id,
        roots=[resolved_seed_root],
        aggr_root=aggr_root,
    )

    return aggr_repo_id, aggr_root


def make_training_dataset(
    repo_id: str,
    root: Path,
    policy_cfg: PreTrainedConfig,
    expert_only: bool = True,
) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    base_dataset = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta_timestamps)
    if expert_only:
        return ExpertOnlyLeRobotDataset(base_dataset)
    return base_dataset
