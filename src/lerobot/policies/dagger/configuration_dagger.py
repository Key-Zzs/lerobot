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

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig


@dataclass
class DAggerDatasetConfig:
    # Fixed offline dataset used as DAgger seed.
    seed_repo_id: str
    seed_root: str | Path | None = None

    # Aggregated dataset where new DAgger episodes are appended.
    aggregated_repo_id: str | None = None
    aggregated_root: str | Path | None = None

    # If aggregated dataset already exists, continue appending to it.
    resume_aggregation: bool = True
    # If aggregated dataset does not exist, bootstrap it by copying seed dataset.
    copy_seed_if_missing: bool = True

    # Optional async image writer setup for video datasets.
    image_writer_processes: int = 0
    image_writer_threads: int = 0


@dataclass
class DAggerRolloutConfig:
    episodes_per_round: int = 5
    episode_time_s: float = 45.0
    reset_time_s: float = 10.0
    fps: int = 30
    single_task: str = "dagger"
    reset_with_expert: bool = True


@dataclass
class DAggerExpertConfig:
    # "teleop": use teleoperator.get_action as expert
    # "callable": load python callable from "module.submodule:callable"
    mode: str = "teleop"
    teleop: TeleoperatorConfig | None = None
    callable_path: str | None = None


@dataclass
class DAggerBetaConfig:
    # Supported: linear, exponential, constant
    schedule: str = "linear"
    beta_start: float = 1.0
    beta_end: float = 0.0
    # Used only for exponential schedule.
    exp_decay: float = 0.8


@dataclass
class DAggerTrainingConfig:
    rounds: int = 10
    steps_per_round: int = 2_000
    batch_size: int = 8
    num_workers: int = 4
    log_freq: int = 100
    save_checkpoint: bool = True


@dataclass
class DAggerPipelineConfig:
    policy: PreTrainedConfig
    dataset: DAggerDatasetConfig
    robot: RobotConfig | None = None
    rollout: DAggerRolloutConfig = field(default_factory=DAggerRolloutConfig)
    expert: DAggerExpertConfig = field(default_factory=DAggerExpertConfig)
    beta: DAggerBetaConfig = field(default_factory=DAggerBetaConfig)
    training: DAggerTrainingConfig = field(default_factory=DAggerTrainingConfig)

    output_dir: Path | None = None
    job_name: str | None = None
    resume: bool = False
    seed: int | None = 1000

    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None

    def validate(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy.type != "act":
            raise ValueError(
                "This minimal DAgger pipeline currently targets ACT only. "
                f"Got policy.type={self.policy.type!r}."
            )

        if not self.job_name:
            self.job_name = f"dagger_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            run_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/dagger") / run_dir
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if self.output_dir.exists() and not self.resume:
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume={self.resume}."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset.aggregated_repo_id is None:
            if "/" in self.dataset.seed_repo_id:
                user, name = self.dataset.seed_repo_id.split("/", maxsplit=1)
                self.dataset.aggregated_repo_id = f"{user}/{name}_dagger"
            else:
                self.dataset.aggregated_repo_id = f"{self.dataset.seed_repo_id}_dagger"

        if self.training.rounds <= 0:
            raise ValueError("training.rounds must be > 0")
        if self.training.steps_per_round <= 0:
            raise ValueError("training.steps_per_round must be > 0")
        if self.training.log_freq <= 0:
            raise ValueError("training.log_freq must be > 0")
        if self.dataset.image_writer_processes < 0:
            raise ValueError("dataset.image_writer_processes must be >= 0")
        if self.dataset.image_writer_threads < 0:
            raise ValueError("dataset.image_writer_threads must be >= 0")
        if self.dataset.image_writer_processes > 0 and self.dataset.image_writer_threads == 0:
            self.dataset.image_writer_threads = 4

        if not self.use_policy_training_preset and (self.optimizer is None):
            raise ValueError(
                "optimizer must be set when use_policy_training_preset=False."
            )
        if self.use_policy_training_preset:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]
