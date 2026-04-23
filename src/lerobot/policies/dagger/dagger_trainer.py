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

import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import combine_feature_dicts, cycle
from lerobot.policies.dagger.beta_schedule import BetaScheduler
from lerobot.policies.dagger.configuration_dagger import DAggerPipelineConfig
from lerobot.policies.dagger.dataset import ensure_aggregated_dataset_ready, make_training_dataset
from lerobot.policies.dagger.expert import ExpertPolicy, make_expert
from lerobot.policies.dagger.rollout import RolloutStats, collect_dagger_rollouts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


def _build_dataloader(
    cfg: DAggerPipelineConfig,
    dataset: LeRobotDataset,
) -> torch.utils.data.DataLoader:
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=cfg.policy.device == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )


def _save_round_checkpoint(
    checkpoint_dir: Path,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    round_state: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
    )
    postprocessor.save_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
    )

    with open(checkpoint_dir / "dagger_round_state.json", "w") as f:
        json.dump(round_state, f, indent=2)


def _compute_train_metrics(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    batch: dict[str, torch.Tensor],
    grad_clip_norm: float,
) -> tuple[float, float, dict[str, float]]:
    device = torch.device(policy.config.device)
    use_amp = scaler.is_enabled() and device.type == "cuda"

    policy.train()
    optimizer.zero_grad(set_to_none=True)

    autocast_ctx = (
        torch.autocast(device_type="cuda") if use_amp else nullcontext()
    )

    with autocast_ctx:
        loss, loss_dict = policy.forward(batch)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
        )
        optimizer.step()

    train_log = {k: float(v) for k, v in (loss_dict or {}).items()}
    return float(loss.item()), float(grad_norm.item()), train_log


def _build_expected_dataset_features(dataset: LeRobotDataset, robot) -> dict[str, dict]:
    (
        teleop_action_processor,
        _,
        robot_observation_processor,
    ) = make_default_processors()

    use_videos = any(ft["dtype"] == "video" for ft in dataset.features.values())

    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=use_videos,
    )
    observation_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=use_videos,
    )
    return combine_feature_dicts(action_features, observation_features)


def train_dagger(cfg: DAggerPipelineConfig) -> None:
    cfg.validate()
    init_logging()
    logging.info("Starting DAgger training")

    if cfg.seed is not None:
        set_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    aggregated_repo_id, aggregated_root = ensure_aggregated_dataset_ready(cfg.dataset)

    robot = make_robot_from_config(cfg.robot)
    (
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
    ) = make_default_processors()

    expert: ExpertPolicy | None = None
    policy = None
    optimizer = None
    scheduler = None

    try:
        robot.connect()
        expert = make_expert(cfg.expert, robot, teleop_action_processor)
        expert.connect()

        # Load aggregated dataset to infer policy input/output features.
        bootstrap_dataset = LeRobotDataset(aggregated_repo_id, root=aggregated_root)

        expected_features = _build_expected_dataset_features(bootstrap_dataset, robot)
        sanity_check_dataset_robot_compatibility(
            bootstrap_dataset,
            robot,
            cfg.rollout.fps,
            expected_features,
        )

        policy = make_policy(cfg=cfg.policy, ds_meta=bootstrap_dataset.meta)

        params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
        optimizer = cfg.optimizer.build(params)
        if cfg.scheduler:
            total_steps = cfg.training.steps_per_round * cfg.training.rounds
            scheduler = cfg.scheduler.build(optimizer, total_steps)
        else:
            scheduler = None
        grad_clip_norm = cfg.optimizer.grad_clip_norm

        scaler = torch.cuda.amp.GradScaler(enabled=cfg.policy.use_amp and cfg.policy.device == "cuda")

        beta_scheduler = BetaScheduler(cfg.beta, cfg.training.rounds)

        latest_preprocessor = None
        latest_postprocessor = None

        for round_idx in range(cfg.training.rounds):
            beta = beta_scheduler.value(round_idx)
            logging.info("[round %d/%d] beta=%.4f", round_idx + 1, cfg.training.rounds, beta)

            rollout_dataset = LeRobotDataset(
                aggregated_repo_id,
                root=aggregated_root,
                batch_encoding_size=1,
            )
            if cfg.dataset.image_writer_processes or cfg.dataset.image_writer_threads:
                rollout_dataset.start_image_writer(
                    num_processes=cfg.dataset.image_writer_processes,
                    num_threads=cfg.dataset.image_writer_threads,
                )

            rollout_preprocessor, rollout_postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                dataset_stats=rollout_dataset.meta.stats,
            )

            rollout_stats: RolloutStats = collect_dagger_rollouts(
                cfg=cfg.rollout,
                dataset=rollout_dataset,
                robot=robot,
                policy=policy,
                preprocessor=rollout_preprocessor,
                postprocessor=rollout_postprocessor,
                expert=expert,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                beta=beta,
                rng=rng,
                log_dir=cfg.output_dir,
            )

            rollout_dataset.finalize()
            rollout_dataset.stop_image_writer()

            train_dataset = make_training_dataset(
                repo_id=aggregated_repo_id,
                root=aggregated_root,
                policy_cfg=cfg.policy,
            )
            train_preprocessor, train_postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                dataset_stats=train_dataset.meta.stats,
            )

            dataloader = _build_dataloader(cfg, train_dataset)
            dl_iter = cycle(dataloader)

            running_loss = 0.0
            running_grad = 0.0
            running_steps = 0
            latest_train_log: dict[str, float] = {}

            for step_idx in range(cfg.training.steps_per_round):
                batch = next(dl_iter)
                batch = train_preprocessor(batch)

                loss, grad_norm, loss_dict = _compute_train_metrics(
                    policy=policy,
                    optimizer=optimizer,
                    scaler=scaler,
                    batch=batch,
                    grad_clip_norm=grad_clip_norm,
                )
                if scheduler is not None:
                    scheduler.step()

                running_loss += loss
                running_grad += grad_norm
                running_steps += 1
                latest_train_log = loss_dict

                is_log_step = (step_idx + 1) % cfg.training.log_freq == 0
                is_last_step = step_idx + 1 == cfg.training.steps_per_round
                if is_log_step or is_last_step:
                    mean_loss = running_loss / max(1, running_steps)
                    mean_grad = running_grad / max(1, running_steps)
                    logging.info(
                        "[round %d] train step %d/%d loss=%.5f grad=%.5f extra=%s",
                        round_idx + 1,
                        step_idx + 1,
                        cfg.training.steps_per_round,
                        mean_loss,
                        mean_grad,
                        latest_train_log,
                    )
                    running_loss = 0.0
                    running_grad = 0.0
                    running_steps = 0

            latest_preprocessor = train_preprocessor
            latest_postprocessor = train_postprocessor

            if cfg.training.save_checkpoint:
                checkpoint_dir = cfg.output_dir / f"round_{round_idx + 1:03d}"
                round_state = {
                    "round": round_idx + 1,
                    "beta": beta,
                    "rollout": {
                        "episodes": rollout_stats.episodes,
                        "frames": rollout_stats.frames,
                        "expert_exec_steps": rollout_stats.expert_exec_steps,
                        "policy_exec_steps": rollout_stats.policy_exec_steps,
                    },
                    "train_dataset": {
                        "num_frames": train_dataset.num_frames,
                        "num_episodes": train_dataset.num_episodes,
                    },
                    "train_last_log": latest_train_log,
                }
                _save_round_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    policy=policy,
                    preprocessor=train_preprocessor,
                    postprocessor=train_postprocessor,
                    round_state=round_state,
                )

        if latest_preprocessor is not None and latest_postprocessor is not None:
            final_dir = cfg.output_dir / "final"
            _save_round_checkpoint(
                checkpoint_dir=final_dir,
                policy=policy,
                preprocessor=latest_preprocessor,
                postprocessor=latest_postprocessor,
                round_state={"status": "finished", "rounds": cfg.training.rounds},
            )

        logging.info("DAgger training completed. Output dir: %s", cfg.output_dir)

    finally:
        # Mirror the record flow: prefer a safe reset and avoid disconnecting by default,
        # because some robot/teleop stacks treat disconnect as an emergency stop.
        reset_on_finish = getattr(cfg, "reset_on_finish", True)
        disconnect_on_finish = getattr(cfg, "disconnect_on_finish", False)

        if reset_on_finish and robot is not None:
            try:
                robot.reset()
            except Exception:  # noqa: BLE001
                logging.exception("Failed to reset robot cleanly.")

        if disconnect_on_finish:
            if expert is not None:
                try:
                    expert.disconnect()
                except Exception:  # noqa: BLE001
                    logging.exception("Failed to disconnect expert cleanly.")
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                logging.exception("Failed to disconnect robot cleanly.")
        else:
            logging.info("[INFO] Skip expert/robot disconnect on finish to avoid emergency stop.")
