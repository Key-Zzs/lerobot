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
import random
import time
from dataclasses import dataclass
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.dagger.configuration_dagger import DAggerRolloutConfig
from lerobot.policies.dagger.expert import ExpertPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device


@dataclass
class RolloutStats:
    episodes: int = 0
    frames: int = 0
    expert_exec_steps: int = 0
    policy_exec_steps: int = 0


def collect_dagger_rollouts(
    cfg: DAggerRolloutConfig,
    dataset: LeRobotDataset,
    robot: Robot,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    expert: ExpertPolicy,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    beta: float,
    rng: random.Random,
) -> RolloutStats:
    stats = RolloutStats()

    for episode_idx in range(cfg.episodes_per_round):
        logging.info(
            "[rollout] episode %d/%d (beta=%.3f)",
            episode_idx + 1,
            cfg.episodes_per_round,
            beta,
        )
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        start_episode_t = time.perf_counter()
        step_idx = 0

        while True:
            now = time.perf_counter()
            timestamp_s = now - start_episode_t
            if timestamp_s >= cfg.episode_time_s:
                break

            loop_start_t = time.perf_counter()
            raw_obs = robot.get_observation()
            obs_processed = robot_observation_processor(raw_obs)
            obs_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            expert_action = expert.get_action(
                observation=obs_processed,
                raw_observation=raw_obs,
                episode_step=step_idx,
                timestamp_s=timestamp_s,
            )

            use_expert = rng.random() < beta
            if use_expert:
                exec_action = expert_action
                # ACT caches chunked actions; when expert overrides we flush queue to avoid stale chunks.
                policy.reset()
                stats.expert_exec_steps += 1
            else:
                policy_action = predict_action(
                    observation=obs_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.single_task,
                    robot_type=robot.robot_type,
                )
                exec_action = make_robot_action(policy_action, dataset.features)
                stats.policy_exec_steps += 1

            robot_action = robot_action_processor((exec_action, raw_obs))
            _ = robot.send_action(robot_action)

            # DAgger stores expert label for every visited state.
            action_frame = build_dataset_frame(dataset.features, expert_action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": cfg.single_task}
            dataset.add_frame(frame)

            stats.frames += 1
            step_idx += 1

            loop_dt_s = time.perf_counter() - loop_start_t
            busy_wait(1 / cfg.fps - loop_dt_s)

        dataset.save_episode()
        stats.episodes += 1

        if cfg.reset_time_s > 0 and episode_idx < cfg.episodes_per_round - 1:
            _run_reset_phase(
                cfg=cfg,
                robot=robot,
                expert=expert,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

    return stats


def _run_reset_phase(
    cfg: DAggerRolloutConfig,
    robot: Robot,
    expert: ExpertPolicy,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
) -> None:
    if cfg.reset_time_s <= 0:
        return

    start_t = time.perf_counter()
    while True:
        elapsed_s = time.perf_counter() - start_t
        if elapsed_s >= cfg.reset_time_s:
            return

        loop_start_t = time.perf_counter()
        if cfg.reset_with_expert:
            raw_obs = robot.get_observation()
            obs_processed = robot_observation_processor(raw_obs)
            reset_action = expert.get_action(
                observation=obs_processed,
                raw_observation=raw_obs,
                episode_step=-1,
                timestamp_s=elapsed_s,
            )
            robot_action = robot_action_processor((reset_action, raw_obs))
            _ = robot.send_action(robot_action)

        loop_dt_s = time.perf_counter() - loop_start_t
        busy_wait(1 / cfg.fps - loop_dt_s)
