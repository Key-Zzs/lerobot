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
import time
from dataclasses import dataclass
from pathlib import Path
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
from lerobot.utils.control_utils import is_headless, predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device


@dataclass
class RolloutStats:
    episodes: int = 0
    frames: int = 0
    expert_exec_steps: int = 0
    policy_exec_steps: int = 0


def _init_dagger_keyboard_listener():
    """Non-blocking keyboard listener for DAgger rollout control."""
    events = {
        "stop_rollout": False,
        "terminate_episode": False,
        "discard_episode": False,
        "force_expert": False,
        "reset_to_home": False,
    }

    if is_headless():
        logging.warning("Headless environment detected. DAgger keyboard controls are disabled.")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow → 结束当前episode并保存数据")
                events["terminate_episode"] = True
            elif key == keyboard.Key.left:
                print("Left arrow ← 丢弃当前episode")
                events["discard_episode"] = True
                events["terminate_episode"] = True
            elif key == keyboard.Key.up:
                events["force_expert"] = True
                print("[FORCE EXPERT MODE] 强制专家模式")
            elif key == keyboard.Key.down:
                events["force_expert"] = False
                print("[MIXED MODE] 混合模式")
            elif key == keyboard.Key.space:
                print("Space 空格键 → 机械臂回到home位置")
                events["reset_to_home"] = True
            elif key == keyboard.Key.enter:
                print("Enter 回车键 → 开始新episode")
                events["start_new_episode"] = True
            elif key == keyboard.Key.esc:
                print("Escape 退出键 → 停止DAgger训练")
                events["stop_rollout"] = True
                events["terminate_episode"] = True
            elif hasattr(key, 'char') and key.char == 'h':
                print("\n=== DAgger 键盘控制说明 ===")
                print("→ 右箭头: 结束当前episode并保存数据")
                print("← 左箭头: 丢弃当前episode")
                print("↑ 上箭头: 强制专家模式")
                print("↓ 下箭头: 混合模式")
                print("空格键: 机械臂回到home位置")
                print("回车键: 开始新episode")
                print("Esc键: 停止DAgger训练")
                print("h键: 显示帮助")
                print("==========================\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Error handling DAgger key press: {exc}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def _prompt_episode_success() -> bool:
    while True:
        ans = input("Was this episode successful? (y/n): ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def _append_episode_log(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    log_dir: Path | None = None,
) -> RolloutStats:
    stats = RolloutStats()
    listener, events = _init_dagger_keyboard_listener()
    episode_log_path = (log_dir / "dagger_episode_logs.jsonl") if log_dir is not None else None

    try:
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
            last_mode_msg: str | None = None
            episode_action_source: list[str] = []
            episode_beta: list[float] = []
            episode_terminated_early = False
            episode_discarded = False
            episode_success: bool | None = None

            while True:
                # 处理键盘事件
                if events["stop_rollout"]:
                    episode_terminated_early = True
                    episode_discarded = True
                    break
                if events["terminate_episode"]:
                    episode_terminated_early = True
                    break
                if events["reset_to_home"]:
                    print("执行机械臂回到home位置...")
                    # 调用机械臂的reset功能
                    robot.reset()
                    events["reset_to_home"] = False
                    print("机械臂已回到home位置")

                now = time.perf_counter()
                timestamp_s = now - start_episode_t
                # 使用键盘控制替代固定时间限制
                if timestamp_s >= cfg.episode_time_s:
                    logging.info("Episode时间超时，自动结束")
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

                if events["force_expert"]:
                    use_expert = True
                    mode_msg = "[FORCE EXPERT MODE]"
                else:
                    use_expert = rng.random() < beta
                    mode_msg = f"[MIXED MODE beta={beta:.3f}]"

                if mode_msg != last_mode_msg:
                    logging.info(mode_msg)
                    last_mode_msg = mode_msg

                if use_expert:
                    exec_action = expert_action
                    # ACT caches chunked actions; when expert overrides we flush queue to avoid stale chunks.
                    policy.reset()
                    stats.expert_exec_steps += 1
                    action_source = "expert"
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
                    action_source = "policy"

                robot_action = robot_action_processor((exec_action, raw_obs))
                _ = robot.send_action(robot_action)

                episode_action_source.append(action_source)
                episode_beta.append(float(beta))

                # DAgger stores expert label for every visited state.
                action_frame = build_dataset_frame(dataset.features, expert_action, prefix=ACTION)
                frame = {**obs_frame, **action_frame, "task": cfg.single_task}
                dataset.add_frame(frame)

                stats.frames += 1
                step_idx += 1

                loop_dt_s = time.perf_counter() - loop_start_t
                busy_wait(1 / cfg.fps - loop_dt_s)

            if not episode_discarded and not events["stop_rollout"]:
                episode_success = _prompt_episode_success()

            if episode_log_path is not None:
                _append_episode_log(
                    episode_log_path,
                    {
                        "episode_index": episode_idx + 1,
                        "success": episode_success,
                        "discarded": episode_discarded,
                        "terminated_early": episode_terminated_early,
                        "frames": len(episode_action_source),
                        "action_source": episode_action_source,
                        "beta": episode_beta,
                        "beta_current": beta,
                        "saved": not episode_discarded and not events["stop_rollout"],
                    },
                )

            if episode_discarded:
                logging.info("[rollout] episode %d discarded.", episode_idx + 1)
                dataset.clear_episode_buffer()
            elif not events["stop_rollout"]:
                dataset.save_episode()
                stats.episodes += 1

            if events["stop_rollout"]:
                break

            events["terminate_episode"] = False
            events["discard_episode"] = False

            if cfg.reset_time_s > 0 and episode_idx < cfg.episodes_per_round - 1:
                _run_reset_phase(
                    cfg=cfg,
                    robot=robot,
                    expert=expert,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    events=events,
                )

        return stats
    finally:
        if listener is not None:
            listener.stop()


def _run_reset_phase(
    cfg: DAggerRolloutConfig,
    robot: Robot,
    expert: ExpertPolicy,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    events: dict[str, bool] | None = None,
) -> None:
    if cfg.reset_time_s <= 0:
        return

    start_t = time.perf_counter()
    while True:
        if events and (events.get("stop_rollout") or events.get("terminate_episode")):
            return

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