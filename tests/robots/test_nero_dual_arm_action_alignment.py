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

import sys
from pathlib import Path

import numpy as np

TELEOP_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "dual_arm_data_collection" / "lerobot_dual_arm_teleop"
)
if str(TELEOP_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_PACKAGE_ROOT))

from robots.dual_agilx_nero.config_nero import NeroDualArmConfig
from robots.dual_agilx_nero.nero_dual_arm import NeroDualArm


class _ServoRecorder:
    def __init__(self):
        self.calls: list[tuple[str, np.ndarray, bool]] = []

    def servo_p_OL(self, robot_arm: str, pose, delta: bool) -> bool:
        self.calls.append((robot_arm, np.asarray(pose, dtype=float), delta))
        return True


def _make_robot(action_delta_alignment: str = "step_wise") -> NeroDualArm:
    robot = NeroDualArm(
        NeroDualArmConfig(
            debug=False,
            cameras={},
            action_delta_alignment=action_delta_alignment,
        )
    )
    robot._robot = _ServoRecorder()
    robot.is_connected = True
    robot._should_send_action = lambda: True
    return robot


def _cartesian_action() -> dict[str, float]:
    action = {}
    for prefix, base in (("left_delta_ee_pose", 0.1), ("right_delta_ee_pose", 0.2)):
        for axis, offset in zip(("x", "y", "z", "rx", "ry", "rz"), range(6), strict=True):
            action[f"{prefix}.{axis}"] = base + offset
    return action


def test_send_action_cartesian_stepwise_uses_delta_true():
    robot = _make_robot("step_wise")

    robot.send_action_cartesian(_cartesian_action())

    assert len(robot._robot.calls) == 2
    assert robot._robot.calls[0][0] == "left_robot"
    assert robot._robot.calls[0][2] is True
    assert robot._robot.calls[1][0] == "right_robot"
    assert robot._robot.calls[1][2] is True


def test_send_action_cartesian_chunkwise_uses_delta_false_even_with_delta_feature_names():
    robot = _make_robot("chunk_wise")

    robot.send_action_cartesian(_cartesian_action())

    assert len(robot._robot.calls) == 2
    assert robot._robot.calls[0][2] is False
    assert robot._robot.calls[1][2] is False


def test_send_action_defaults_to_stepwise_when_not_configured():
    robot = NeroDualArm(NeroDualArmConfig(debug=False, cameras={}))
    robot._robot = _ServoRecorder()
    robot.is_connected = True
    robot._should_send_action = lambda: True

    robot.send_action_cartesian(_cartesian_action())

    assert len(robot._robot.calls) == 2
    assert robot._robot.calls[0][2] is True
    assert robot._robot.calls[1][2] is True


def test_send_action_forwards_mode_to_cartesian_execution():
    robot = _make_robot("chunk_wise")

    returned_action = robot.send_action(_cartesian_action())

    assert returned_action["left_delta_ee_pose.x"] == 0.1
    assert len(robot._robot.calls) == 2
    assert robot._robot.calls[0][2] is False
    assert robot._robot.calls[1][2] is False
