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
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TELEOP_PACKAGE_ROOT = REPO_ROOT / "dual_arm_data_collection" / "lerobot_dual_arm_teleop"
for package_root in (SRC_ROOT, TELEOP_PACKAGE_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from robots.dual_agilx_nero.config_nero import NeroDualArmConfig
from robots.dual_agilx_nero.nero_dual_arm import NeroDualArm


class _ServoRecorder:
    def __init__(self, left_pose=None, right_pose=None):
        self.calls: list[tuple[str, np.ndarray, bool]] = []
        self.left_pose = None if left_pose is None else np.asarray(left_pose, dtype=float)
        self.right_pose = None if right_pose is None else np.asarray(right_pose, dtype=float)

    def servo_p_OL(self, robot_arm: str, pose, delta: bool) -> bool:
        self.calls.append((robot_arm, np.asarray(pose, dtype=float), delta))
        return True

    def left_robot_get_ee_pose(self):
        return self.left_pose

    def right_robot_get_ee_pose(self):
        return self.right_pose

    def left_robot_get_servo_p_ol_reference_pose(self):
        return self.left_pose

    def right_robot_get_servo_p_ol_reference_pose(self):
        return self.right_pose


def _make_robot(
    action_delta_alignment: str = "step_wise",
    left_ref=None,
    right_ref=None,
    chunkwise_reference_pose_source: str = "servo_ol",
) -> NeroDualArm:
    robot = NeroDualArm(
        NeroDualArmConfig(
            debug=False,
            cameras={},
            action_delta_alignment=action_delta_alignment,
            chunkwise_reference_pose_source=chunkwise_reference_pose_source,
        )
    )
    robot._robot = _ServoRecorder(left_pose=left_ref, right_pose=right_ref)
    robot.is_connected = True
    robot._should_send_action = lambda: True
    return robot


def _set_prev_observation(robot: NeroDualArm, left_pose, right_pose) -> None:
    canonical_axes = ("x", "y", "z", "rx", "ry", "rz")
    stored_axes = tuple(robot.config.ee_pose_observation_axis_order)
    observation = {}
    for arm_side, pose in (("left", left_pose), ("right", right_pose)):
        for _semantic_axis, stored_axis, value in zip(canonical_axes, stored_axes, pose, strict=True):
            observation[f"{arm_side}_ee_pose.{stored_axis}"] = float(value)
    robot._prev_observation = observation


def _cartesian_action(left_pose=None, right_pose=None) -> dict[str, float]:
    action = {}
    if left_pose is None:
        left_pose = [0.1 + offset for offset in range(6)]
    if right_pose is None:
        right_pose = [0.2 + offset for offset in range(6)]
    for prefix, pose in (("left_delta_ee_pose", left_pose), ("right_delta_ee_pose", right_pose)):
        for axis, value in zip(("x", "y", "z", "rx", "ry", "rz"), pose, strict=True):
            action[f"{prefix}.{axis}"] = float(value)
    return action


def test_send_action_cartesian_stepwise_uses_delta_true():
    robot = _make_robot("step_wise")

    robot.send_action_cartesian(_cartesian_action())

    assert len(robot._robot.calls) == 2
    assert robot._robot.calls[0][0] == "left_robot"
    assert robot._robot.calls[0][2] is True
    assert robot._robot.calls[1][0] == "right_robot"
    assert robot._robot.calls[1][2] is True


def test_send_action_cartesian_chunkwise_converts_absolute_target_to_delta_true():
    robot = _make_robot(
        "chunk_wise",
        left_ref=[10.0, 20.0, 30.0, 0.1, 0.2, 0.3],
        right_ref=[100.0, 200.0, 300.0, -0.2, 0.4, -0.6],
    )
    action = _cartesian_action(
        left_pose=[11.0, 22.0, 33.0, 0.1, 0.2, 0.3],
        right_pose=[101.0, 202.0, 303.0, -0.2, 0.4, -0.6],
    )

    robot.send_action_cartesian(action)

    assert len(robot._robot.calls) == 2
    assert robot._robot.calls[0][2] is True
    assert robot._robot.calls[1][2] is True
    np.testing.assert_allclose(robot._robot.calls[0][1][:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(robot._robot.calls[1][1][:3], [1.0, 2.0, 3.0])
    assert action["left_delta_ee_pose.x"] == 11.0


def test_send_action_cartesian_chunkwise_defaults_to_servo_ol_reference_not_observation():
    robot = _make_robot(
        "chunk_wise",
        left_ref=[10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
        right_ref=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
    )
    _set_prev_observation(
        robot,
        left_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        right_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    robot.send_action_cartesian(
        _cartesian_action(
            left_pose=[11.0, 22.0, 33.0, 0.0, 0.0, 0.0],
            right_pose=[101.0, 202.0, 303.0, 0.0, 0.0, 0.0],
        )
    )

    np.testing.assert_allclose(robot._robot.calls[0][1][:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(robot._robot.calls[1][1][:3], [1.0, 2.0, 3.0])


def test_send_action_cartesian_chunkwise_can_use_observation_reference_when_configured():
    robot = _make_robot("chunk_wise", chunkwise_reference_pose_source="observation")
    _set_prev_observation(
        robot,
        left_pose=[10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
        right_pose=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
    )

    robot.send_action_cartesian(
        _cartesian_action(
            left_pose=[11.0, 22.0, 33.0, 0.0, 0.0, 0.0],
            right_pose=[101.0, 202.0, 303.0, 0.0, 0.0, 0.0],
        )
    )

    np.testing.assert_allclose(robot._robot.calls[0][1][:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(robot._robot.calls[1][1][:3], [1.0, 2.0, 3.0])


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
    robot = _make_robot(
        "chunk_wise",
        left_ref=[10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
        right_ref=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
    )
    action = _cartesian_action(
        left_pose=[10.1, 20.2, 30.3, 0.0, 0.0, 0.0],
        right_pose=[100.2, 200.3, 300.4, 0.0, 0.0, 0.0],
    )

    returned_action = robot.send_action(action)

    assert returned_action["left_delta_ee_pose.x"] == 10.1
    assert len(robot._robot.calls) == 2
    assert robot._robot.calls[0][2] is True
    assert robot._robot.calls[1][2] is True
    np.testing.assert_allclose(robot._robot.calls[0][1][:3], [0.1, 0.2, 0.3])


def test_send_action_cartesian_chunkwise_requires_current_reference_pose(caplog):
    robot = _make_robot("chunk_wise")

    with caplog.at_level(logging.WARNING):
        robot.send_action_cartesian(_cartesian_action())

    assert robot._robot.calls == []
    assert "current execution reference pose" in caplog.text


def test_convert_absolute_pose_to_servo_delta_inverts_server_rotation_compose():
    current_pose = np.array([1.0, 2.0, 3.0, 0.2, -0.1, 0.3])
    expected_delta = np.array([0.0, 0.0, 0.0, 0.05, -0.02, 0.03])

    current_quat = NeroDualArm._euler_xyz_to_quaternion(current_pose[3:])
    delta_quat = NeroDualArm._euler_xyz_to_quaternion(expected_delta[3:])
    target_quat = NeroDualArm._quaternion_multiply(delta_quat, current_quat)
    target_pose = current_pose.copy()
    target_pose[3:] = NeroDualArm._quaternion_to_euler_xyz(target_quat)

    recovered_delta = NeroDualArm._convert_absolute_pose_to_servo_delta(target_pose, current_pose)

    np.testing.assert_allclose(recovered_delta[3:], expected_delta[3:], atol=1e-6)
