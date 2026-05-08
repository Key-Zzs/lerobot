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

import re
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


ACT_CHUNKWISE_LABELS_CONVERTED_KEY = "_act_chunkwise_action_labels_converted"


@dataclass(frozen=True)
class _PoseFieldLayout:
    arm_name: str
    translation_indices: tuple[int, int, int] | None = None
    rotation_indices: tuple[int, int, int] | None = None
    rotation_representation: Literal["rotvec", "euler_xyz"] | None = None


def convert_stepwise_to_chunkwise_actions(actions: Tensor, action_feature_names: tuple[str, ...]) -> Tensor:
    """Convert step-wise action deltas into chunk-wise deltas for ACT training labels.

    Translation channels are accumulated with a cumulative sum. Rotation channels are accumulated with
    quaternion composition so we do not rely on naive per-axis addition. Features that are not recognized as
    end-effector pose deltas, such as gripper commands, are preserved as-is.
    """
    if actions.ndim != 3:
        raise ValueError(f"`actions` must have shape [B, chunk_size, action_dim]. Got {tuple(actions.shape)}.")
    if actions.shape[-1] != len(action_feature_names):
        raise ValueError(
            "The number of `action_feature_names` must match the action dimension. "
            f"Got {len(action_feature_names)} names for action_dim={actions.shape[-1]}."
        )

    layouts = _build_pose_field_layouts(action_feature_names)
    converted = actions.clone()

    for layout in layouts:
        if layout.translation_indices is not None:
            # Chunk-wise xyz means "offset from the chunk start after repeatedly applying step-wise deltas".
            converted[:, :, list(layout.translation_indices)] = actions[:, :, list(layout.translation_indices)].cumsum(
                dim=1
            )
        if layout.rotation_indices is not None and layout.rotation_representation is not None:
            # Rotation cannot be accumulated component-wise in a stable way. We instead compose per-step
            # rotations in quaternion space and convert back to the original representation expected by labels.
            rotation_chunk = actions[:, :, list(layout.rotation_indices)]
            converted[:, :, list(layout.rotation_indices)] = _accumulate_rotation_chunk(
                rotation_chunk, layout.rotation_representation
            ).to(dtype=actions.dtype)

    return converted


def _build_pose_field_layouts(action_feature_names: tuple[str, ...]) -> list[_PoseFieldLayout]:
    grouped_fields: dict[str, dict[str, dict[str, int]]] = {}

    for index, feature_name in enumerate(action_feature_names):
        parsed = _parse_pose_feature_name(feature_name)
        if parsed is None:
            continue

        arm_name, field_group, axis_name = parsed
        grouped_fields.setdefault(arm_name, {}).setdefault(field_group, {})[axis_name] = index

    layouts: list[_PoseFieldLayout] = []
    for arm_name, arm_fields in grouped_fields.items():
        # We build one layout per arm so left/right pose channels are accumulated independently.
        translation_indices = _ordered_indices(arm_fields.get("translation"), ("x", "y", "z"))

        rotation_indices = None
        rotation_representation = None
        if "rotvec" in arm_fields and "euler_xyz" in arm_fields:
            raise ValueError(
                f"Found both rotvec-like and euler-like rotation fields for arm '{arm_name}'. "
                "Please provide a single rotation representation."
            )
        if "rotvec" in arm_fields:
            rotation_indices = _ordered_indices(arm_fields["rotvec"], ("x", "y", "z"))
            rotation_representation = "rotvec"
        elif "euler_xyz" in arm_fields:
            rotation_indices = _ordered_indices(arm_fields["euler_xyz"], ("roll", "pitch", "yaw"))
            rotation_representation = "euler_xyz"

        if translation_indices is None and rotation_indices is None:
            continue

        layouts.append(
            _PoseFieldLayout(
                arm_name=arm_name,
                translation_indices=translation_indices,
                rotation_indices=rotation_indices,
                rotation_representation=rotation_representation,
            )
        )

    if not layouts:
        raise ValueError(
            "Unable to identify end-effector pose delta fields from `action_feature_names`. "
            "Expected names such as `left_delta_ee_pose.x`, `right_delta_ee_pose.rx`, `ee.wx`, or similar."
        )

    return layouts


def _ordered_indices(
    field_indices: dict[str, int] | None, expected_axes: tuple[str, str, str]
) -> tuple[int, int, int] | None:
    if field_indices is None:
        return None

    missing_axes = [axis for axis in expected_axes if axis not in field_indices]
    if missing_axes:
        raise ValueError(f"Missing pose axes {missing_axes} for fields {field_indices}.")

    return tuple(field_indices[axis] for axis in expected_axes)


def _parse_pose_feature_name(feature_name: str) -> tuple[str, str, str] | None:
    normalized_name = feature_name.lower()
    if "gripper" in normalized_name:
        # Gripper labels are intentionally left step-wise unless a future dataset explicitly encodes them
        # as deltas too.
        return None

    arm_name = "single_arm"
    for prefix, parsed_arm_name in (
        ("left_", "left"),
        ("right_", "right"),
        ("left.", "left"),
        ("right.", "right"),
        ("l_", "left"),
        ("r_", "right"),
    ):
        if normalized_name.startswith(prefix):
            arm_name = parsed_arm_name
            normalized_name = normalized_name[len(prefix) :]
            break

    name_parts = [part for part in re.split(r"[./]", normalized_name) if part]
    if len(name_parts) < 2:
        return None

    descriptor = ".".join(name_parts[:-1])
    axis_name = name_parts[-1]
    if "ee" not in descriptor and "pose" not in descriptor:
        return None

    if axis_name in {"x", "y", "z"}:
        return arm_name, "translation", axis_name
    if axis_name in {"wx", "wy", "wz"}:
        return arm_name, "rotvec", axis_name[1:]
    if axis_name in {"rx", "ry", "rz"}:
        # Align ACT training with the active server-side `servo_p_OL` implementation, which interprets
        # rx/ry/rz as incremental roll/pitch/yaw values before converting them to quaternions.
        return arm_name, "euler_xyz", {"rx": "roll", "ry": "pitch", "rz": "yaw"}[axis_name]
    if axis_name in {"roll", "pitch", "yaw"}:
        return arm_name, "euler_xyz", axis_name

    return None


def _accumulate_rotation_chunk(rotation_chunk: Tensor, representation: Literal["rotvec", "euler_xyz"]) -> Tensor:
    working_dtype = torch.float64 if rotation_chunk.dtype == torch.float64 else torch.float32
    chunk = rotation_chunk.to(dtype=working_dtype)
    batch_size, chunk_size, _ = chunk.shape

    # Identity rotation in [x, y, z, w] form. Each step left-multiplies the new delta, matching the active
    # server-side `servo_p_OL` update rule: target_quat = delta_quat * current_quat.
    cumulative_quaternion = torch.zeros(batch_size, 4, dtype=working_dtype, device=chunk.device)
    cumulative_quaternion[:, 3] = 1.0
    cumulative_rotations = []

    for step in range(chunk_size):
        step_delta = chunk[:, step]
        if representation == "rotvec":
            delta_quaternion = _rotvec_to_quaternion(step_delta)
        else:
            delta_quaternion = _euler_xyz_to_quaternion(step_delta)
        cumulative_quaternion = _quaternion_multiply(delta_quaternion, cumulative_quaternion)

        if representation == "rotvec":
            cumulative_rotations.append(_quaternion_to_rotvec(cumulative_quaternion))
        else:
            cumulative_rotations.append(_quaternion_to_euler_xyz(cumulative_quaternion))

    return torch.stack(cumulative_rotations, dim=1)


def _rotvec_to_quaternion(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half_angle = angle * 0.5

    small_angle = angle < 1e-8
    scale = torch.where(small_angle, torch.full_like(angle, 0.5), torch.sin(half_angle) / angle)

    quaternion = torch.cat([rotvec * scale, torch.cos(half_angle)], dim=-1)
    return _normalize_quaternion(quaternion)


def _euler_xyz_to_quaternion(euler_xyz: Tensor) -> Tensor:
    # Compose intrinsic xyz Euler increments as qz * qy * qx so the resulting quaternion matches the
    # helper used by the server-side controller for roll / pitch / yaw inputs.
    half_angles = euler_xyz * 0.5
    cx, cy, cz = torch.cos(half_angles).unbind(dim=-1)
    sx, sy, sz = torch.sin(half_angles).unbind(dim=-1)

    qx = torch.stack([sx, torch.zeros_like(sx), torch.zeros_like(sx), cx], dim=-1)
    qy = torch.stack([torch.zeros_like(sy), sy, torch.zeros_like(sy), cy], dim=-1)
    qz = torch.stack([torch.zeros_like(sz), torch.zeros_like(sz), sz, cz], dim=-1)

    return _normalize_quaternion(_quaternion_multiply(qz, _quaternion_multiply(qy, qx)))


def _quaternion_to_rotvec(quaternion: Tensor) -> Tensor:
    quaternion = _normalize_quaternion(quaternion)
    quaternion = torch.where(quaternion[..., 3:].lt(0), -quaternion, quaternion)

    xyz = quaternion[..., :3]
    w = quaternion[..., 3:].clamp(min=-1.0, max=1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half < 1e-8, torch.full_like(sin_half, 2.0), angle / sin_half)
    return xyz * scale


def _quaternion_to_euler_xyz(quaternion: Tensor) -> Tensor:
    quaternion = _normalize_quaternion(quaternion)
    x, y, z, w = quaternion.unbind(dim=-1)

    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(min=-1.0, max=1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return torch.stack([roll, pitch, yaw], dim=-1)


def _quaternion_multiply(lhs: Tensor, rhs: Tensor) -> Tensor:
    # Returns `lhs * rhs` with quaternions stored as [x, y, z, w].
    # The variable names unpack `rhs` first only to keep the scalar formula compact; the multiplication
    # order itself still matches the server helper `quat_multiply(q1, q2) -> q1 * q2`.
    x1, y1, z1, w1 = rhs.unbind(dim=-1)
    x2, y2, z2, w2 = lhs.unbind(dim=-1)

    quaternion = torch.stack(
        [
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        ],
        dim=-1,
    )
    return _normalize_quaternion(quaternion)


def _normalize_quaternion(quaternion: Tensor) -> Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
