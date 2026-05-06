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

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from lerobot.policies.act.action_delta_utils import (
    _PoseFieldLayout,
    _build_pose_field_layouts,
    _euler_xyz_to_quaternion,
    _quaternion_multiply,
    _quaternion_to_euler_xyz,
    _quaternion_to_rotvec,
    _rotvec_to_quaternion,
)


@dataclass(frozen=True)
class _ChunkReferencePose:
    arm_name: str
    translation: Tensor | None
    rotation: Tensor | None
    rotation_representation: Literal["rotvec", "euler_xyz"] | None


def decode_chunkwise_actions_to_absolute_actions(
    actions: Tensor,
    chunk_ref_state: Tensor,
    action_feature_names: tuple[str, ...],
    observation_state_feature_names: tuple[str, ...],
) -> Tensor:
    """Decode chunk-wise ACT outputs into absolute target actions.

    Each action in the predicted chunk is defined relative to the pose observed at the start of that chunk.
    We therefore recover the reference end-effector pose from `observation.state`, compose the chunk-wise
    delta with that pose, and return an absolute target chunk that downstream execution code can consume
    without changing its feature mapping.
    """
    if actions.ndim != 3:
        raise ValueError(f"`actions` must have shape [B, chunk_size, action_dim]. Got {tuple(actions.shape)}.")
    if chunk_ref_state.ndim != 2:
        raise ValueError(
            "`chunk_ref_state` must have shape [B, state_dim] so chunk-wise inference can decode against the "
            f"current absolute pose. Got {tuple(chunk_ref_state.shape)}."
        )
    if actions.shape[0] != chunk_ref_state.shape[0]:
        raise ValueError(
            "Batch size mismatch between `actions` and `chunk_ref_state`. "
            f"Got {actions.shape[0]} and {chunk_ref_state.shape[0]}."
        )
    if actions.shape[-1] != len(action_feature_names):
        raise ValueError(
            "The number of `action_feature_names` must match the action dimension. "
            f"Got {len(action_feature_names)} names for action_dim={actions.shape[-1]}."
        )
    if chunk_ref_state.shape[-1] != len(observation_state_feature_names):
        raise ValueError(
            "The number of `observation_state_feature_names` must match the state dimension. "
            f"Got {len(observation_state_feature_names)} names for state_dim={chunk_ref_state.shape[-1]}."
        )

    action_layouts = _build_pose_field_layouts(action_feature_names)
    reference_layouts = {
        layout.arm_name: layout for layout in _build_pose_field_layouts(observation_state_feature_names)
    }
    # The action chunk and the reference pose are parsed independently from feature names so we can fail
    # loudly if training-time action channels and inference-time state channels stop matching. This avoids
    # silently composing the wrong pose dimensions together.
    reference_poses = _extract_chunk_reference_poses(chunk_ref_state, action_layouts, reference_layouts)

    decoded = actions.clone()
    for layout in action_layouts:
        reference_pose = reference_poses[layout.arm_name]
        if layout.translation_indices is not None:
            if reference_pose.translation is None:
                raise ValueError(
                    f"Missing translation reference pose for arm '{layout.arm_name}' during chunk-wise inference."
                )
            decoded[:, :, list(layout.translation_indices)] = (
                actions[:, :, list(layout.translation_indices)] + reference_pose.translation.unsqueeze(1)
            )
        if layout.rotation_indices is not None and layout.rotation_representation is not None:
            if reference_pose.rotation is None or reference_pose.rotation_representation is None:
                raise ValueError(
                    f"Missing rotation reference pose for arm '{layout.arm_name}' during chunk-wise inference."
                )
            decoded[:, :, list(layout.rotation_indices)] = _decode_absolute_rotations(
                rotation_chunk=actions[:, :, list(layout.rotation_indices)],
                chunk_ref_rotation=reference_pose.rotation,
                chunk_ref_representation=reference_pose.rotation_representation,
                output_representation=layout.rotation_representation,
            ).to(dtype=actions.dtype)

    return decoded


def _extract_chunk_reference_poses(
    chunk_ref_state: Tensor,
    action_layouts: list[_PoseFieldLayout],
    reference_layouts: dict[str, _PoseFieldLayout],
) -> dict[str, _ChunkReferencePose]:
    reference_poses: dict[str, _ChunkReferencePose] = {}
    for action_layout in action_layouts:
        # We require a real absolute pose source for every arm that the action chunk controls. Intentionally
        # no fallback to previous predictions is allowed here, because that would change the meaning of the
        # chunk reference pose and silently corrupt chunk-wise semantics.
        reference_layout = reference_layouts.get(action_layout.arm_name)
        if reference_layout is None:
            raise ValueError(
                "Chunk-wise ACT inference requires absolute end-effector pose fields in `observation.state`. "
                f"Could not find a reference pose for arm '{action_layout.arm_name}'."
            )

        translation = None
        if action_layout.translation_indices is not None:
            if reference_layout.translation_indices is None:
                raise ValueError(
                    f"Chunk-wise ACT inference is missing translation reference fields for arm "
                    f"'{action_layout.arm_name}'."
                )
            translation = chunk_ref_state[:, list(reference_layout.translation_indices)]

        rotation = None
        rotation_representation = None
        if action_layout.rotation_indices is not None:
            if reference_layout.rotation_indices is None or reference_layout.rotation_representation is None:
                raise ValueError(
                    f"Chunk-wise ACT inference is missing rotation reference fields for arm "
                    f"'{action_layout.arm_name}'."
                )
            rotation = chunk_ref_state[:, list(reference_layout.rotation_indices)]
            rotation_representation = reference_layout.rotation_representation

        reference_poses[action_layout.arm_name] = _ChunkReferencePose(
            arm_name=action_layout.arm_name,
            translation=translation,
            rotation=rotation,
            rotation_representation=rotation_representation,
        )

    return reference_poses


def _decode_absolute_rotations(
    rotation_chunk: Tensor,
    chunk_ref_rotation: Tensor,
    chunk_ref_representation: Literal["rotvec", "euler_xyz"],
    output_representation: Literal["rotvec", "euler_xyz"],
) -> Tensor:
    working_dtype = torch.float64 if rotation_chunk.dtype == torch.float64 else torch.float32
    chunk = rotation_chunk.to(dtype=working_dtype)
    reference = chunk_ref_rotation.to(dtype=working_dtype)
    reference_quaternion = _rotation_to_quaternion(reference, chunk_ref_representation)

    absolute_rotations = []
    for step in range(chunk.shape[1]):
        delta_quaternion = _rotation_to_quaternion(chunk[:, step], output_representation)
        # Match the active server-side pose update: target_quat = delta_quat * current_quat.
        # This is the crucial chunk-wise inference compose step:
        # each predicted rotation is interpreted as "offset from the chunk start pose", so we apply the
        # delta on top of the absolute reference pose for that chunk, not on top of previous predictions.
        absolute_quaternion = _quaternion_multiply(delta_quaternion, reference_quaternion)
        absolute_rotations.append(_quaternion_to_representation(absolute_quaternion, output_representation))

    return torch.stack(absolute_rotations, dim=1)


def _rotation_to_quaternion(rotation: Tensor, representation: Literal["rotvec", "euler_xyz"]) -> Tensor:
    if representation == "rotvec":
        return _rotvec_to_quaternion(rotation)
    return _euler_xyz_to_quaternion(rotation)


def _quaternion_to_representation(
    quaternion: Tensor, representation: Literal["rotvec", "euler_xyz"]
) -> Tensor:
    if representation == "rotvec":
        return _quaternion_to_rotvec(quaternion)
    return _quaternion_to_euler_xyz(quaternion)
