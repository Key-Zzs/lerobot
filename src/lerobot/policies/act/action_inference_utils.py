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
    convert_stepwise_to_chunkwise_actions,
)

_CANONICAL_POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


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
    observation_state_pose_axis_order: tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz"),
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
    effective_observation_state_feature_names = _remap_observation_state_feature_names(
        observation_state_feature_names,
        observation_state_pose_axis_order,
    )
    reference_layouts = {
        layout.arm_name: layout for layout in _build_pose_field_layouts(effective_observation_state_feature_names)
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


def inspect_chunkwise_feature_mapping(
    action_feature_names: tuple[str, ...],
    observation_state_feature_names: tuple[str, ...],
    observation_state_pose_axis_order: tuple[str, ...] = _CANONICAL_POSE_AXES,
    *,
    require_right_arm: bool = False,
) -> dict:
    """Validate and summarize the feature indices used by chunk-wise ACT decode."""
    _raise_on_duplicate_names("action_feature_names", action_feature_names)
    _raise_on_duplicate_names("observation_state_feature_names", observation_state_feature_names)
    if tuple(observation_state_pose_axis_order) != _CANONICAL_POSE_AXES and len(
        observation_state_pose_axis_order
    ) != len(_CANONICAL_POSE_AXES):
        raise ValueError(
            "`observation_state_pose_axis_order` must describe [x, y, z, rx, ry, rz]. "
            f"Got {observation_state_pose_axis_order}."
        )

    action_layouts = _build_pose_field_layouts(action_feature_names)
    effective_observation_state_feature_names = _remap_observation_state_feature_names(
        observation_state_feature_names,
        observation_state_pose_axis_order,
    )
    observation_layouts = {
        layout.arm_name: layout for layout in _build_pose_field_layouts(effective_observation_state_feature_names)
    }

    arms: dict[str, dict] = {}
    for action_layout in action_layouts:
        observation_layout = observation_layouts.get(action_layout.arm_name)
        if observation_layout is None:
            raise ValueError(
                f"Missing observation.state ee pose slice for arm '{action_layout.arm_name}' during "
                "chunk-wise feature mapping check."
            )
        arms[action_layout.arm_name] = {
            "action": _layout_axis_mapping(action_layout, action_feature_names),
            "observation_state": _layout_axis_mapping(
                observation_layout,
                observation_state_feature_names,
                effective_feature_names=effective_observation_state_feature_names,
            ),
        }

    if require_right_arm and ("right" not in arms or "z" not in arms["right"]["action"]):
        raise ValueError(
            "Chunk-wise right-arm z debugging requires `right_delta_ee_pose.z` in action_feature_names."
        )

    return {
        "canonical_pose_order": list(_CANONICAL_POSE_AXES),
        "observation_state_pose_axis_order": list(observation_state_pose_axis_order),
        "action_feature_names": list(action_feature_names),
        "observation_state_feature_names": list(observation_state_feature_names),
        "effective_observation_state_feature_names": list(effective_observation_state_feature_names),
        "arms": arms,
    }


def check_chunkwise_train_decode_inverse(
    stepwise_actions: Tensor,
    chunk_ref_state: Tensor,
    action_feature_names: tuple[str, ...],
    observation_state_feature_names: tuple[str, ...],
    observation_state_pose_axis_order: tuple[str, ...] = _CANONICAL_POSE_AXES,
) -> dict:
    """Check that training conversion and inference decode agree for a step-wise delta chunk.

    This does not run the model. It uses the same feature names and chunk reference pose that deployment uses,
    converts a step-wise delta chunk into chunk-wise labels, decodes those labels back to absolute targets, and
    compares them to a direct step-wise integration from the same reference pose.
    """
    if stepwise_actions.ndim != 3:
        raise ValueError(
            f"`stepwise_actions` must have shape [B, chunk_size, action_dim]. Got {tuple(stepwise_actions.shape)}."
        )
    if chunk_ref_state.ndim == 1:
        chunk_ref_state = chunk_ref_state.unsqueeze(0)
    chunk_ref_state = chunk_ref_state.to(device=stepwise_actions.device, dtype=stepwise_actions.dtype)

    mapping = inspect_chunkwise_feature_mapping(
        action_feature_names,
        observation_state_feature_names,
        observation_state_pose_axis_order,
        require_right_arm=True,
    )
    chunkwise_delta = convert_stepwise_to_chunkwise_actions(stepwise_actions, action_feature_names)
    decoded_absolute = decode_chunkwise_actions_to_absolute_actions(
        chunkwise_delta,
        chunk_ref_state=chunk_ref_state,
        action_feature_names=action_feature_names,
        observation_state_feature_names=observation_state_feature_names,
        observation_state_pose_axis_order=observation_state_pose_axis_order,
    )
    reference_absolute = _integrate_stepwise_actions_to_absolute_actions(
        stepwise_actions,
        chunk_ref_state=chunk_ref_state,
        action_feature_names=action_feature_names,
        observation_state_feature_names=observation_state_feature_names,
        observation_state_pose_axis_order=observation_state_pose_axis_order,
    )

    error = decoded_absolute - reference_absolute
    for layout in _build_pose_field_layouts(action_feature_names):
        if layout.rotation_indices is not None and layout.rotation_representation == "euler_xyz":
            error[:, :, list(layout.rotation_indices)] = _wrap_angle_delta(
                error[:, :, list(layout.rotation_indices)]
            )

    right_z_index = mapping["arms"]["right"]["action"]["z"]["index"]
    right_z = {
        "feature_name": action_feature_names[right_z_index],
        "index": right_z_index,
        "stepwise_delta": _tensor_trace(stepwise_actions[0, :, right_z_index]),
        "chunkwise_delta": _tensor_trace(chunkwise_delta[0, :, right_z_index]),
        "decoded_absolute": _tensor_trace(decoded_absolute[0, :, right_z_index]),
        "reference_absolute": _tensor_trace(reference_absolute[0, :, right_z_index]),
        "error": _tensor_trace(error[0, :, right_z_index]),
    }
    return {
        "ok": bool(torch.max(torch.abs(error)).item() < 1e-6),
        "max_abs_error": float(torch.max(torch.abs(error)).item()),
        "right_z": right_z,
    }


def _remap_observation_state_feature_names(
    observation_state_feature_names: tuple[str, ...],
    observation_state_pose_axis_order: tuple[str, ...],
) -> tuple[str, ...]:
    canonical_pose_axes = ("x", "y", "z", "rx", "ry", "rz")
    if tuple(observation_state_pose_axis_order) == canonical_pose_axes:
        return observation_state_feature_names
    if len(observation_state_pose_axis_order) != len(canonical_pose_axes):
        raise ValueError(
            "`observation_state_pose_axis_order` must describe 6 ee-pose axes. "
            f"Got {observation_state_pose_axis_order}."
        )

    stored_axis_to_semantic_axis = {
        stored_axis: semantic_axis
        for semantic_axis, stored_axis in zip(canonical_pose_axes, observation_state_pose_axis_order, strict=True)
    }
    remapped_names: list[str] = []
    for feature_name in observation_state_feature_names:
        for stored_axis, semantic_axis in stored_axis_to_semantic_axis.items():
            suffix = f".{stored_axis}"
            if feature_name.endswith(suffix):
                remapped_names.append(feature_name[: -len(suffix)] + f".{semantic_axis}")
                break
        else:
            remapped_names.append(feature_name)

    return tuple(remapped_names)


def _raise_on_duplicate_names(label: str, names: tuple[str, ...]) -> None:
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Duplicate entries in `{label}`: {duplicate_names}.")


def _layout_axis_mapping(
    layout: _PoseFieldLayout,
    feature_names: tuple[str, ...],
    *,
    effective_feature_names: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    if layout.translation_indices is not None:
        for axis, index in zip(("x", "y", "z"), layout.translation_indices, strict=True):
            mapping[axis] = _feature_entry(index, feature_names, effective_feature_names)
    if layout.rotation_indices is not None:
        rotation_axes = ("rx", "ry", "rz") if layout.rotation_representation == "euler_xyz" else ("wx", "wy", "wz")
        for axis, index in zip(rotation_axes, layout.rotation_indices, strict=True):
            mapping[axis] = _feature_entry(index, feature_names, effective_feature_names)
    _raise_on_duplicate_indices(layout.arm_name, mapping)
    return mapping


def _feature_entry(
    index: int,
    feature_names: tuple[str, ...],
    effective_feature_names: tuple[str, ...] | None,
) -> dict:
    entry = {"index": int(index), "feature_name": feature_names[index]}
    if effective_feature_names is not None:
        entry["effective_feature_name"] = effective_feature_names[index]
    return entry


def _raise_on_duplicate_indices(arm_name: str, mapping: dict[str, dict]) -> None:
    indices = [entry["index"] for entry in mapping.values()]
    duplicate_indices = sorted({index for index in indices if indices.count(index) > 1})
    if duplicate_indices:
        raise ValueError(f"Duplicate pose indices for arm '{arm_name}': {duplicate_indices}.")


def _tensor_trace(values: Tensor, limit: int = 12) -> list[float]:
    return [round(float(value), 6) for value in values.detach().cpu()[:limit].tolist()]


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


def _integrate_stepwise_actions_to_absolute_actions(
    stepwise_actions: Tensor,
    chunk_ref_state: Tensor,
    action_feature_names: tuple[str, ...],
    observation_state_feature_names: tuple[str, ...],
    observation_state_pose_axis_order: tuple[str, ...],
) -> Tensor:
    action_layouts = _build_pose_field_layouts(action_feature_names)
    effective_observation_state_feature_names = _remap_observation_state_feature_names(
        observation_state_feature_names,
        observation_state_pose_axis_order,
    )
    reference_layouts = {
        layout.arm_name: layout for layout in _build_pose_field_layouts(effective_observation_state_feature_names)
    }
    reference_poses = _extract_chunk_reference_poses(chunk_ref_state, action_layouts, reference_layouts)

    absolute = stepwise_actions.clone()
    for layout in action_layouts:
        reference_pose = reference_poses[layout.arm_name]
        if layout.translation_indices is not None:
            absolute[:, :, list(layout.translation_indices)] = (
                reference_pose.translation.unsqueeze(1)
                + stepwise_actions[:, :, list(layout.translation_indices)].cumsum(dim=1)
            )
        if layout.rotation_indices is not None and layout.rotation_representation is not None:
            rotation_steps = stepwise_actions[:, :, list(layout.rotation_indices)]
            current_quaternion = _rotation_to_quaternion(
                reference_pose.rotation,
                reference_pose.rotation_representation,
            )
            absolute_rotations = []
            for step in range(rotation_steps.shape[1]):
                delta_quaternion = _rotation_to_quaternion(rotation_steps[:, step], layout.rotation_representation)
                current_quaternion = _quaternion_multiply(delta_quaternion, current_quaternion)
                absolute_rotations.append(
                    _quaternion_to_representation(current_quaternion, layout.rotation_representation)
                )
            absolute[:, :, list(layout.rotation_indices)] = torch.stack(absolute_rotations, dim=1).to(
                dtype=stepwise_actions.dtype
            )
    return absolute


def _wrap_angle_delta(angle_delta: Tensor) -> Tensor:
    return (angle_delta + torch.pi) % (2.0 * torch.pi) - torch.pi


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
