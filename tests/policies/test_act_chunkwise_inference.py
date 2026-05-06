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

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


ACTION_FEATURE_NAMES = (
    "left_delta_ee_pose.x",
    "left_delta_ee_pose.y",
    "left_delta_ee_pose.z",
    "left_delta_ee_pose.rx",
    "left_delta_ee_pose.ry",
    "left_delta_ee_pose.rz",
    "left_gripper",
)
OBSERVATION_STATE_FEATURE_NAMES = (
    "left_ee_pose.x",
    "left_ee_pose.y",
    "left_ee_pose.z",
    "left_ee_pose.rx",
    "left_ee_pose.ry",
    "left_ee_pose.rz",
)


class DummyChunkModel(nn.Module):
    def __init__(self, chunks: list[torch.Tensor]):
        super().__init__()
        self.chunks = chunks
        self.calls = 0

    def forward(self, batch: dict[str, torch.Tensor]):
        chunk = self.chunks[self.calls]
        self.calls += 1
        return chunk, None


def _make_policy(
    *,
    action_delta_alignment: str,
    chunk_size: int,
    n_action_steps: int = 1,
    temporal_ensemble_coeff: float | None = None,
) -> ACTPolicy:
    config = ACTConfig(
        use_vae=False,
        device="cpu",
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(1,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(ACTION_FEATURE_NAMES),))},
        action_delta_alignment=action_delta_alignment,
        action_feature_names=ACTION_FEATURE_NAMES,
        observation_state_feature_names=OBSERVATION_STATE_FEATURE_NAMES,
    )
    return ACTPolicy(config)


def _raw_observation(x: float, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.tensor([[x, 0.0, 0.0, rx, ry, rz]], dtype=torch.float32),
    }


def _batch() -> dict[str, torch.Tensor]:
    return {OBS_ENV_STATE: torch.zeros(1, 1)}


def test_stepwise_select_action_queue_behavior_is_unchanged():
    policy = _make_policy(action_delta_alignment="step_wise", chunk_size=2, n_action_steps=2)
    policy.model = DummyChunkModel(
        [
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                    ]
                ]
            )
        ]
    )

    first_action = policy.select_action(_batch())
    second_action = policy.select_action(_batch())

    torch.testing.assert_close(first_action, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]))
    torch.testing.assert_close(second_action, torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]]))


def test_stepwise_temporal_ensemble_behavior_is_unchanged():
    policy = _make_policy(
        action_delta_alignment="step_wise",
        chunk_size=2,
        temporal_ensemble_coeff=0.0,
    )
    policy.model = DummyChunkModel(
        [
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
        ]
    )

    first_action = policy.select_action(_batch())
    second_action = policy.select_action(_batch())

    torch.testing.assert_close(first_action, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(second_action, torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


def test_chunkwise_predict_action_chunk_decodes_absolute_targets():
    policy = _make_policy(action_delta_alignment="chunk_wise", chunk_size=2)
    policy.model = DummyChunkModel(
        [
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                    ]
                ]
            )
        ]
    )

    action_chunk = policy.predict_action_chunk(
        _batch(),
        raw_observation=_raw_observation(x=10.0),
        postprocessor=lambda action: action,
    )

    expected = torch.tensor(
        [
            [
                [11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            ]
        ]
    )
    torch.testing.assert_close(action_chunk, expected)


def test_chunkwise_select_action_returns_absolute_targets():
    policy = _make_policy(action_delta_alignment="chunk_wise", chunk_size=1)
    policy.model = DummyChunkModel(
        [torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]])]
    )

    action = policy.select_action(
        _batch(),
        raw_observation=_raw_observation(x=10.0),
        postprocessor=lambda tensor: tensor,
    )

    torch.testing.assert_close(action, torch.tensor([[11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]))


def test_chunkwise_temporal_ensemble_decodes_before_averaging():
    policy = _make_policy(
        action_delta_alignment="chunk_wise",
        chunk_size=2,
        temporal_ensemble_coeff=0.0,
    )
    policy.model = DummyChunkModel(
        [
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
        ]
    )

    first_action = policy.select_action(
        _batch(),
        raw_observation=_raw_observation(x=10.0),
        postprocessor=lambda tensor: tensor,
    )
    second_action = policy.select_action(
        _batch(),
        raw_observation=_raw_observation(x=100.0),
        postprocessor=lambda tensor: tensor,
    )

    torch.testing.assert_close(first_action, torch.tensor([[11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(second_action, torch.tensor([[57.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


def test_chunkwise_inference_requires_observation_state_for_chunk_ref_pose():
    policy = _make_policy(action_delta_alignment="chunk_wise", chunk_size=1)
    policy.model = DummyChunkModel(
        [torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])]
    )

    with pytest.raises(ValueError, match="observation.state"):
        policy.predict_action_chunk(
            _batch(),
            raw_observation={},
            postprocessor=lambda action: action,
        )
