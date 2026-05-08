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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.action_delta_utils import (
    ACT_CHUNKWISE_LABELS_CONVERTED_KEY,
    convert_stepwise_to_chunkwise_actions,
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE


def test_convert_stepwise_to_chunkwise_actions_xyz_cumulative():
    action_feature_names = (
        "left_delta_ee_pose.x",
        "left_delta_ee_pose.y",
        "left_delta_ee_pose.z",
        "left_delta_ee_pose.rx",
        "left_delta_ee_pose.ry",
        "left_delta_ee_pose.rz",
        "left_gripper",
    )
    actions = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            ]
        ]
    )

    converted = convert_stepwise_to_chunkwise_actions(actions, action_feature_names)

    expected = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            ]
        ]
    )
    torch.testing.assert_close(converted, expected)


def test_convert_stepwise_to_chunkwise_actions_preserves_shape_dtype_and_device():
    action_feature_names = (
        "left_delta_ee_pose.x",
        "left_delta_ee_pose.y",
        "left_delta_ee_pose.z",
        "right_delta_ee_pose.x",
        "right_delta_ee_pose.y",
        "right_delta_ee_pose.z",
        "left_gripper",
        "right_gripper",
    )
    actions = torch.randn(2, 4, len(action_feature_names), dtype=torch.float64)

    converted = convert_stepwise_to_chunkwise_actions(actions, action_feature_names)

    assert converted.shape == actions.shape
    assert converted.dtype == actions.dtype
    assert converted.device == actions.device


def test_convert_stepwise_to_chunkwise_actions_treats_rx_ry_rz_like_server_euler_deltas():
    action_feature_names = (
        "left_delta_ee_pose.x",
        "left_delta_ee_pose.y",
        "left_delta_ee_pose.z",
        "left_delta_ee_pose.rx",
        "left_delta_ee_pose.ry",
        "left_delta_ee_pose.rz",
    )
    quarter_turn = torch.pi / 2
    actions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, quarter_turn, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, quarter_turn, 0.0],
            ]
        ]
    )

    converted = convert_stepwise_to_chunkwise_actions(actions, action_feature_names)

    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, quarter_turn, 0.0, 0.0],
                [0.0, 0.0, 0.0, quarter_turn, quarter_turn, 0.0],
            ]
        ]
    )
    torch.testing.assert_close(converted, expected, atol=1e-5, rtol=1e-5)


def test_act_policy_stepwise_alignment_keeps_training_actions_unchanged():
    config = ACTConfig(
        use_vae=False,
        device="cpu",
        input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = ACTPolicy(config)
    batch = {
        OBS_ENV_STATE: torch.randn(2, 4),
        ACTION: torch.randn(2, 3, 7),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }

    prepared_batch = policy._prepare_training_batch(batch)

    assert prepared_batch is batch
    assert prepared_batch[ACTION] is batch[ACTION]


def test_act_chunkwise_training_preprocessor_converts_raw_actions_before_normalization():
    action_feature_names = (
        "left_delta_ee_pose.x",
        "left_delta_ee_pose.y",
        "left_delta_ee_pose.z",
        "right_delta_ee_pose.x",
        "right_delta_ee_pose.y",
        "right_delta_ee_pose.z",
    )
    config = ACTConfig(
        use_vae=False,
        device="cpu",
        input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(action_feature_names),))},
        action_delta_alignment="chunk_wise",
        action_feature_names=action_feature_names,
    )
    right_z_index = action_feature_names.index("right_delta_ee_pose.z")
    action_mean = torch.zeros(len(action_feature_names))
    action_mean[right_z_index] = -0.1
    action_std = torch.ones(len(action_feature_names))
    preprocessor, _ = make_act_pre_post_processors(
        config,
        dataset_stats={ACTION: {"mean": action_mean, "std": action_std}},
    )
    batch = {
        OBS_ENV_STATE: torch.randn(1, 4),
        ACTION: torch.zeros(1, 2, len(action_feature_names)),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }
    batch[ACTION][0, :, right_z_index] = torch.tensor([-1.0, -2.0])

    processed = preprocessor(batch)

    assert processed[ACT_CHUNKWISE_LABELS_CONVERTED_KEY] is True
    expected_chunkwise = convert_stepwise_to_chunkwise_actions(batch[ACTION], action_feature_names)
    expected_normalized = expected_chunkwise.clone()
    expected_normalized[:, :, right_z_index] = (
        expected_chunkwise[:, :, right_z_index] - action_mean[right_z_index]
    ) / action_std[right_z_index]
    torch.testing.assert_close(processed[ACTION], expected_normalized)
