#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.act.action_delta_utils import (
    ACT_CHUNKWISE_LABELS_CONVERTED_KEY,
    convert_stepwise_to_chunkwise_actions,
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@dataclass
@ProcessorStepRegistry.register(name="act_chunkwise_action_label_processor")
class ACTChunkwiseActionLabelProcessorStep(ProcessorStep):
    """Convert ACT chunk-wise training labels before action normalization.

    Dataset actions are stored as executable step-wise deltas. For chunk-wise ACT supervision we first convert
    those raw physical deltas into offsets from the chunk start, then the shared normalizer scales the converted
    labels. Running the conversion after normalization would accumulate the normalization mean and corrupt the
    chunk target.
    """

    action_delta_alignment: str = "step_wise"
    action_feature_names: tuple[str, ...] = ()

    def __call__(self, transition):
        if self.action_delta_alignment == "step_wise":
            return transition
        if self.action_delta_alignment != "chunk_wise":
            raise ValueError(
                "`action_delta_alignment` must be either 'step_wise' or 'chunk_wise'. "
                f"Got {self.action_delta_alignment}."
            )

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        converted_transition = transition.copy()
        converted_transition[TransitionKey.ACTION] = convert_stepwise_to_chunkwise_actions(
            action,
            self.action_feature_names,
        )
        complementary_data = dict(converted_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary_data[ACT_CHUNKWISE_LABELS_CONVERTED_KEY] = True
        converted_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return converted_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]
    if config.action_delta_alignment == "chunk_wise":
        input_steps.append(
            ACTChunkwiseActionLabelProcessorStep(
                action_delta_alignment=config.action_delta_alignment,
                action_feature_names=config.action_feature_names,
            )
        )
    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        )
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
