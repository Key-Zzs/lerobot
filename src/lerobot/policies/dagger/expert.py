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

import importlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from lerobot.policies.dagger.configuration_dagger import DAggerExpertConfig
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config


class ExpertPolicy(ABC):
    """Minimal expert interface used by the DAgger loop."""

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self,
        observation: RobotObservation,
        raw_observation: RobotObservation,
        episode_step: int,
        timestamp_s: float,
    ) -> RobotAction:
        raise NotImplementedError


class TeleopExpert(ExpertPolicy):
    def __init__(
        self,
        teleop: Teleoperator,
        teleop_action_processor: RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ],
    ):
        self.teleop = teleop
        self.teleop_action_processor = teleop_action_processor

    def connect(self) -> None:
        self.teleop.connect()

    def disconnect(self) -> None:
        self.teleop.disconnect()

    def get_action(
        self,
        observation: RobotObservation,
        raw_observation: RobotObservation,
        episode_step: int,
        timestamp_s: float,
    ) -> RobotAction:
        del observation, episode_step, timestamp_s
        raw_action = self.teleop.get_action()
        return self.teleop_action_processor((raw_action, raw_observation))


class CallableExpert(ExpertPolicy):
    """
    Wraps a Python callable expert.

    Callable signature can use any subset of:
    observation, raw_observation, robot, episode_step, timestamp_s
    """

    def __init__(self, fn: Callable[..., RobotAction], robot: Robot):
        self.fn = fn
        self.robot = robot

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def get_action(
        self,
        observation: RobotObservation,
        raw_observation: RobotObservation,
        episode_step: int,
        timestamp_s: float,
    ) -> RobotAction:
        kwargs = {
            "observation": observation,
            "raw_observation": raw_observation,
            "robot": self.robot,
            "episode_step": episode_step,
            "timestamp_s": timestamp_s,
        }
        action = _call_with_supported_kwargs(self.fn, kwargs)
        if not isinstance(action, dict):
            raise TypeError(
                "Callable expert must return dict[str, Any] action. "
                f"Got type={type(action).__name__}."
            )
        return action


def _call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return fn(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    try:
        return fn(**filtered)
    except TypeError:
        if len(params) == 1:
            # Common fallback: fn(raw_observation)
            return fn(kwargs["raw_observation"])
        raise


def load_expert_callable(callable_path: str) -> Callable[..., RobotAction]:
    if ":" not in callable_path:
        raise ValueError(
            "expert.callable_path must be in format 'module.submodule:callable_or_class'. "
            f"Got {callable_path!r}."
        )
    module_name, attr_name = callable_path.split(":", maxsplit=1)

    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)

    if inspect.isclass(target):
        target = target()

    if hasattr(target, "get_action") and callable(target.get_action):
        return target.get_action

    if callable(target):
        return target

    raise ValueError(
        f"Loaded expert target from {callable_path!r} is neither callable nor object with get_action()."
    )


def make_expert(
    cfg: DAggerExpertConfig,
    robot: Robot,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
) -> ExpertPolicy:
    if cfg.mode == "teleop":
        if cfg.teleop is None:
            raise ValueError("cfg.teleop cannot be None when expert.mode='teleop'.")
        teleop = make_teleoperator_from_config(cfg.teleop)
        return TeleopExpert(teleop, teleop_action_processor)

    if cfg.mode == "callable":
        if not cfg.callable_path:
            raise ValueError("cfg.callable_path cannot be empty when expert.mode='callable'.")
        fn = load_expert_callable(cfg.callable_path)
        return CallableExpert(fn, robot)

    raise ValueError(f"Unsupported expert mode: {cfg.mode}")
