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

import math

from lerobot.policies.dagger.configuration_dagger import DAggerBetaConfig


class BetaScheduler:
    """Computes DAgger beta (expert action probability) by round."""

    def __init__(self, config: DAggerBetaConfig, num_rounds: int):
        self.config = config
        self.num_rounds = max(1, num_rounds)

    def value(self, round_index: int) -> float:
        start = self.config.beta_start
        end = self.config.beta_end

        if self.config.schedule == "constant":
            beta = start
        elif self.config.schedule == "linear":
            if self.num_rounds == 1:
                beta = end
            else:
                progress = round_index / (self.num_rounds - 1)
                beta = start + (end - start) * progress
        elif self.config.schedule == "exponential":
            beta = end + (start - end) * math.exp(-self.config.exp_decay * round_index)
        else:
            raise ValueError(f"Unsupported beta schedule: {self.config.schedule}")

        return float(min(1.0, max(0.0, beta)))
