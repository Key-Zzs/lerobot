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
from pathlib import Path


@dataclass
class DAggerDatasetConfig:
    """Dataset paths used by the round DAgger ACT training helper."""

    # Fixed seed demo dataset used as the schema/training fallback.
    seed_repo_id: str
    seed_root: str | Path | None = None

    # Aggregated dataset produced by run_dagger_rounds.py.
    aggregated_repo_id: str | None = None
    aggregated_root: str | Path | None = None

    # If aggregated dataset already exists, use it instead of bootstrapping.
    resume_aggregation: bool = True
    # If aggregated dataset does not exist, bootstrap it by copying seed dataset.
    copy_seed_if_missing: bool = True

    # Kept for compatibility with older config dictionaries.
    image_writer_processes: int = 0
    image_writer_threads: int = 0
