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

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.dagger.configuration_dagger import DAggerPipelineConfig
from lerobot.policies.dagger.dataset import ensure_aggregated_dataset_ready, make_training_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


def _build_dataloader(
    cfg: DAggerPipelineConfig,
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.DataLoader:
    can_use_episode_sampler = (
        hasattr(dataset, "meta")
        and hasattr(cfg.policy, "drop_n_last_frames")
        and getattr(dataset, "supports_episode_sampler", True)
    )

    sampler = None
    shuffle = True
    if can_use_episode_sampler:
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        pin_memory=cfg.policy.device == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )


def _save_round_checkpoint(
    checkpoint_dir: Path,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    round_state: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
    )
    postprocessor.save_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
    )

    with open(checkpoint_dir / "dagger_round_state.json", "w") as f:
        json.dump(round_state, f, indent=2)


def _compute_train_metrics(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    batch: dict[str, torch.Tensor],
    grad_clip_norm: float,
) -> tuple[float, float, dict[str, float]]:
    device = torch.device(policy.config.device)
    use_amp = scaler.is_enabled() and device.type == "cuda"

    policy.train()
    optimizer.zero_grad(set_to_none=True)

    autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()

    with autocast_ctx:
        loss, loss_dict = policy.forward(batch)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
        )
        optimizer.step()

    train_log = {k: float(v) for k, v in (loss_dict or {}).items()}
    return float(loss.item()), float(grad_norm.item()), train_log


def train_dagger(cfg: DAggerPipelineConfig) -> None:
    """
    Offline DAgger training: trains ACT only from collected dagger_data (expert-labeled rows).

    Online rollout, beta scheduling, expert RPC, and keyboard interaction are intentionally disabled.
    """
    cfg.validate()
    init_logging()
    logging.info("Starting offline DAgger training")

    if cfg.seed is not None:
        set_seed(cfg.seed)

    train_repo_id, train_root = ensure_aggregated_dataset_ready(cfg.dataset)
    train_dataset = make_training_dataset(
        repo_id=train_repo_id,
        root=train_root,
        policy_cfg=cfg.policy,
        expert_only=True,
    )

    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)

    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    optimizer = cfg.optimizer.build(params)

    if cfg.scheduler:
        total_steps = cfg.training.steps_per_round * cfg.training.rounds
        scheduler = cfg.scheduler.build(optimizer, total_steps)
    else:
        scheduler = None

    grad_clip_norm = cfg.optimizer.grad_clip_norm
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.policy.use_amp and cfg.policy.device == "cuda")

    train_preprocessor, train_postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=train_dataset.meta.stats,
    )

    dataloader = _build_dataloader(cfg, train_dataset)
    dl_iter = cycle(dataloader)

    latest_train_log: dict[str, float] = {}

    for round_idx in range(cfg.training.rounds):
        logging.info("[round %d/%d] offline training", round_idx + 1, cfg.training.rounds)

        running_loss = 0.0
        running_grad = 0.0
        running_steps = 0

        for step_idx in range(cfg.training.steps_per_round):
            batch = next(dl_iter)
            batch = train_preprocessor(batch)

            loss, grad_norm, loss_dict = _compute_train_metrics(
                policy=policy,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                grad_clip_norm=grad_clip_norm,
            )
            if scheduler is not None:
                scheduler.step()

            running_loss += loss
            running_grad += grad_norm
            running_steps += 1
            latest_train_log = loss_dict

            is_log_step = (step_idx + 1) % cfg.training.log_freq == 0
            is_last_step = step_idx + 1 == cfg.training.steps_per_round
            if is_log_step or is_last_step:
                mean_loss = running_loss / max(1, running_steps)
                mean_grad = running_grad / max(1, running_steps)
                logging.info(
                    "[round %d] train step %d/%d loss=%.5f grad=%.5f extra=%s",
                    round_idx + 1,
                    step_idx + 1,
                    cfg.training.steps_per_round,
                    mean_loss,
                    mean_grad,
                    latest_train_log,
                )
                running_loss = 0.0
                running_grad = 0.0
                running_steps = 0

        if cfg.training.save_checkpoint:
            checkpoint_dir = cfg.output_dir / f"round_{round_idx + 1:03d}"
            round_state = {
                "round": round_idx + 1,
                "mode": "offline_dagger",
                "train_dataset": {
                    "repo_id": train_repo_id,
                    "root": str(train_root),
                    "num_frames": train_dataset.num_frames,
                    "num_episodes": train_dataset.num_episodes,
                },
                "train_last_log": latest_train_log,
            }
            _save_round_checkpoint(
                checkpoint_dir=checkpoint_dir,
                policy=policy,
                preprocessor=train_preprocessor,
                postprocessor=train_postprocessor,
                round_state=round_state,
            )

    final_dir = cfg.output_dir / "final"
    _save_round_checkpoint(
        checkpoint_dir=final_dir,
        policy=policy,
        preprocessor=train_preprocessor,
        postprocessor=train_postprocessor,
        round_state={
            "status": "finished",
            "mode": "offline_dagger",
            "rounds": cfg.training.rounds,
            "train_dataset": {
                "repo_id": train_repo_id,
                "root": str(train_root),
                "num_frames": train_dataset.num_frames,
                "num_episodes": train_dataset.num_episodes,
            },
        },
    )

    logging.info("Offline DAgger training completed. Output dir: %s", cfg.output_dir)
