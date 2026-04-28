from dataclasses import dataclass, field
from pathlib import Path

import draccus

from lerobot.configs.policies import CONFIG_NAME, PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("fused_act")
@dataclass
class FusedACTConfig(PreTrainedConfig):
    """
    复合 ACT 配置：包含 3 个子 ACT 策略配置和一个融合器配置。
    """

    # 三个子策略的配置（可共享大部分参数，只需区分 image_features）
    sub_configs: list[ACTConfig] = field(default_factory=list)
    # Optional explicit camera keys. If omitted, the first num_sub_policies visual features are used.
    camera_keys: list[str] = field(default_factory=list)

    # 融合器输入特征（默认为机器人状态 + 环境状态）
    fuser_input_features: list[str] = field(
        default_factory=lambda: ["observation.state", "observation.environment_state"]
    )
    # 融合器隐藏层维度
    fuser_hidden_dim: int = 256
    # 融合器输出维度 = 子策略个数（此处固定为3）
    num_sub_policies: int = 3
    # Action chunk settings forwarded to each internal ACT policy.
    chunk_size: int = 100
    n_action_steps: int = 100

    # 当前训练阶段： "sub_policy_0", "sub_policy_1", "sub_policy_2", "fuser"
    training_stage: str = "sub_policy_0"

    # Training preset for the fuser. Sub-policy stages use the selected ACT sub-config preset.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    # Keep a top-level normalization map so train.py can create/load processors uniformly.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.sub_configs and len(self.sub_configs) != self.num_sub_policies:
            raise ValueError(f"期望 {self.num_sub_policies} 个子配置，实际提供 {len(self.sub_configs)}")
        # 将每个子配置的 device 同步到主配置
        for cfg in self.sub_configs:
            cfg.device = self.device

    def build_sub_configs_from_features(self) -> None:
        """Create one single-camera ACT config per visual feature when sub_configs are omitted."""
        if self.sub_configs:
            return

        if self.camera_keys:
            missing = [key for key in self.camera_keys if key not in self.image_features]
            if missing:
                raise ValueError(f"fused_act camera_keys 中包含数据集中不存在的视觉特征: {missing}")
            image_items = [(key, self.image_features[key]) for key in self.camera_keys]
        else:
            image_items = list(self.image_features.items())

        if len(image_items) < self.num_sub_policies:
            raise ValueError(
                f"fused_act 需要至少 {self.num_sub_policies} 个视觉输入，当前只有 {len(image_items)} 个。"
            )

        non_visual_inputs = {
            key: ft for key, ft in self.input_features.items() if ft.type is not FeatureType.VISUAL
        }
        self.sub_configs = []
        for image_key, image_feature in image_items[: self.num_sub_policies]:
            sub_cfg = ACTConfig(
                input_features={**non_visual_inputs, image_key: image_feature},
                output_features=self.output_features,
                device=self.device,
                use_amp=self.use_amp,
                chunk_size=self.chunk_size,
                n_action_steps=self.n_action_steps,
                push_to_hub=False,
                normalization_mapping=self.normalization_mapping,
            )
            self.sub_configs.append(sub_cfg)

    def get_optimizer_preset(self) -> AdamWConfig:
        if self.training_stage.startswith("sub_policy_") and self.sub_configs:
            idx = int(self.training_stage.split("_")[-1])
            return self.sub_configs[idx].get_optimizer_preset()
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        if self.training_stage.startswith("sub_policy_") and self.sub_configs:
            idx = int(self.training_stage.split("_")[-1])
            return self.sub_configs[idx].get_scheduler_preset()
        return None

    def validate_features(self) -> None:
        if len(self.image_features) < self.num_sub_policies and not self.sub_configs:
            raise ValueError(
                f"fused_act 需要至少 {self.num_sub_policies} 个视觉输入来自动创建子策略。"
            )
        if not self.output_features or ACTION not in self.output_features:
            raise ValueError("fused_act 需要 action 作为输出特征。")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        if self.sub_configs:
            return self.sub_configs[0].action_delta_indices
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        encoded = draccus.encode(self)
        # ACTConfig is not registered as a draccus choice class, so nested
        # ACT configs must be saved without the discriminator field.
        for sub_cfg in encoded.get("sub_configs", []):
            if isinstance(sub_cfg, dict):
                sub_cfg.pop("type", None)
        with open(save_directory / CONFIG_NAME, "w") as f:
            with draccus.config_type("json"):
                draccus.dump(encoded, f, indent=4)
