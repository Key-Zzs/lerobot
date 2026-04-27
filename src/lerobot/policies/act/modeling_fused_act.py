from collections import deque
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.act.configuration_fused_act import FusedACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

class ActionFuser(nn.Module):
    """小型 MLP 融合器，根据机器人状态/环境状态输出子策略权重。"""
    def __init__(self, input_dim: int, hidden_dim: int, num_weights: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_weights),
        )

    def forward(self, state_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_feats: (B, input_dim) 拼接的状态特征
        Returns:
            weights: (B, num_weights) 已归一化（和为1）的正权重
        """
        logits = self.net(state_feats)
        weights = F.softmax(logits, dim=-1)   # 保证正且和为1
        return weights

class FusedACTPolicy(PreTrainedPolicy):
    """
    多相机融合 ACT 策略。
    内部包含三个独立的子 ACT 策略（单相机）和一个融合器。
    """

    config_class = FusedACTConfig
    name = "fused_act"

    def __init__(self, config: FusedACTConfig):
        super().__init__(config)
        self.config = config
        config.validate_features()
        config.build_sub_configs_from_features()

        # 初始化三个子策略（它们各自拥有独立的网络参数）
        self.sub_policies = nn.ModuleList([
            ACTPolicy(cfg) for cfg in config.sub_configs
        ])

        # 确定融合器的输入维度
        fuser_input_dim = 0
        for feat in config.fuser_input_features:
            if feat == OBS_STATE and config.sub_configs[0].robot_state_feature is not None:
                fuser_input_dim += config.sub_configs[0].robot_state_feature.shape[0]
            elif feat == OBS_ENV_STATE and config.sub_configs[0].env_state_feature is not None:
                fuser_input_dim += config.sub_configs[0].env_state_feature.shape[0]
        if fuser_input_dim == 0:
            raise ValueError(
                "fused_act fuser has no input features. Keep observation.state in the dataset or set "
                "fuser_input_features to an available non-visual feature."
            )

        self.fuser = ActionFuser(
            input_dim=fuser_input_dim,
            hidden_dim=config.fuser_hidden_dim,
            num_weights=config.num_sub_policies,
        )

        # 动作队列（复用原 ACT 的逻辑）
        self._action_queue = deque([], maxlen=config.sub_configs[0].n_action_steps)
        self.reset()

    def reset(self):
        for p in self.sub_policies:
            p.reset()
        self._action_queue.clear()

    def _get_fuser_input(self, batch: Dict[str, Tensor]) -> Tensor:
        """从 batch 中提取融合器需要的特征并拼接"""
        feats = []
        for key in self.config.fuser_input_features:
            if key in batch:
                # batch[key] shape: (B, *) 经过预处理可能已经 flatten，需确保是2D
                val = batch[key]
                if val.dim() > 2:
                    val = val.flatten(1)
                feats.append(val)
        if not feats:
            raise ValueError(
                f"Batch does not contain any fuser input features: {self.config.fuser_input_features}"
            )
        return torch.cat(feats, dim=-1)

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        """
        统一的前向接口。
        根据 config.training_stage 决定行为：
            - "sub_policy_i"：只训练第 i 个子策略（其他部分冻结/不计算梯度）
            - "fuser"：只训练融合器，子策略冻结
        推理时调用 select_action（内部使用融合器加权）
        """
        stage = self.config.training_stage

        # ---- 子策略单独训练阶段 ----
        if stage.startswith("sub_policy_"):
            idx = int(stage.split("_")[-1])
            sub_pol = self.sub_policies[idx]
            loss, loss_dict = sub_pol.forward(batch)
            return loss, loss_dict

        # ---- 融合器训练阶段 ----
        elif stage == "fuser":
            # 冻结子策略
            for p in self.sub_policies:
                p.eval()
                for param in p.parameters():
                    param.requires_grad = False

            # 计算三个子策略各自的预测
            actions_list = []
            with torch.no_grad():
                for i, sub_pol in enumerate(self.sub_policies):
                    # 子策略的 predict_action_chunk 返回 (B, chunk, action_dim)
                    act = sub_pol.predict_action_chunk(batch)
                    actions_list.append(act)

            # 堆叠为 (B, num_policies, chunk, action_dim)
            stacked_actions = torch.stack(actions_list, dim=1)

            # 融合器前向
            fuser_input = self._get_fuser_input(batch)
            weights = self.fuser(fuser_input)  # (B, num_policies)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)

            # 加权动作
            weighted_action = (stacked_actions * weights).sum(dim=1)  # (B, chunk, action_dim)

            # 与真值动作计算 MSE
            true_action = batch[ACTION]
            if "action_is_pad" in batch:
                mask = ~batch["action_is_pad"].unsqueeze(-1)
                loss = (F.mse_loss(weighted_action, true_action, reduction="none") * mask).mean()
            else:
                loss = F.mse_loss(weighted_action, true_action)
            loss_dict = {"mse_loss": loss.item()}
            return loss, loss_dict

        else:
            raise ValueError(f"未知训练阶段: {stage}")

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """环境交互时的动作选择，使用融合器加权输出"""
        self.eval()
        if len(self._action_queue) == 0:
            # 获取每个子策略的 chunk
            actions_list = []
            for i, sub_pol in enumerate(self.sub_policies):
                act = sub_pol.predict_action_chunk(batch)  # (B, chunk, dim)
                actions_list.append(act)
            stacked_actions = torch.stack(actions_list, dim=1)  # (B, 3, chunk, dim)

            # 融合器权重
            fuser_input = self._get_fuser_input(batch)
            weights = self.fuser(fuser_input)  # (B, 3)
            weights = weights.unsqueeze(-1).unsqueeze(-1)
            weighted_chunk = (stacked_actions * weights).sum(dim=1)  # (B, chunk, dim)

            # 按 n_action_steps 截断并加入队列
            n_steps = self.config.sub_configs[0].n_action_steps
            actions = weighted_chunk[:, :n_steps]
            self._action_queue.extend(actions.transpose(0, 1))  # (n_steps, B, dim)
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor]) -> Tensor:
        """Predict a fused action chunk from all sub-policies."""
        self.eval()
        actions_list = [sub_pol.predict_action_chunk(batch) for sub_pol in self.sub_policies]
        stacked_actions = torch.stack(actions_list, dim=1)
        fuser_input = self._get_fuser_input(batch)
        weights = self.fuser(fuser_input).unsqueeze(-1).unsqueeze(-1)
        return (stacked_actions * weights).sum(dim=1)

    def get_optim_params(self):
        # 根据阶段返回对应可训练参数
        stage = self.config.training_stage
        if stage.startswith("sub_policy_"):
            idx = int(stage.split("_")[-1])
            return self.sub_policies[idx].get_optim_params()
        elif stage == "fuser":
            return [{"params": self.fuser.parameters()}]
        else:
            return [{"params": self.parameters()}]
