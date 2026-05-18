"""
混合专家（MoE）模块

DeepSeek-V4 Pro架构：
    - Hash路由MoE：前n_hash_layers使用token_id % num_experts
    - Learned-gate MoE：后续层使用门控网络选择专家
    - 路由专家 + 共享专家结构
"""

from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelArgs
from .layers import Linear


class MLP(nn.Module):
    """
    SwiGLU MLP（共享专家使用）。

    Args:
        dim: 输入/输出维度。
        inter_dim: 中间层维度。
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: w2(silu(w1(x)) * w3(x))"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """
    单个路由专家。

    Args:
        dim: 输入/输出维度。
        inter_dim: 中间层维度。
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    MoE门控路由。

    支持两种模式：
        - Hash路由（前n_hash_layers）：token_id % num_experts
        - Learned-gate路由：sqrt(softplus)打分 + topk选择

    Args:
        layer_id: 层索引。
        args: 模型配置。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.n_routed_experts = args.n_routed_experts

        # Hash路由不需要参数（直接取模）
        if not self.hash:
            self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts))
            nn.init.normal_(self.weight, std=0.02)
            nn.init.zeros_(self.bias)

    def _hash_route(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Hash路由：token_id % num_experts。

        均匀分配token到专家，固定不变（无学习参数）。
        """
        return (input_ids % self.n_routed_experts).unsqueeze(-1)

    def _learned_route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learned-gate路由：学习选择专家。

        使用sqrt(softplus)打分，topk选择。
        """
        # 统一使用float32计算
        x_fp32 = x.float()
        weight_fp32 = self.weight.float()

        scores = F.linear(x_fp32, weight_fp32)

        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()

        scores = scores + self.bias
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)

        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        weights = weights * self.route_scale

        return weights, indices

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        """
        门控前向传播。

        Args:
            x: 输入 [num_tokens, dim]。
            input_ids: token IDs [num_tokens]，hash路由时需要。

        Returns:
            (weights, indices): 专家权重和索引。
        """
        if self.hash:
            # Hash路由：返回均匀权重 + hash计算的索引
            indices = self._hash_route(input_ids)
            weights = torch.ones_like(indices, dtype=torch.float32)
            return weights, indices
        else:
            return self._learned_route(x)


class MoE(nn.Module):
    """
    混合专家模块。

    结构：路由专家（topk选择）+ 共享专家（所有token经过）

    Args:
        layer_id: 层索引。
        args: 模型配置。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts

        self.gate = Gate(layer_id, args)

        # 路由专家（单GPU环境：全部创建）
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            for _ in range(args.n_routed_experts)
        ])

        # 共享专家
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入 [batch, seq, dim]。
            input_ids: token IDs [batch, seq]，hash路由时使用。

        Returns:
            输出 [batch, seq, dim]。
        """
        shape = x.shape
        x = x.view(-1, self.dim)

        # 获取路由信息
        if input_ids is not None:
            flat_ids = input_ids.flatten()
        else:
            flat_ids = None

        weights, indices = self.gate(x, flat_ids)

        # 收集各专家输出
        expert_outs = []
        for i in range(self.n_routed_experts):
            # 找到分配给该专家的token
            mask = (indices == i).any(dim=-1)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            expert = self.experts[i]
            # 取对应的权重（hash路由时权重为1）
            w = weights[mask].mean(dim=-1, keepdim=True)
            out = expert(x[mask]) * w
            expert_outs.append((idx, out))

        # 合并专家输出
        if expert_outs:
            all_idx = torch.cat([idx for idx, _ in expert_outs])
            all_out = torch.cat([out for _, out in expert_outs], dim=0)
            # 使用scatter_add聚合
            base = torch.zeros_like(x)
            y = base.scatter_add(0, all_idx.unsqueeze(-1).expand(-1, x.size(-1)), all_out)
        else:
            y = torch.zeros_like(x)

        # 共享专家（所有token经过）
        z = self.shared_experts(x)

        return (y + z).view(shape)
