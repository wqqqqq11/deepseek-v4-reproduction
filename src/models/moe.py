"""
混合专家（MoE）模块

包含：
    - MLP       多层感知机（前馈网络），用于 Block 中的密集层和共享专家
    - Gate      门控路由机制，决定每个 token 被分配给哪些专家
    - Expert    单个专家（小型 MLP）
    - MoE       混合专家模块，将 Gate、多个 Expert 和共享专家组合在一起
"""

from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .config import ModelArgs
from .layers import linear, Linear, ColumnParallelLinear, RowParallelLinear
from .RuntimeConfig import RuntimeConfig


class MLP(nn.Module):
    """
    多层感知机（前馈网络），使用 SwiGLU 激活。

    两个列并行层和一个行并行层的组合：
        w1 (ColumnParallelLinear)  + Swish →
        w3 (ColumnParallelLinear)           → * → w2 (RowParallelLinear) → 输出
    门控乘积结构，广泛应用于 Transformer 的 FFN。

    属性:
        w1: 第一列并行线性层，dim → inter_dim。
        w2: 行并行线性层，inter_dim → dim（聚合各 GPU 结果）。
        w3: 第二列并行线性层，dim → inter_dim。
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        初始化 MLP 层。

        Args:
            dim:       输入/输出特征维度。
            inter_dim: 中间隐藏层维度（列并行切分后的本地维度）。
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)   # 输入 → 隐藏（列切分）
        self.w2 = RowParallelLinear(inter_dim, dim)      # 隐藏 → 输出（行切分 + all_reduce）
        self.w3 = ColumnParallelLinear(dim, inter_dim)   # 输入 → 隐藏（列切分），门控分支

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        计算 SwiGLU 门控前馈：
            output = w2( silu(w1(x)) * w3(x) )
        w1 和 w3 的输出在本地 GPU 上逐元素相乘后，通过 w2 的行聚合得到完整输出。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, dim)。

        Returns:
            同形状输出张量。
        """
        # w1(x) 经过 SiLU 激活，与 w3(x) 逐元素相乘（门控机制）
        # w2 内部会做 all_reduce 聚合各 GPU 的部分结果
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    MoE 门控路由，决定每个 token 应该分配给哪些专家。

    支持两种评分函数：
        - "softmax"：对专家得分做 softmax 归一化。
        - "sigmoid"：对得分做 sigmoid，最后归一化权重。

    可选分组路由（n_groups > 1）：先将专家分为多组，先选组再选组内专家。

    属性:
        dim (int):             输入特征维度。
        topk (int):            每个 token 激活的专家数量。
        n_groups (int):        专家分组数。
        topk_groups (int):     路由到的组数。
        score_func (str):      评分函数，"softmax" 或 "sigmoid"。
        route_scale (float):   路由权重缩放因子。
        weight (Parameter):    门控权重矩阵 (n_routed_experts, dim)。
        bias (Parameter|None): 可选门控偏置（仅 dim==7168 时使用）。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.hash:
            with torch.no_grad():
                self.tid2eid.copy_(
                    torch.randint(
                        0,
                        args.n_routed_experts,
                        self.tid2eid.shape,
                        dtype=torch.int32,
                    )
                )
        else:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """
    单个专家层，结构与 MLP 一致但使用普通 Linear（非并行）。

    每个专家拥有自己独立的参数，不同专家可以学到不同的模式。

    属性:
        w1 (Linear): 输入 → 隐藏层。
        w2 (Linear): 隐藏层 → 输出。
        w3 (Linear): 输入 → 隐藏层（门控分支）。
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        初始化单个专家。

        Args:
            dim:       输入/输出特征维度。
            inter_dim: 中间隐藏层维度。
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)     # 无并行，完整的 dim → inter_dim
        self.w2 = Linear(inter_dim, dim)     # 无并行，完整的 inter_dim → dim
        self.w3 = Linear(dim, inter_dim)     # 门控分支

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状 (num_selected_tokens, dim)。

        Returns:
            同形状输出张量。
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    混合专家（MoE）模块。

    每个 token 通过 Gate 选择 top-k 个专家，各专家独立计算，
    输出按门控权重加权求和。同时所有 token 都经过共享专家（Shared Expert），
    最终结果 = 路由专家加权和 + 共享专家输出。

    分布式下，专家均匀分配到各 GPU，每个 GPU 只计算本地持有的专家，
    最后通过 all_reduce 聚合所有 GPU 的专家输出。

    属性:
        dim (int):                 输入特征维度。
        n_routed_experts (int):    路由专家总数（全局）。
        n_local_experts (int):     当前 GPU 持有的本地专家数。
        n_activated_experts (int): 每个 token 激活的专家数。
        experts_start_idx (int):   本 GPU 负责的专家起始索引。
        experts_end_idx (int):     本 GPU 负责的专家结束索引（不包含）。
        gate (Gate):               门控路由模块。
        experts (ModuleList):      专家列表，非本地专家位置为 None。
        shared_experts (MLP):      共享专家，每个 token 都经过。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化 MoE 模块。

        Args:
            layer_id: 层索引，用于 Gate 的 hash 路由判断。
            args: 模型参数，包含 dim、n_routed_experts、n_activated_experts、
                  n_shared_experts、moe_inter_dim 等。
        """
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        # 替换全局 world_size
        assert args.n_routed_experts % RuntimeConfig.default().world_size == 0, (
            f"Number of experts ({args.n_routed_experts}) must be divisible by world_size ({RuntimeConfig.default().world_size})"
        )
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // RuntimeConfig.default().world_size
        self.n_activated_experts = args.n_activated_experts
        # 替换全局 rank
        self.experts_start_idx = RuntimeConfig.default().rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Gate(layer_id, args)

        # 创建专家列表：只有属于本 GPU 范围内的专家才实例化，其余为 None
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            if self.experts_start_idx <= i < self.experts_end_idx
            else None
            for i in range(self.n_routed_experts)
        ])

        # 共享专家：所有 token 都会通过，内部维度 = shared_experts * moe_inter_dim
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, dim)。
            input_ids: 输入 token IDs，形状 (batch_size, seq_len)。
                      用于 hash 路由（当前保留兼容性，未来可扩展）。

        Returns:
            同形状输出张量。
        """
        shape = x.size()
        # 展平成 (num_tokens, dim)，方便按 token 路由
        x = x.view(-1, self.dim)

        # 第1步：门控路由，获取每个 token 的专家权重和索引
        # 如果 input_ids 不为 None，展平后传递给 Gate（用于 hash 路由）
        if input_ids is not None:
            flat_input_ids = input_ids.flatten()
            weights, indices = self.gate(x, flat_input_ids)
        else:
            weights, indices = self.gate(x, None)           # weights: (num_tokens, topk), indices: (num_tokens, topk)

        # 第2步：统计每个专家的 token 分配数量
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # 第3步：收集各专家输出（避免 in-place 操作以支持梯度追踪）
        expert_outs = []
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            out = expert(x[idx]) * weights[idx, top, None]
            expert_outs.append((idx, out))

        # 第4步：合并专家输出
        # 从 x 派生零张量保持梯度连接，再用 scatter_add 聚合专家输出
        if expert_outs:
            all_idx = torch.cat([idx for idx, _ in expert_outs])
            all_out = torch.cat([out for _, out in expert_outs], dim=0).type_as(x)
            base = x * 0
            y = base.scatter_add(0, all_idx.unsqueeze(-1).expand(-1, x.size(-1)), all_out)
        else:
            y = x * 0

        # 第5步：共享专家计算（所有 token 都经过）
        z = self.shared_experts(x)

        # 第6步：分布式聚合
        if RuntimeConfig.default().world_size > 1:
            dist.all_reduce(y)

        # 第7步：合并输出并恢复形状
        return (y + z).view(shape)
