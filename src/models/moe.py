"""
混合专家（MoE）模块

包含：
    - MLP       多层感知机（前馈网络），用于 Block 中的密集层和共享专家
    - Gate      门控路由机制，决定每个 token 被分配给哪些专家
    - Expert    单个专家（小型 MLP）
    - MoE       混合专家模块，将 Gate、多个 Expert 和共享专家组合在一起
"""

from typing import Tuple
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

    def __init__(self, args: ModelArgs):
        """
        初始化门控模块。

        Args:
            args: 模型参数，包含 n_activated_experts、n_expert_groups、n_limited_groups、
                  score_func、route_scale、n_routed_experts、dim 等。
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 偏置仅在与原始 DeepSeek 特定维度一致时使用
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，计算路由权重和专家索引。

        Args:
            x: 输入张量，形状 (num_tokens, dim)，通常已展平批次和序列维度。

        Returns:
            weights: (num_tokens, topk) 每个 token 分配给所选专家的权重。
            indices: (num_tokens, topk) 所选专家的索引（int64）。
        """
        # 第1步：计算原始得分 logits
        scores = linear(x, self.weight)                     # (num_tokens, n_routed_experts)

        # 第2步：根据评分函数转换为概率/权重
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()                       # sigmoid 得分
        original_scores = scores.clone()                            # 保存原始权重，用于最终加权

        # 第3步：加上可选的偏置
        if self.bias is not None:
            scores = scores + self.bias

        # 第4步：分组路由（如果有多组）
        if self.n_groups > 1:
            # 将专家分数按组重塑
            scores = scores.view(x.size(0), self.n_groups, -1)  # (num_tokens, n_groups, experts_per_group)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)               # 每组取最高分
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)  # 有偏置时取 top-2 和
            # 选出 topk_groups 个组
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # 第5步：选择 top-k 专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]   # (num_tokens, topk)
        weights = original_scores.gather(1, indices)          # 取对应原始权重

        # 第6步：sigmoid 模式下需要归一化权重
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # 第7步：缩放权重
        weights = weights * self.route_scale
        return weights.type_as(x), indices


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

    def __init__(self, args: ModelArgs):
        """
        初始化 MoE 模块。

        Args:
            args: 模型参数，包含 dim、n_routed_experts、n_activated_experts、
                  n_shared_experts、moe_inter_dim 等。
        """
        super().__init__()
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

        self.gate = Gate(args)

        # 创建专家列表：只有属于本 GPU 范围内的专家才实例化，其余为 None
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            if self.experts_start_idx <= i < self.experts_end_idx
            else None
            for i in range(self.n_routed_experts)
        ])

        # 共享专家：所有 token 都会通过，内部维度 = shared_experts * moe_inter_dim
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, dim)。

        Returns:
            同形状输出张量。
        """
        shape = x.size()
        # 展平成 (num_tokens, dim)，方便按 token 路由
        x = x.view(-1, self.dim)

        # 第1步：门控路由，获取每个 token 的专家权重和索引
        weights, indices = self.gate(x)                     # weights: (num_tokens, topk), indices: (num_tokens, topk)

        # 第2步：初始化输出张量
        y = torch.zeros_like(x)

        # 第3步：统计每个专家的 token 分配数量
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # 第4步：遍历本 GPU 持有的专家，对分配给它们的 token 进行计算
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue   # 没有 token 路由到此专家，跳过
            expert = self.experts[i]
            # 找到所有被路由到专家 i 的 token 位置 (idx) 及其在 topk 中的排名 (top)
            idx, top = torch.where(indices == i)
            # 专家输出 * 对应权重 累加到 y
            y[idx] += expert(x[idx]) * weights[idx, top, None]   # None 用于广播乘法

        # 第5步：共享专家计算（所有 token 都经过）
        z = self.shared_experts(x)

        # 第6步：分布式聚合：all_reduce 各 GPU 的专家输出
        # 替换全局 world_size
        if RuntimeConfig.default().world_size > 1:
            dist.all_reduce(y)

        # 第7步：路由专家输出 + 共享专家输出，恢复形状
        return (y + z).view(shape)
