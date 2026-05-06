"""
Transformer Block 模块 -  Hyper-Connections 实现

集成 Hyper-Connections (HC) 流形超连接，维护 hc_mult 个并行残差流，
通过 Sinkhorn 正则化学习最优混合权重，解决超深 MoE 模型梯度消失问题。

结构：
    Input [b,s,hc,d] → hc_pre → [b,s,d] → RMSNorm → MLA → hc_post → [b,s,hc,d]
                     ↓ residual [b,s,hc,d]                     ↑
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelArgs
from .layers import RMSNorm
from .mla_attention import MLA
from .moe import MoE
from .kernel import hc_split_sinkhorn


class Block(nn.Module):
    """
    DeepSeek-V4 Transformer Block，带 Hyper-Connections (HC) 混合。

    不同于标准残差连接，HC 维护 hc_mult 个并行残差流副本：
    - hc_pre: 通过 Sinkhorn 正则化将 hc_mult 个副本混合为 1 个输入子层
    - hc_post: 将子层输出扩展回 hc_mult 个副本，并与残差流混合

    属性:
        layer_id: 层索引
        attn (MLA): 多头潜在注意力（集成 CSA + HCA）
        ffn (MoE):  混合专家前馈网络（所有层都用 MoE）
        attn_norm, ffn_norm (RMSNorm): Pre-Norm 归一化
        hc_mult: Hyper-Connections 倍数
        hc_attn_fn, hc_ffn_fn: 混合权重参数 [mix_hc, hc*dim]
        hc_attn_scale, hc_ffn_scale: 缩放参数 [3]
        hc_attn_base, hc_ffn_base: 偏置参数 [mix_hc]
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化带 Hyper-Connections 的 Transformer Block。

        Args:
            layer_id: 层索引（0-based）
            args: 模型参数，包含 hc_mult, hc_sinkhorn_iters, hc_eps 等 HC 配置
        """
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps

        # 注意力子层：MLA 集成 CSA 滑动窗口 + HCA 压缩稀疏
        self.attn = MLA(layer_id, args)

        # 前馈子层：所有层都用 MoE
        self.ffn = MoE(layer_id, args)

        # Pre-Norm 归一化层
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)

        # Hyper-Connections 配置
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps

        # mix_hc = (2 + hc_mult) * hc_mult = pre_size + post_size + comb_size
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim

        # HC 参数：学习残差流的混合权重
        # 使用 fp32 存储以保证数值稳定性
        with torch.no_grad():
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        # 初始化（官方使用正态分布或均匀分布）
        nn.init.normal_(self.hc_attn_fn, std=0.02)
        nn.init.normal_(self.hc_ffn_fn, std=0.02)
        nn.init.zeros_(self.hc_attn_base)
        nn.init.zeros_(self.hc_ffn_base)
        nn.init.ones_(self.hc_attn_scale)
        nn.init.ones_(self.hc_ffn_scale)

    def hc_pre(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Hyper-Connections 预处理：将 hc_mult 个副本混合为 1 个。

        Args:
            x: 输入 [batch, seq, hc_mult, dim]
            hc_fn: 混合函数参数 [mix_hc, hc_mult*dim]
            hc_scale: 缩放参数 [3] (pre/post/comb)
            hc_base: 偏置参数 [mix_hc]

        Returns:
            (y, post, comb)
            - y: 混合后的输入 [batch, seq, dim]
            - post: post 混合权重 [batch, seq, hc_mult]
            - comb: 组合矩阵 [batch, seq, hc_mult, hc_mult]
        """
        shape, dtype = x.size(), x.dtype  # [b, s, hc, d]

        # 展平为 [b, s, hc*dim]
        x_flat = x.flatten(2).float()

        # RMSNorm 归一化（保持数值稳定）
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)

        # 计算混合分数
        mixes = F.linear(x_flat, hc_fn) * rsqrt  # [b, s, mix_hc]

        # Sinkhorn 正则化分割为 pre, post, comb
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )

        # pre: [b, s, hc] 用于将多副本混合为 1 个
        # x.view(shape): [b, s, hc, d]
        # pre.unsqueeze(-1): [b, s, hc, 1]
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)  # [b, s, d]

        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor
    ) -> Tensor:
        """
        Hyper-Connections 后处理：将 1 个扩展回 hc_mult 个副本，并与残差混合。

        Args:
            x: 子层输出 [batch, seq, dim]
            residual: 残差流 [batch, seq, hc_mult, dim]
            post: post 混合权重 [batch, seq, hc_mult]
            comb: 组合矩阵 [batch, seq, hc_mult, hc_mult]

        Returns:
            y: 混合后的输出 [batch, seq, hc_mult, dim]
        """
        # post.unsqueeze(-1): [b, s, hc, 1] * x.unsqueeze(-2): [b, s, 1, d] -> [b, s, hc, d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2)

        # comb.unsqueeze(-1): [b, s, hc, hc, 1] * residual.unsqueeze(-2): [b, s, 1, hc, d]
        # -> sum(dim=2) -> [b, s, hc, d]
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

        return y.type_as(x)

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        input_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        前向传播，带 Hyper-Connections。

        Args:
            x: 输入张量 [batch, seq, hc_mult, dim]
            start_pos: 当前序列起始位置（用于 KV 缓存）
            input_ids: 输入 token IDs（用于 MoE 路由，可选）

        Returns:
            输出张量 [batch, seq, hc_mult, dim]
        """
        # ---------- Attention 子层 ----------
        residual = x  # [b, s, hc, d]

        # HC 预处理：hc_mult 个副本 -> 1 个
        x, post, comb = self.hc_pre(
            residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )  # x: [b, s, d]

        # Pre-Norm + MLA
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)  # [b, s, d]

        # HC 后处理：1 个 -> hc_mult 个副本，与残差混合
        x = self.hc_post(x, residual, post, comb)  # [b, s, hc, d]

        # ---------- MoE 子层 ----------
        residual = x  # [b, s, hc, d]

        # HC 预处理
        x, post, comb = self.hc_pre(
            residual, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )  # x: [b, s, d]

        # Pre-Norm + MoE
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)  # [b, s, d]

        # HC 后处理
        x = self.hc_post(x, residual, post, comb)  # [b, s, hc, d]

        return x
