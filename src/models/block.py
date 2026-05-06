"""
Transformer Block 模块

将多头潜在注意力（MLA）和前馈网络（MLP / MoE）组装为一个完整的 Transformer 层，
使用 Pre-Norm 残差结构。
"""

from typing import Optional
import torch
from torch import nn
from .config import ModelArgs
from .layers import RMSNorm
from .mhc_attention import MLA
from .moe import MLP, MoE


class Block(nn.Module):
    """
    Transformer 的基本构建块，由注意力子层和前馈子层组成。

    采用 Pre-Norm 残差结构：
        x = x + Attention( Norm(x) )
        x = x + FFN( Norm(x) )

    前几层使用密集 MLP，后续层使用 MoE（混合专家）。

    属性:
        attn (MLA):          多头潜在注意力层。
        ffn (MLP | MoE):     前馈网络（密集层为 MLP，稀疏层为 MoE）。
        attn_norm (RMSNorm): 注意力前的归一化。
        ffn_norm (RMSNorm):  前馈网络前的归一化。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化 Transformer Block。

        Args:
            layer_id: 层索引（0-based）。用于决定使用密集 MLP 还是 MoE：
                      前 n_dense_layers 层使用 MLP，其余使用 MoE。
            args:     模型参数。
        """
        super().__init__()
        # 注意力子层
        self.attn = MLA(args)
        # 前馈子层：前 n_dense_layers 层用密集 MLP，后面的用 MoE
        self.ffn = (
            MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        )
        # 两个 Pre-Norm 归一化层
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        前向传播。

        1. Pre-Norm → 注意力 → 残差连接
        2. Pre-Norm → 前馈网络 → 残差连接

        Args:
            x:          输入张量，形状 (batch_size, seq_len, dim)。
            start_pos:  当前序列起始位置（用于 KV 缓存）。
            freqs_cis:  预计算的 RoPE 复指数位置编码。
            mask:       注意力 mask。

        Returns:
            同形状输出张量。
        """
        # 注意力子层：Pre-Norm + 残差连接
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # 前馈子层：Pre-Norm + 残差连接
        x = x + self.ffn(self.ffn_norm(x))
        return x