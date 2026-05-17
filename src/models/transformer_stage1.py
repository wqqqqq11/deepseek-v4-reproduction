"""
阶段1 Transformer 模型（Dense 版本）

简化版 Transformer，专为阶段1预训练设计：
    - 无 Hyper-Connections（hc_mult=1）
    - 无 MoE，使用 Dense SwiGLU FFN
    - 使用简化版 MLA（低秩 KV 压缩）
    - 支持 MTP（Multi-Token Prediction）

结构：
    Embedding → [RMSNorm → MLA → RMSNorm → DenseMLP] × n_layers
            → RMSNorm → Head → Logits
"""

import torch
import torch.nn as nn
from .config import ModelArgs
from .layers import RMSNorm
from .mla_attention_stage1 import MLAStage1
from .dense_mlp import DenseMLP


class BlockStage1(nn.Module):
    """
    阶段1 Transformer Block。

    结构（Pre-Norm）：
        x_norm = RMSNorm(x)
        h = x + MLA(x_norm)
        h_norm = RMSNorm(h)
        out = h + DenseMLP(h_norm)

    Args:
        args: 模型配置参数。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = MLAStage1(args)
        self.mlp_norm = RMSNorm(args.dim, args.norm_eps)
        self.mlp = DenseMLP(args.dim, args.inter_dim)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 [batch, seq, dim]。
            start_pos: 序列起始位置（用于 KV 缓存）。

        Returns:
            torch.Tensor: 输出张量 [batch, seq, dim]。
        """
        # Attention 子层
        h = x + self.attn(self.attn_norm(x), start_pos)

        # FFN 子层
        out = h + self.mlp(self.mlp_norm(h))

        return out


class TransformerStage1(nn.Module):
    """
    阶段1 Transformer 模型（Dense 版本）。

    结构：
        tokens → Embedding → [Block × n_layers] → RMSNorm → Head → logits

    特性：
        - 纯 Dense 结构，无 MoE
        - 支持 MTP（多令牌预测）
        - 可选 FP8 量化

    Args:
        args: 模型配置参数。

    Example:
        >>> model = TransformerStage1(args)
        >>> logits, mtp_logits = model(tokens)  # 训练模式
        >>> logits = model(tokens)  # 推理模式
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim

        # 词嵌入
        self.embed = nn.Embedding(args.vocab_size, args.dim)

        # Transformer 层
        self.layers = nn.ModuleList([
            BlockStage1(args) for _ in range(args.n_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(args.dim, args.norm_eps)

        # 输出头（与嵌入共享或不共享）
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 可选：MTP（Multi-Token Prediction）头
        self.mtp_enabled = getattr(args, "mtp_num_future_tokens", 0) > 0
        if self.mtp_enabled:
            self.mtp_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 权重初始化
        self._init_weights()

        # 可选：共享嵌入和输出头权重
        if getattr(args, "tie_word_embeddings", True):
            self.head.weight = self.embed.weight

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        return_mtp: bool = False
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            tokens: 输入 token ID [batch_size, seq_len]。
            start_pos: 序列起始位置（用于 KV 缓存）。
            return_mtp: 是否返回 MTP 预测结果。

        Returns:
            torch.Tensor: 主 logits [batch_size, seq_len, vocab_size]。
            如果 return_mtp=True，还返回 mtp_logits。
        """
        batch_size, seq_len = tokens.shape

        # 词嵌入
        h = self.embed(tokens)  # [b, s, dim]

        # 逐层传递
        for layer in self.layers:
            h = layer(h, start_pos)

        # 最终归一化
        h = self.norm(h)  # [b, s, dim]

        # 主输出头
        logits = self.head(h)  # [b, s, vocab_size]

        # MTP 输出（可选）
        if return_mtp and self.mtp_enabled:
            mtp_logits = self.mtp_head(h)
            return logits, mtp_logits

        return logits

    def get_num_params(self) -> int:
        """获取模型总参数量"""
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        估算模型浮点利用率（MFU）。

        Args:
            fwdbwd_per_iter: 每次迭代的浮点运算数。
            dt: 每次迭代的时间（秒）。

        Returns:
            float: MFU 百分比。
        """
        # 假设使用 A100（312 TFLOPS bf16）
        flops_per_sec = fwdbwd_per_iter / dt
        a100_flops = 312e12
        return flops_per_sec / a100_flops


# 别名
DeepSeekV4Stage1 = TransformerStage1
