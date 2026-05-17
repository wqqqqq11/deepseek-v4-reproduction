"""
MLA (Multi-Head Latent Attention) - 阶段1简化版

保留核心低秩 KV 压缩机制，简化稀疏注意力计算：
    1. Query 低秩压缩（q_lora_rank）
    2. KV 联合低秩压缩（kv_lora_rank）
    3. 标准分组查询注意力（GQA），无稀疏索引

适合小模型阶段1预训练，后续可替换完整版 MLA。
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelArgs
from .layers import RMSNorm
from .rotary_embedding import apply_rotary_emb, precompute_freqs_cis


class MLAStage1(nn.Module):
    """
    阶段1 MLA 简化实现。

    核心特征：
        - Query 低秩压缩：dim → q_lora_rank → n_heads * head_dim
        - KV 联合压缩：dim → kv_lora_rank (+ rope_dim) → n_heads * nope_head_dim
        - 标准 GQA 注意力（无 CSA/HCA 稀疏优化）
        - 支持 KV 缓存（压缩后的 latent 向量）

    Args:
        args: 模型配置参数。

    Example:
        >>> mla = MLAStage1(args)
        >>> out = mla(x, start_pos=0)  # x: [batch, seq, dim]
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads  # 单卡训练

        # 低秩压缩维度
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank

        # 头维度
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim

        # Query 投影（低秩）
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # KV 投影（联合低秩压缩）
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.norm_eps)
        # K_nope 和 V 分别投影
        self.wk_b = nn.Linear(self.kv_lora_rank, self.n_heads * self.nope_head_dim, bias=False)
        self.wv_b = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)

        # 输出投影
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        # 注意力缩放
        self.softmax_scale = self.head_dim ** -0.5

        # KV 缓存（压缩后的 latent 维度）
        cache_dim = self.kv_lora_rank + self.rope_head_dim
        max_cache_len = args.max_seq_len
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, max_cache_len, cache_dim),
            persistent=False
        )

        # RoPE 频率缓存
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            args.original_seq_len,
            args.rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 [batch_size, seq_len, dim]。
            start_pos: 序列起始位置（用于 KV 缓存）。

        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, dim]。
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # 获取 RoPE 频率
        freqs_cis = self.freqs_cis[start_pos:end_pos]

        # ---------- Query 投影 ----------
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # Query RMSNorm（DeepSeek-V4 特有，使用 FP32 数值稳定计算）
        q_norm = q.to(torch.float32).norm(dim=-1, keepdim=True)
        q = (q / (q_norm + 1e-6) * (self.head_dim ** 0.5)).to(q.dtype)

        # 应用 RoPE（分离 nope 和 rope 部分）
        q_nope = q[..., :self.nope_head_dim]
        q_pe = apply_rotary_emb(q[..., self.nope_head_dim:], freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # ---------- KV 投影 ----------
        kv = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)

        # 合并 KV 缓存
        kv_full = torch.cat([kv_latent, k_pe], dim=-1)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_full

        # K = [latent @ wk_b, k_pe_rot]
        k_nope = self.wk_b(kv_latent).view(bsz, seqlen, self.n_local_heads, self.nope_head_dim)
        
        # K 的 RoPE：先扩展 head 维度，再应用 RoPE
        k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, self.n_local_heads, -1)
        k_pe_rot = apply_rotary_emb(k_pe_expanded, freqs_cis)
        k = torch.cat([k_nope, k_pe_rot], dim=-1)

        # V = latent @ wv_b
        v = self.wv_b(kv_latent).view(bsz, seqlen, self.n_local_heads, self.v_head_dim)

        # ---------- 注意力计算 ----------
        q = q.transpose(1, 2)  # [b, n_heads, s, head_dim]
        k = k.transpose(1, 2)  # [b, n_heads, s, head_dim]
        v = v.transpose(1, 2)  # [b, n_heads, s, v_head_dim]

        # 标准缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)

        # 加权求和
        out = torch.matmul(attn, v)  # [b, n_heads, s, v_head_dim]
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # ---------- 输出投影 ----------
        return self.wo(out)

    def get_kv_cache_size(self) -> int:
        """获取 KV 缓存大小（以元素计）"""
        return self.kv_cache.numel()
