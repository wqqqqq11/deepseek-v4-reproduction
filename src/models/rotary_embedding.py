# src/models/rotary_embedding.py
"""
旋转位置编码（RoPE）模块

提供 YaRN 扩展的 RoPE 预计算和应用函数，用于在注意力机制中注入位置信息。
"""

import math
import torch
from .config import ModelArgs


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算旋转位置编码的复指数值（freqs_cis）。

    当 max_seq_len > original_seq_len 时，使用 YaRN 插值方法平滑扩展频率。

    Args:
        args: 模型参数，包含 qk_rope_head_dim、max_seq_len、original_seq_len、
              beta_fast、beta_slow、rope_theta、rope_factor 等。

    Returns:
        torch.Tensor: 形状 (max_seq_len, qk_rope_head_dim//2) 的复指数张量。
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码应用到输入张量上。

    Args:
        x:         输入张量，形状 (batch_size, seq_len, n_heads, head_dim)。
        freqs_cis: 预计算的复指数位置编码。

    Returns:
        torch.Tensor: 应用 RoPE 后的张量，形状与输入一致。
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)
