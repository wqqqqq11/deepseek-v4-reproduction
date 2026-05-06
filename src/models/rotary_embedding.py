# src/models/rotary_embedding.py
"""
旋转位置编码（RoPE）模块

提供 YaRN 扩展的 RoPE 预计算和应用函数，用于在注意力机制中注入位置信息。
"""

import math
from functools import lru_cache
from typing import Tuple
import torch


@lru_cache(maxsize=4)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """
    预计算旋转位置编码的复指数值（freqs_cis）。

    当 seqlen > original_seq_len 时，使用 YaRN 插值方法平滑扩展频率。
    使用 LRU 缓存避免重复计算相同参数的 freqs_cis。

    Args:
        dim: 位置编码维度（通常为 qk_rope_head_dim）。
        seqlen: 最大序列长度。
        original_seq_len: 预训练的原始序列长度（用于 YaRN 插值）。
        base: RoPE 基础频率 theta。
        factor: YaRN 扩展因子。
        beta_fast: YaRN 快速修正参数。
        beta_slow: YaRN 慢速修正参数。

    Returns:
        torch.Tensor: 形状 (seqlen, dim//2) 的复指数张量。
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim_val):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim_val, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN 长序列扩展
    if original_seq_len > 0 and seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """
    将旋转位置编码应用到输入张量上。

    Args:
        x: 输入张量，形状 (..., head_dim)，其中 head_dim 为偶数。
        freqs_cis: 预计算的复指数位置编码，形状 (seq_len, head_dim//2)。
        inverse: 是否进行逆旋转（de-rotation），用于某些特殊场景。

    Returns:
        torch.Tensor: 应用 RoPE 后的张量，形状与输入一致。
    """
    dtype = x.dtype

    # 将最后一维拆分为复数对 (real, imag)
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))

    # 逆旋转：使用共轭复数
    if inverse:
        freqs_cis = freqs_cis.conj()

    # 调整 freqs_cis 形状以匹配 x
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))

    # 复数乘法并展平回实数，返回新张量（非原地操作）
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(dtype)
