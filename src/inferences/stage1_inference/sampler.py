"""
采样模块

使用 Gumbel-max trick 实现 temperature 采样，避免 GPU-CPU 同步。
"""

import torch


def sample(logits, temperature: float = 0.6):
    """
    Temperature 采样（Gumbel-max trick）。

    Gumbel-max trick 等价于 multinomial 采样，但避免了 torch.multinomial
    中的 GPU 到 CPU 同步，在 GPU 上更快。

    Args:
        logits: 模型输出的 logits，形状 [vocab_size]。
        temperature: 温度参数，默认 0.6（参考官方代码）。

    Returns:
        int: 采样的 token ID。
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    # Gumbel-max: probs / exponential(1) 然后取 argmax
    noise = torch.empty_like(probs).exponential_(1)
    return (probs / noise).argmax(dim=-1).item()
