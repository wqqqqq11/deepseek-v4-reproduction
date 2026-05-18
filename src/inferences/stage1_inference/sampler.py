"""
采样策略模块

提供多种采样方式：greedy、temperature、top-k、top-p
"""

import torch
import torch.nn.functional as F


def greedy_sample(logits):
    """贪心采样：选择概率最高的token。"""
    return torch.argmax(logits, dim=-1).item()


def temperature_sample(logits, temperature=1.0):
    """温度采样：调节分布尖锐程度。"""
    if temperature == 0:
        return greedy_sample(logits)

    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def top_k_sample(logits, k=50, temperature=1.0):
    """Top-K采样：只从概率最高的K个中选择。"""
    if k <= 0:
        return temperature_sample(logits, temperature)

    values, indices = torch.topk(logits, min(k, logits.size(-1)))
    probs = F.softmax(values / temperature, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()

    return indices[idx].item()


def top_p_sample(logits, p=0.9, temperature=1.0):
    """Top-P (nucleus)采样：从累积概率≥P的最小集合中选择。"""
    if p >= 1.0:
        return temperature_sample(logits, temperature)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 找到累积概率超过p的位置
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[0] = False  # 至少保留第一个

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits_filtered = logits.clone()
    logits_filtered[indices_to_remove] = float('-inf')

    return temperature_sample(logits_filtered, temperature)


def sample_next_token(logits, strategy="greedy", temperature=1.0, top_k=None, top_p=None):
    """
    统一的token采样接口。

    Args:
        logits: 模型输出的logits，形状 [vocab_size]。
        strategy: 采样策略，"greedy" | "temperature" | "top_k" | "top_p"。
        temperature: 温度参数。
        top_k: Top-K参数。
        top_p: Top-P参数。

    Returns:
        int: 采样的token ID。
    """
    logits = logits.float()

    if strategy == "greedy":
        return greedy_sample(logits)
    elif strategy == "temperature":
        return temperature_sample(logits, temperature)
    elif strategy == "top_k":
        return top_k_sample(logits, top_k or 50, temperature)
    elif strategy == "top_p":
        return top_p_sample(logits, top_p or 0.9, temperature)
    else:
        return temperature_sample(logits, temperature)
