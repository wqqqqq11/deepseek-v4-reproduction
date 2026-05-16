import torch
import torch.nn.functional as F
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean with an optional boolean/float mask."""
    if mask is None:
        return values.mean()

    mask = mask.to(device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom

# 交叉熵损失函数--基础
def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} and labels shape "
            f"{tuple(labels.shape)} are not aligned."
        )

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )

# 自回归大语言模型损失函数----这里要注意后续适配训练代码的shift，确保输入和标签对齐
def language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    自回归语言模型 loss。
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
    """
    return cross_entropy_loss(logits, labels, ignore_index)

# SFT loss，本质是带 ignore_index 的 CE
def sft_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """
    Supervised fine-tuning loss.

    labels 中的 prompt/system/user 部分应提前置为 ignore_index，只让
    assistant response token 参与交叉熵。
    """
    return cross_entropy_loss(logits, labels, ignore_index)

# Multi-Token Prediction loss
def build_mtp_labels(
    labels: Tensor,
    depth: int,
    ignore_index: int = -100,
) -> Tensor:
    """
    Build future-token labels for Multi-Token Prediction.

    labels: [batch, seq_len]
    returns: [batch, seq_len, depth]

    depth 0 predicts t+1, depth 1 predicts t+2, and so on.
    """
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq_len].")
    if depth <= 0:
        raise ValueError("depth must be positive.")

    bsz, seqlen = labels.shape
    out = labels.new_full((bsz, seqlen, depth), ignore_index)
    for i in range(depth):
        shift = i + 1
        if shift < seqlen:
            out[:, :-shift, i] = labels[:, shift:]
    return out


def mtp_loss(
    mtp_logits: Tensor | list[Tensor] | tuple[Tensor, ...],
    mtp_labels: Tensor | list[Tensor] | tuple[Tensor, ...],
    loss_weight: float = 0.3,
    ignore_index: int = -100,
) -> Tensor:
    """
    Multi-Token Prediction loss.

    支持两种输入：
    - mtp_logits 为 list/tuple，每个元素形状 [batch, seq_len, vocab_size]
    - mtp_logits 为 Tensor:
        [batch, seq_len, vocab_size] 或 [batch, seq_len, depth, vocab_size]

    mtp_labels 可以是：
    - [batch, seq_len]
    - [batch, seq_len, depth]
    - list/tuple，与 mtp_logits 一一对应
    """
    if isinstance(mtp_logits, torch.Tensor):
        if mtp_logits.ndim == 4:
            logits_list = [mtp_logits[:, :, i, :] for i in range(mtp_logits.size(2))]
        elif mtp_logits.ndim == 3:
            logits_list = [mtp_logits]
        else:
            raise ValueError("mtp_logits must have shape [b,s,v] or [b,s,depth,v].")
    else:
        logits_list = list(mtp_logits)

    if isinstance(mtp_labels, torch.Tensor):
        if mtp_labels.ndim == 3:
            labels_list = [mtp_labels[:, :, i] for i in range(mtp_labels.size(2))]
        elif mtp_labels.ndim == 2:
            future_labels = build_mtp_labels(mtp_labels, len(logits_list), ignore_index)
            labels_list = [future_labels[:, :, i] for i in range(future_labels.size(2))]
        else:
            raise ValueError("mtp_labels must have shape [b,s] or [b,s,depth].")
    else:
        labels_list = list(mtp_labels)

    if len(logits_list) != len(labels_list):
        raise ValueError("mtp_logits and mtp_labels must have the same number of heads.")

    losses = [
        cross_entropy_loss(logits, labels, ignore_index)
        for logits, labels in zip(logits_list, labels_list)
    ]
    return loss_weight * torch.stack(losses).mean()

# MoE 专家负载均衡 loss
def moe_balance_loss(
    router_probs: Tensor | None = None,
    expert_indices: Tensor | None = None,
    num_experts: int | None = None,
    attention_mask: Tensor | None = None,
    loss_weight: float = 1e-4,
    eps: float = 1e-9,
) -> Tensor:
    """
    DeepSeek-V4-like lightweight sequence-wise MoE balance loss.

    说明：
    1. router_probs 分支是可导的，训练时使用。
       router_probs: [batch, seq_len, num_experts]

    2. expert_indices 分支不可导，作为 fallback / metric。
       expert_indices: [batch, seq_len] 或 [batch, seq_len, top_k]

    3. 这个损失函数不是完整的 auxiliary-loss-free routing。
       DeepSeek-V4 的 auxiliary-loss-free 在 router bias / routing 策略里实现。
       这里实现的是论文中提到的 slight sequence-wise balance loss。
    """
    if router_probs is None and expert_indices is None:
        raise ValueError("Either router_probs or expert_indices must be provided.")

    # ============================================================
    # 1. 可导版本：基于 router_probs 的 sequence-wise balance loss
    # ============================================================
    if router_probs is not None:
        if router_probs.ndim < 3:
            raise ValueError(
                "router_probs should have shape [batch, seq_len, num_experts]."
            )

        if num_experts is None:
            num_experts = router_probs.size(-1)

        if router_probs.size(-1) != num_experts:
            raise ValueError(
                f"router_probs last dim {router_probs.size(-1)} != num_experts {num_experts}"
            )

        probs = router_probs.float()  # [B, S, E]

        # attention_mask: [B, S]
        if attention_mask is None:
            mask = torch.ones(
                probs.shape[:-1],
                device=probs.device,
                dtype=probs.dtype,
            )
        else:
            mask = attention_mask.to(device=probs.device, dtype=probs.dtype)

        # 扩展成 [B, S, 1]
        while mask.ndim < probs.ndim:
            mask = mask.unsqueeze(-1)

        # token 维度，一般是 seq_len 维
        token_dims = tuple(range(1, probs.ndim - 1))

        # 每条 sequence 的有效 token 数: [B, 1]
        token_count = mask.sum(dim=token_dims).clamp_min(eps)

        # 每条 sequence 中，每个 expert 获得的概率质量: [B, E]
        seq_load = (probs * mask).sum(dim=token_dims) / token_count

        # 如果某条 sequence 全是 padding，则不参与 loss
        valid_seq = token_count.squeeze(-1) > eps
        if not valid_seq.any():
            return probs.new_zeros(())

        seq_load = seq_load[valid_seq]  # [valid_B, E]

        # 均匀目标分布
        uniform = torch.full_like(seq_load, 1.0 / num_experts)

        # 每条 sequence 单独算均衡 loss，然后 batch 平均
        loss = num_experts * (seq_load - uniform).pow(2).sum(dim=-1)

        return loss_weight * loss.mean()

    # ============================================================
    # 2. 不可导版本：基于 expert_indices 的统计型 balance loss
    # ============================================================
    assert expert_indices is not None

    if num_experts is None:
        valid_indices = expert_indices[expert_indices >= 0]
        if valid_indices.numel() == 0:
            return torch.zeros((), device=expert_indices.device, dtype=torch.float32)
        num_experts = int(valid_indices.max().item()) + 1

    indices = expert_indices.to(dtype=torch.long)

    if indices.ndim < 2:
        raise ValueError(
            "expert_indices should have shape [batch, seq_len] or [batch, seq_len, top_k]."
        )

    # attention_mask: [B, S]
    if attention_mask is None:
        valid_mask = torch.ones_like(indices, dtype=torch.float32)
    else:
        valid_mask = attention_mask.to(device=indices.device, dtype=torch.float32)

        # 如果 expert_indices 是 [B, S, top_k]，mask 要扩展到 [B, S, top_k]
        while valid_mask.ndim < indices.ndim:
            valid_mask = valid_mask.unsqueeze(-1)

        valid_mask = valid_mask.expand_as(indices)

    # 防止 -1 或越界 index 影响 one_hot
    index_valid = (indices >= 0) & (indices < num_experts)
    valid_mask = valid_mask * index_valid.to(dtype=valid_mask.dtype)

    if valid_mask.sum() == 0:
        return torch.zeros((), device=indices.device, dtype=torch.float32)

    safe_indices = indices.clamp(min=0, max=num_experts - 1)

    # [B, S, E] 或 [B, S, top_k, E]
    one_hot = F.one_hot(safe_indices, num_classes=num_experts).float()
    one_hot = one_hot * valid_mask.unsqueeze(-1)

    # 除 batch 和 expert 维之外，其余都当作 token/selection 维度
    reduce_dims = tuple(range(1, one_hot.ndim - 1))

    # [B, E]
    counts = one_hot.sum(dim=reduce_dims)
    denom = valid_mask.sum(dim=tuple(range(1, valid_mask.ndim))).clamp_min(eps)

    # [B, E]
    seq_load = counts / denom.unsqueeze(-1)

    valid_seq = denom > eps
    if not valid_seq.any():
        return torch.zeros((), device=indices.device, dtype=torch.float32)

    seq_load = seq_load[valid_seq]

    uniform = torch.full_like(seq_load, 1.0 / num_experts)
    loss = num_experts * (seq_load - uniform).pow(2).sum(dim=-1)

    return loss_weight * loss.mean()

# OPD full-vocab reverse KL 蒸馏 loss---
def opd_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    teacher_weights: Tensor | None = None,
    temperature: float = 1.0,
    loss_mask: Tensor | None = None,
    labels: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """
    On-Policy Distillation full-vocabulary reverse KL loss.

    Computes KL(student || teacher). teacher_logits can be:
    - [batch, seq_len, vocab_size]
    - [num_teachers, batch, seq_len, vocab_size]
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if labels is not None and loss_mask is None:
        loss_mask = labels.ne(ignore_index)

    if teacher_logits.ndim == student_logits.ndim:
        teachers = teacher_logits.unsqueeze(0)
    elif teacher_logits.ndim == student_logits.ndim + 1:
        teachers = teacher_logits
    else:
        raise ValueError("teacher_logits must be [b,s,v] or [num_teachers,b,s,v].")

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    student_probs = student_log_probs.exp()

    per_teacher_losses = []
    for teacher in teachers:
        teacher_log_probs = F.log_softmax(teacher.detach().float() / temperature, dim=-1)
        kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        kl = kl * (temperature ** 2)
        per_teacher_losses.append(_masked_mean(kl, loss_mask))

    losses = torch.stack(per_teacher_losses)
    if teacher_weights is None:
        return losses.mean()

    weights = teacher_weights.to(device=losses.device, dtype=losses.dtype)
    weights = weights / weights.sum().clamp_min(1e-9)
    return (losses * weights).sum()

# grpo_loss-选择性打分loss
def grpo_loss(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    response_mask: Tensor | None = None,
    ref_logprobs: Tensor | None = None,
    kl_coef: float = 0.0,
    clip_ratio: float = 0.2,
) -> Tensor:
    """
    GRPO/PPO-style clipped policy loss with optional reference KL penalty.

    new_logprobs/old_logprobs/ref_logprobs: [batch, seq_len]
    advantages: [batch] or [batch, seq_len]
    response_mask: [batch, seq_len], masks padding/prompt tokens.
    """
    if advantages.ndim == new_logprobs.ndim - 1:
        advantages = advantages.unsqueeze(-1)
    advantages = advantages.to(device=new_logprobs.device, dtype=new_logprobs.dtype)

    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    per_token_loss = -torch.minimum(unclipped, clipped)

    if ref_logprobs is not None and kl_coef > 0:
        log_ratio = ref_logprobs - new_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        per_token_loss = per_token_loss + kl_coef * kl

    if response_mask is None:
        return per_token_loss.mean()

    mask = response_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
    response_token_counts = mask.sum(dim=-1).clamp_min(1.0)
    response_losses = (per_token_loss * mask).sum(dim=-1) / response_token_counts
    valid_responses = response_mask.to(device=per_token_loss.device).any(dim=-1)
    return response_losses[valid_responses].mean()
