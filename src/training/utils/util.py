"""
训练工具函数

提供模型训练相关的通用工具函数：
    - 梯度裁剪
    - 参数统计
    - 随机种子设置
    - 检查点保存/加载
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def set_seed(seed: int) -> None:
    """
    设置全局随机种子，确保可复现性。
    
    Args:
        seed: 随机种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    全局梯度裁剪。
    
    Args:
        model: 待裁剪梯度的模型。
        max_norm: 最大梯度范数。
        
    Returns:
        float: 裁剪前的梯度范数。
    """
    return clip_grad_norm_(model.parameters(), max_norm)


def get_grad_norm(model: nn.Module) -> float:
    """
    计算全局梯度范数。
    
    Args:
        model: 模型。
        
    Returns:
        float: 梯度范数。
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量。
    
    Args:
        model: 模型。
        
    Returns:
        Tuple[int, int]: (总参数数, 可训练参数数)。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型大小统计信息。
    
    Args:
        model: 模型。
        
    Returns:
        Dict[str, Any]: 包含参数数量、内存占用等信息的字典。
    """
    total_params, trainable_params = count_parameters(model)
    
    # 估算内存占用（假设 fp16/bf16）
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 ** 2)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": round(param_mb, 2),
        "estimated_size_mb": round(param_mb * 3, 2),  # 参数 + 梯度 + 优化器状态
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    is_best: bool = False,
) -> None:
    """
    保存训练检查点。
    
    Args:
        model: 模型。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        epoch: 当前 epoch。
        step: 当前步数。
        loss: 当前损失。
        path: 保存路径。
        is_best: 是否为最优模型。
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    
    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    torch.save(checkpoint, path)
    
    # 如果是最佳模型，保存副本
    if is_best:
        best_path = Path(path).parent / "best.pt"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Any,
    path: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    加载训练检查点。
    
    Args:
        model: 模型。
        optimizer: 优化器（可选）。
        scheduler: 学习率调度器（可选）。
        path: 检查点路径。
        device: 加载设备。
        
    Returns:
        Dict[str, Any]: 包含 epoch, step, loss 等信息的字典。
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
    }


def compute_perplexity(loss: float) -> float:
    """
    从交叉熵损失计算困惑度。
    
    Args:
        loss: 平均交叉熵损失。
        
    Returns:
        float: 困惑度（PPL）。
    """
    return np.exp(loss)


def format_number(num: int) -> str:
    """
    格式化大数字显示。
    
    Args:
        num: 数字。
        
    Returns:
        str: 格式化字符串（如 "1.2M", "500K"）。
    """
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    else:
        return str(num)


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息。
    
    Returns:
        Dict[str, Any]: 设备信息字典。
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info
