"""
三段式学习率调度器

实现 DeepSeek-V4 学习率调度策略：
    阶段1：线性预热（warmup）
    阶段2：恒定峰值（constant peak）
    阶段3：余弦衰减（cosine decay）

同时支持 MTP 损失权重的同步衰减。
"""

import math
from typing import Optional, Dict, Any
import torch
from torch.optim import Optimizer


class ThreeStageScheduler:
    """
    三段式学习率调度器。
    
    调度策略：
        1. [0, warmup_steps): 从 0 线性增长到 max_lr
        2. [warmup_steps, warmup + peak): 保持 max_lr 恒定
        3. [warmup + peak, total): 从 max_lr 余弦衰减到 min_lr
    
    Args:
        optimizer: 要调度的优化器。
        max_lr: 最大学习率（峰值）。
        min_lr: 最小学习率（衰减终点）。
        warmup_steps: 预热步数。
        peak_steps: 峰值恒定步数。
        total_steps: 总步数（用于计算衰减阶段）。
        mtp_initial_weight: MTP 损失初始权重。
        mtp_final_weight: MTP 损失最终权重。
    
    Example:
        >>> scheduler = ThreeStageScheduler(
        ...     optimizer,
        ...     max_lr=3e-4,
        ...     min_lr=3e-5,
        ...     warmup_steps=2000,
        ...     peak_steps=6000,
        ...     total_steps=20000,
        ... )
        >>> for step in range(total_steps):
        ...     train_step()
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5,
        warmup_steps: int = 2000,
        peak_steps: int = 6000,
        total_steps: int = 20000,
        mtp_initial_weight: float = 0.3,
        mtp_final_weight: float = 0.1,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.peak_steps = peak_steps
        self.cosine_steps = total_steps - warmup_steps - peak_steps
        self.total_steps = total_steps
        
        self.mtp_initial_weight = mtp_initial_weight
        self.mtp_final_weight = mtp_final_weight
        
        self.current_step = 0
        self.current_lr = 0.0
        self.current_mtp_weight = mtp_initial_weight
    
    def _compute_lr(self, step: int) -> float:
        """
        计算当前步的学习率。
        
        Args:
            step: 当前步数。
            
        Returns:
            float: 当前学习率。
        """
        if step < self.warmup_steps:
            # 阶段1：线性预热
            return self.max_lr * (step / self.warmup_steps)
        
        elif step < self.warmup_steps + self.peak_steps:
            # 阶段2：恒定峰值
            return self.max_lr
        
        else:
            # 阶段3：余弦衰减
            progress = (step - self.warmup_steps - self.peak_steps) / max(1, self.cosine_steps)
            progress = min(progress, 1.0)  # 防止超出
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
    
    def _compute_mtp_weight(self, step: int) -> float:
        """
        计算当前步的 MTP 损失权重。
        
        在衰减阶段，MTP 权重从初始值线性降到最终值。
        
        Args:
            step: 当前步数。
            
        Returns:
            float: 当前 MTP 权重。
        """
        decay_start = self.warmup_steps + self.peak_steps
        
        if step < decay_start:
            # 预热和峰值阶段：保持初始权重
            return self.mtp_initial_weight
        else:
            # 衰减阶段：线性衰减
            progress = (step - decay_start) / max(1, self.cosine_steps)
            progress = min(progress, 1.0)
            
            return self.mtp_initial_weight + (
                self.mtp_final_weight - self.mtp_initial_weight
            ) * progress
    
    def step(self) -> Dict[str, float]:
        """
        执行单步调度，更新优化器学习率。
        
        Returns:
            Dict[str, float]: 包含当前学习率和 MTP 权重的字典。
        """
        self.current_step += 1
        
        # 计算当前学习率
        self.current_lr = self._compute_lr(self.current_step)
        
        # 计算当前 MTP 权重
        self.current_mtp_weight = self._compute_mtp_weight(self.current_step)
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr
        
        return {
            "lr": self.current_lr,
            "mtp_weight": self.current_mtp_weight,
            "step": self.current_step,
        }
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr
    
    def get_mtp_weight(self) -> float:
        """获取当前 MTP 权重"""
        return self.current_mtp_weight
    
    def state_dict(self) -> Dict[str, Any]:
        """返回调度器状态字典"""
        return {
            "current_step": self.current_step,
            "current_lr": self.current_lr,
            "current_mtp_weight": self.current_mtp_weight,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "peak_steps": self.peak_steps,
            "cosine_steps": self.cosine_steps,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从状态字典恢复调度器"""
        self.current_step = state_dict["current_step"]
        self.current_lr = state_dict["current_lr"]
        self.current_mtp_weight = state_dict["current_mtp_weight"]
        
        # 恢复学习率到优化器
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 2000,
    peak_steps: int = 6000,
    cosine_steps: int = 12000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    mtp_initial_weight: float = 0.3,
    mtp_final_weight: float = 0.1,
) -> ThreeStageScheduler:
    """
    便捷创建三段式调度器。
    
    Args:
        optimizer: 要调度的优化器。
        warmup_steps: 预热步数。
        peak_steps: 峰值恒定步数。
        cosine_steps: 余弦衰减步数。
        max_lr: 最大学习率。
        min_lr: 最小学习率。
        mtp_initial_weight: MTP 初始权重。
        mtp_final_weight: MTP 最终权重。
        
    Returns:
        ThreeStageScheduler: 配置好的调度器。
    """
    total_steps = warmup_steps + peak_steps + cosine_steps
    
    return ThreeStageScheduler(
        optimizer=optimizer,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        peak_steps=peak_steps,
        total_steps=total_steps,
        mtp_initial_weight=mtp_initial_weight,
        mtp_final_weight=mtp_final_weight,
    )
