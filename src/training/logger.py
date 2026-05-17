"""
训练日志模块

整合 TensorBoard 和 tqdm，提供统一的训练监控接口：
    - 实时指标记录到 TensorBoard
    - 命令行进度条显示
    - 支持损失、学习率、梯度、硬件监控
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TrainingLogger:
    """
    训练日志记录器。
    
    整合 TensorBoard 和 tqdm 进度条，提供：
        - 每 step 指标记录（损失、学习率、速度）
        - 每 N steps 分布记录（参数/梯度直方图）
        - 实时进度条显示
    
    Args:
        log_dir: TensorBoard 日志目录。
        log_every_steps: 记录间隔步数（默认 10）。
        enable_tensorboard: 是否启用 TensorBoard（默认 True）。
    
    Example:
        >>> logger = TrainingLogger("logs/stage1")
        >>> for step in range(1000):
        ...     loss = train_step()
        ...     logger.log_step(step, {"loss": loss, "lr": 3e-4})
        >>> logger.close()
    """
    
    def __init__(
        self,
        log_dir: str,
        log_every_steps: int = 10,
        enable_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_steps = log_every_steps
        self.enable_tensorboard = enable_tensorboard
        
        # TensorBoard writer
        self.writer = None
        if enable_tensorboard:
            self.writer = SummaryWriter(log_dir)
        
        # 进度条
        self.pbar: Optional[tqdm] = None
        self.total_steps = 0
        
        # 时间记录
        self.start_time = time.time()
        self.step_time = self.start_time
        
        # 当前 epoch 状态
        self.current_epoch = 0
        self.epoch_start_step = 0
    
    def start_epoch(
        self,
        epoch: int,
        total_steps: int,
        initial_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        开始新 epoch，初始化进度条。
        
        Args:
            epoch: 当前 epoch 数。
            total_steps: 本 epoch 总步数。
            initial_metrics: 初始指标字典。
        """
        self.current_epoch = epoch
        self.epoch_start_step = self.total_steps
        self.total_steps = total_steps
        
        # 创建进度条
        desc = f"Epoch {epoch}"
        if initial_metrics:
            desc += f" | loss: {initial_metrics.get('loss', 0):.3f}"
        
        self.pbar = tqdm(
            total=total_steps,
            desc=desc,
            unit="step",
            ncols=100,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        
        # 重置计时
        self.step_time = time.time()
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        记录单步训练指标。
        
        Args:
            step: 当前步数（epoch 内）。
            metrics: 指标字典，可包含：
                - loss/total: 总损失
                - loss/lm: LM 损失
                - loss/mtp: MTP 损失
                - lr: 学习率
                - tokens_per_sec: 吞吐率
                - grad_norm: 梯度范数
            model: 可选模型，用于记录参数/梯度分布。
        """
        global_step = self.epoch_start_step + step
        
        # 更新进度条
        if self.pbar is not None:
            self._update_pbar(step, metrics)
        
        # 记录到 TensorBoard
        if self.writer is not None and step % self.log_every_steps == 0:
            self._write_scalars(global_step, metrics)
            
            # 每 10 个记录步记录分布
            if step % (self.log_every_steps * 10) == 0 and model is not None:
                self._write_distributions(global_step, model)
    
    def _update_pbar(self, step: int, metrics: Dict[str, float]) -> None:
        """更新进度条显示"""
        # 计算速度
        current_time = time.time()
        elapsed = current_time - self.step_time
        self.step_time = current_time
        
        tokens_per_sec = metrics.get("tokens_per_sec", 0)
        loss = metrics.get("loss/total", metrics.get("loss", 0))
        lr = metrics.get("lr", 0)
        
        # 构建 postfix
        postfix_parts = [
            f"loss: {loss:.3f}",
            f"lr: {lr:.2e}",
        ]
        
        if tokens_per_sec > 0:
            postfix_parts.append(f"tok/s: {tokens_per_sec:.0f}")
        
        if "grad_norm" in metrics:
            postfix_parts.append(f"gnorm: {metrics['grad_norm']:.2f}")
        
        if "gpu_memory" in metrics:
            postfix_parts.append(f"gpu: {metrics['gpu_memory']:.1f}GB")
        
        self.pbar.set_postfix_str(" | ".join(postfix_parts))
        self.pbar.update(1)
    
    def _write_scalars(self, global_step: int, metrics: Dict[str, float]) -> None:
        """写入标量指标到 TensorBoard"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, global_step)
    
    def _write_distributions(
        self,
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        """写入参数和梯度分布"""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 参数直方图
            try:
                if param.numel() > 0 and param.std() > 0:
                    self.writer.add_histogram(f"param/{name}", param, global_step)
            except (ValueError, RuntimeError):
                pass
            
            # 梯度直方图
            if param.grad is not None:
                try:
                    if param.grad.numel() > 0 and param.grad.std() > 0:
                        self.writer.add_histogram(
                            f"grad/{name}", param.grad, global_step
                        )
                except (ValueError, RuntimeError):
                    pass
    
    def log_eval(self, step: int, metrics: Dict[str, float]) -> None:
        """
        记录验证指标。
        
        Args:
            step: 当前步数。
            metrics: 验证指标字典，应包含 val/loss, val/ppl 等。
        """
        global_step = self.epoch_start_step + step
        
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step)
        
        # 打印验证结果
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if self.pbar:
            self.pbar.write(f"[Eval Step {step}] {metric_str}")
        else:
            print(f"[Eval Step {step}] {metric_str}")
    
    def end_epoch(self) -> None:
        """结束当前 epoch，关闭进度条"""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
    
    def close(self) -> None:
        """关闭日志记录器"""
        self.end_epoch()
        if self.writer is not None:
            self.writer.close()
    
    def get_log_dir(self) -> str:
        """获取日志目录路径"""
        return str(self.log_dir)


def get_gpu_memory() -> float:
    """
    获取当前 GPU 显存占用（GB）。
    
    Returns:
        float: 显存占用（GB）。
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def format_time(seconds: float) -> str:
    """
    格式化时间显示。
    
    Args:
        seconds: 秒数。
        
    Returns:
        str: 格式化字符串（如 "02:30"）。
    """
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"
