"""
训练日志模块

提供流式日志记录功能：
    - 同时输出到控制台和文件
    - 每 N 步记录训练指标
    - 标记 [VAL] 区分验证日志
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class TrainingLogger:
    """
    训练日志记录器。

    使用标准 logging 模块，同时输出到控制台和文件。

    Args:
        log_dir: 日志目录。
        log_every_steps: 记录间隔步数（默认 10）。
        prefix: 日志文件名前缀（默认 "train"）。
    """

    def __init__(
        self,
        log_dir: str,
        log_every_steps: int = 10,
        prefix: str = "train",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_steps = log_every_steps

        # 创建 logger
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        # 控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 文件 handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{prefix}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        self.log_file = log_file
        self.current_epoch = 0
        self.global_step = 0
        self.start_time = time.time()
        self.step_times = []

    def log_train(
        self,
        step: int,
        total_steps: int,
        metrics: Dict[str, Any],
    ) -> None:
        """
        记录训练日志。

        Args:
            step: 当前步数。
            total_steps: 总步数。
            metrics: 指标字典，包含 loss, lr, grad_norm, tok/s。
        """
        self.global_step += 1

        if step % self.log_every_steps != 0:
            return

        # 计算时间
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed / max(step, 1))
        avg_step_time = sum(self.step_times[-10:]) / len(self.step_times[-10:])
        remaining_steps = total_steps - step
        remaining_time = avg_step_time * remaining_steps

        # 格式化时间
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining_time)

        # 构建日志消息
        parts = [
            f"Epoch {self.current_epoch}",
            f"{step}/{total_steps} [{elapsed_str}<{remaining_str}]",
            f"loss: {metrics.get('loss', 0):.4f}",
            f"lr: {metrics.get('lr', 0):.2e}",
            f"gnorm: {metrics.get('grad_norm', 0):.2f}",
            f"tok/s: {metrics.get('tokens_per_sec', 0):.0f}",
        ]

        msg = " | ".join(parts)
        self.logger.info(msg)

    def _format_time(self, seconds: float) -> str:
        """格式化时间为 MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def log_eval(self, epoch: int, val_loss: float, val_ppl: float) -> None:
        """
        记录验证日志。

        Args:
            epoch: 当前 epoch。
            val_loss: 验证损失。
            val_ppl: 验证困惑度。
        """
        msg = f"[VAL] Epoch {epoch} | val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}"
        self.logger.info(msg)

    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """
        记录 epoch 结束信息。

        Args:
            epoch: 当前 epoch。
            train_loss: 训练平均损失。
            val_loss: 验证损失。
        """
        msg = f"Epoch {epoch} 完成 | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
        self.logger.info(msg)
        self.current_epoch = epoch

    def reset_timer(self) -> None:
        """重置计时器"""
        self.start_time = time.time()
        self.step_times = []

    def get_log_file(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)


def get_gpu_memory() -> float:
    """
    获取当前 GPU 显存占用（GB）。

    Returns:
        float: 显存占用（GB）。
    """
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0
