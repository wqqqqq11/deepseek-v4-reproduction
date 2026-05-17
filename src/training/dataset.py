"""
Token 数据集加载器

提供高效的 .bin 文件内存映射加载，支持：
    - 内存映射避免内存爆炸
    - 每个 epoch 无放回随机打乱
    - 快速 (x, y) 样本生成
"""

import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple, Iterator, Dict, Any
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class TokenDataset(IterableDataset):
    """
    内存映射方式加载 .bin 格式 token 数据。
    
    .bin 文件格式：
        - uint16 或 uint32 编码的 token stream
        - 每个 token 已包含 eos_token_id
        - 通过 memory map 实现随机访问
    
    样本生成：
        - 从 token stream 中随机采样 context_size + 1 长度的序列
        - x = tokens[:-1], y = tokens[1:]（自回归预测）
        - 确保每个样本在单个 epoch 内只出现一次
    
    Args:
        bin_path: .bin 文件路径。
        context_size: 序列长度（默认 1024）。
        num_samples: 样本总数（从 metadata.json 读取或手动指定）。
        shuffle_seed: 随机打乱种子（每个 epoch 不同）。
        dtype: token 数据类型（默认 np.uint16）。
    
    Example:
        >>> dataset = TokenDataset("train.bin", context_size=1024)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for x, y in loader:
        ...     # x: [batch, 1024], y: [batch, 1024]
        ...     pass
    """
    
    def __init__(
        self,
        bin_path: str,
        context_size: int = 1024,
        num_samples: Optional[int] = None,
        shuffle_seed: int = 42,
        dtype: np.dtype = np.uint16,
    ):
        self.bin_path = Path(bin_path)
        self.context_size = context_size
        self.base_seed = shuffle_seed
        self.dtype = dtype
        
        # 验证文件存在
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Bin file not found: {bin_path}")
        
        # 读取 metadata
        self.metadata = self._load_metadata()
        
        # 确定样本数
        if num_samples is not None:
            self.num_samples = num_samples
        elif self.metadata and "num_samples" in self.metadata:
            self.num_samples = self.metadata["num_samples"]
        else:
            # 从文件大小计算
            file_size = self.bin_path.stat().st_size
            bytes_per_token = 2 if dtype == np.uint16 else 4
            total_tokens = file_size // bytes_per_token
            self.num_samples = max(0, total_tokens - context_size)
        
        # 内存映射
        self.tokens = np.memmap(bin_path, dtype=dtype, mode="r")
        self.epoch = 0
    
    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """从 metadata.json 加载元数据"""
        meta_path = self.bin_path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def _get_sample_indices(self, epoch: int) -> np.ndarray:
        """
        生成当前 epoch 的样本起始索引。
        
        使用确定的随机种子，确保可复现。
        
        Args:
            epoch: 当前 epoch 数。
            
        Returns:
            np.ndarray: 打乱的样本起始索引数组。
        """
        # 每个 epoch 使用不同的种子
        seed = self.base_seed + epoch
        rng = np.random.RandomState(seed)
        
        # 生成所有可能的起始位置
        max_start = len(self.tokens) - self.context_size - 1
        indices = np.arange(max(0, min(max_start, self.num_samples)))
        
        # 打乱
        rng.shuffle(indices)
        
        return indices[:self.num_samples]
    
    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本。
        
        Args:
            idx: token stream 中的起始位置。
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (x, y) 张量对。
        """
        start = int(idx)
        end = start + self.context_size + 1
        
        # 安全边界检查
        if end > len(self.tokens):
            start = max(0, len(self.tokens) - self.context_size - 1)
            end = len(self.tokens)
        
        chunk = self.tokens[start:end]
        
        # 转换为 torch 张量
        tokens = torch.from_numpy(chunk.astype(np.int64))
        
        x = tokens[:-1]
        y = tokens[1:]
        
        return x, y
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        迭代生成样本。
        
        支持多 worker 数据加载，每个 worker 处理部分样本。
        """
        worker_info = get_worker_info()
        
        if worker_info is None:
            # 单进程模式
            indices = self._get_sample_indices(self.epoch)
            for idx in indices:
                yield self._get_sample(idx)
        else:
            # 多 worker 模式：分割索引
            indices = self._get_sample_indices(self.epoch)
            
            # 当前 worker 负责的索引范围
            per_worker = len(indices) // worker_info.num_workers
            worker_start = worker_info.id * per_worker
            worker_end = worker_start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(indices)
            
            worker_indices = indices[worker_start:worker_end]
            
            for idx in worker_indices:
                yield self._get_sample(idx)
    
    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch，影响随机打乱种子。
        
        应在每个 epoch 开始时调用。
        
        Args:
            epoch: 当前 epoch 数。
        """
        self.epoch = epoch
    
    def __len__(self) -> int:
        """返回数据集样本数"""
        return self.num_samples


def create_dataloader(
    bin_path: str,
    context_size: int = 1024,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_seed: int = 42,
    epoch: int = 0,
) -> DataLoader:
    """
    便捷创建 DataLoader。
    
    Args:
        bin_path: .bin 文件路径。
        context_size: 序列长度。
        batch_size: 批次大小。
        num_workers: 数据加载 worker 数。
        shuffle_seed: 随机打乱种子。
        epoch: 初始 epoch。
        
    Returns:
        DataLoader: 配置好的数据加载器。
    """
    dataset = TokenDataset(
        bin_path=bin_path,
        context_size=context_size,
        shuffle_seed=shuffle_seed,
    )
    dataset.set_epoch(epoch)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 数据集内部处理 shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def load_metadata_from_bin(bin_path: str) -> Dict[str, Any]:
    """
    从 .bin 文件对应的 metadata.json 加载元数据。
    
    Args:
        bin_path: .bin 文件路径。
        
    Returns:
        Dict[str, Any]: 元数据字典。
    """
    meta_path = Path(bin_path).parent / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 默认元数据
    return {
        "vocab_size": 6400,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
