"""
困惑度计算模块

提供统一的PPL计算接口，供训练和测试复用。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_ppl(model, dataloader: DataLoader, device: torch.device, vocab_size: int) -> float:
    """
    计算困惑度。

    Args:
        model: 模型实例。
        dataloader: 数据加载器。
        device: 计算设备。
        vocab_size: 词表大小。

    Returns:
        float: 困惑度值。
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return ppl
