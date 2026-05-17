"""
FP8 / FP4 混合精度模拟模块

在 BF16 计算基础上模拟 FP8/FP4 量化行为：
    - 前向传播：模拟 per-tensor FP8 量化
    - 权重：维护 FP8 格式和缩放因子
    - 实际计算：使用 BF16，但记录量化统计信息

用于在消费级 GPU 上验证 FP8 训练流程，后续可替换为真实 FP8 kernel。
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# FP8 E4M3 格式常量
FP8_E4M3_MAX = 448.0  # 最大可表示值
FP8_E4M3_MIN = -448.0


def compute_scale(x: torch.Tensor) -> torch.Tensor:
    """
    计算 per-tensor 量化缩放因子。
    
    scale = max(abs(x)) / FP8_MAX
    
    Args:
        x: 输入张量。
        
    Returns:
        torch.Tensor: 标量缩放因子。
    """
    abs_max = x.abs().max()
    if abs_max < 1e-12:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    scale = abs_max / FP8_E4M3_MAX
    return scale.detach()


def fake_quantize_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    模拟 FP8 量化：量化 → 反量化。
    
    Args:
        x: 输入张量（BF16/FP32）。
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (伪量化后的张量, 缩放因子)。
    """
    scale = compute_scale(x)
    
    # 模拟量化：x_quant = x / scale，裁剪到 FP8 范围
    x_scaled = x / scale
    x_clipped = torch.clamp(x_scaled, FP8_E4M3_MIN, FP8_E4M3_MAX)
    
    # 模拟反量化：x_dequant = x_quant * scale
    x_dequant = x_clipped * scale
    
    return x_dequant.detach(), scale.detach()


class FP8Linear(nn.Linear):
    """
    模拟 FP8 权重的线性层。
    
    实际计算使用 BF16，但：
        1. 对输入模拟 FP8 量化
        2. 对权重维护伪 FP8 格式和缩放因子
        3. 返回缩放后的结果
    
    Args:
        in_features: 输入维度。
        out_features: 输出维度。
        bias: 是否使用偏置。
    
    Example:
        >>> layer = FP8Linear(256, 512)
        >>> out = layer(x)  # x 和 weight 都经过 FP8 模拟
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        
        # 注册权重缩放因子（per-tensor）
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("input_scale", torch.tensor(1.0))
        
        # 是否启用 FP8 模拟
        self.fp8_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，应用 FP8 模拟量化。
        
        Args:
            x: 输入张量。
            
        Returns:
            torch.Tensor: 线性变换结果。
        """
        if not self.fp8_enabled:
            return F.linear(x, self.weight, self.bias)
        
        # 模拟输入 FP8 量化
        x_fp8, x_scale = fake_quantize_to_fp8(x)
        self.input_scale = x_scale.detach()
        
        # 模拟权重 FP8 量化（只在更新时重新计算）
        if self.training:
            w_fp8, w_scale = fake_quantize_to_fp8(self.weight)
            self.weight_scale = w_scale.detach()
        else:
            w_fp8 = self.weight
            w_scale = self.weight_scale
        
        # BF16 计算（但数值已经过 FP8 模拟）
        out = F.linear(x_fp8, w_fp8, self.bias)
        
        # 合并缩放因子（恢复数值范围）
        out = out * x_scale * w_scale
        
        return out
    
    def extra_repr(self) -> str:
        """额外信息字符串"""
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"bias={self.bias is not None}, fp8_enabled={self.fp8_enabled}"


def enable_fp8(model: nn.Module, enabled: bool = True) -> None:
    """
    全局启用/禁用 FP8 模拟。
    
    Args:
        model: 模型。
        enabled: 是否启用。
    """
    for module in model.modules():
        if isinstance(module, FP8Linear):
            module.fp8_enabled = enabled


def get_fp8_stats(model: nn.Module) -> dict:
    """
    获取 FP8 量化统计信息。
    
    Args:
        model: 模型。
        
    Returns:
        dict: 包含各层缩放因子的字典。
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            stats[name] = {
                "weight_scale": module.weight_scale.item(),
                "input_scale": module.input_scale.item(),
            }
    return stats


# 别名：为兼容性提供
Linear = FP8Linear
