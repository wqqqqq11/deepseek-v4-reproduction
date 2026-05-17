"""
Dense SwiGLU FFN 模块

阶段1使用的密集前馈网络，采用 SwiGLU 激活函数：
    SwiGLU(x) = (xW_1 * SiLU(xW_2))W_3

相比标准 GELU/ReLU FFN，SwiGLU 在语言模型中表现更优。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数。
    
    公式：SwiGLU(x) = xW_1 * SiLU(xW_2)
    
    Args:
        dim: 输入维度。
        hidden_dim: 隐藏层维度（通常为 2/3 * 4 * dim）。
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状 (..., dim)。
            
        Returns:
            torch.Tensor: 形状 (..., hidden_dim)。
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        return x1 * F.silu(x2)


class DenseMLP(nn.Module):
    """
    密集 SwiGLU MLP 层。
    
    结构：
        x → SwiGLU → Linear → output
    
    Args:
        dim: 输入/输出维度。
        hidden_dim: 隐藏层维度（默认 None，自动计算为 2/3 * 4 * dim）。
        dropout: Dropout 率（默认 0.0）。
    
    Example:
        >>> mlp = DenseMLP(dim=256, hidden_dim=688)
        >>> out = mlp(x)  # x: [batch, seq, 256], out: [batch, seq, 256]
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if hidden_dim is None:
            # 标准 SwiGLU：约 2/3 * 4 * dim（等价于 8/3 * dim）
            hidden_dim = int(8 / 3 * dim)
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # SwiGLU 门控 + 投影
        self.swiglu = SwiGLU(dim, hidden_dim)
        
        # 输出投影
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状 (..., dim)。
            
        Returns:
            torch.Tensor: 输出张量，形状 (..., dim)。
        """
        # SwiGLU 激活
        h = self.swiglu(x)
        
        # 输出投影
        out = self.w3(h)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
    
    def extra_repr(self) -> str:
        """额外信息字符串"""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


# 别名：保持与项目其他部分兼容
MLP = DenseMLP
