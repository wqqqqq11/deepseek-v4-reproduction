"""
模拟 fp8 量化 kernel 函数。

原始 DeepSeek V4 代码依赖外部 kernel 模块（act_quant, weight_dequant, fp8_gemm）
进行 float8 量化推理加速。本模块提供纯 PyTorch fallback 实现，确保在无 fp8 硬件
支持的环境下也能正常运行。

当 RuntimeConfig.gemm_impl == "fp8" 时，这些函数会被调用：
  - act_quant: 对激活值进行分块量化（模拟）
  - weight_dequant: 对量化权重进行反量化（模拟）
  - fp8_gemm: 使用 fp8 进行矩阵乘法（模拟，实际退化为 bf16 计算）
"""

import torch
import torch.nn.functional as F


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对激活值进行分块量化（模拟实现）。

    原始 kernel 会将输入按 block_size 分块，每块计算 scale 后量化为 fp8。
    这里直接返回原始张量和一个 dummy scale，实际计算退化为 bf16。

    Args:
        x: 输入张量
        block_size: 分块大小，默认 128
        scale_fmt: 缩放因子格式，None 表示不使用量化

    Returns:
        (量化后的 x, 每块的 scale)
        模拟实现中 x 不变，scale 为全 1 张量
    """
    # 模拟：不实际量化，返回原张量
    # scale 的形状应与分块后的 x 相匹配
    *batch_dims, rows, cols = x.shape
    # 计算分块后的行数和列数
    num_row_blocks = (rows + block_size - 1) // block_size
    num_col_blocks = (cols + block_size - 1) // block_size
    scale = torch.ones(
        *batch_dims, num_row_blocks, num_col_blocks,
        dtype=torch.float32,
        device=x.device,
    )
    return x, scale


def weight_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    对量化权重进行反量化（模拟实现）。

    原始 kernel 会使用 scale 将 fp8 权重反量化为 bf16。
    这里直接返回原始权重（假设权重未被实际量化）。

    Args:
        weight: 量化后的权重张量
        scale: 量化时使用的缩放因子
        block_size: 分块大小，默认 128

    Returns:
        反量化后的权重（模拟实现中直接返回原权重）
    """
    # 模拟：假设权重量化前后相同，直接返回
    return weight


def fp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    fp8 通用矩阵乘法（模拟实现）。

    原始 kernel 使用 fp8 精度进行矩阵乘法加速。
    这里退化为标准的 bf16/float32 线性变换。

    Args:
        x: 输入（可能是分块量化后的）
        x_scale: 输入的缩放因子（未使用）
        weight: 权重矩阵
        weight_scale: 权重的缩放因子（未使用）

    Returns:
        矩阵乘法结果
    """
    # 退化为标准线性变换
    return F.linear(x, weight)