"""
基础层模块（可拔插设计）

本模块提供 Transformer 模型的基础构建块，所有组件均不直接依赖全局变量，
而是通过 RuntimeConfig 注入运行时配置（world_size、rank、block_size 等），
因此可被任何其他模型项目直接复用。

包含组件：
    - linear()               统一线性变换函数
    - Linear                自定义线性层（支持量化权重）
    - ColumnParallelLinear  列并行线性层（输出维度切分）
    - RowParallelLinear     行并行线性层（输入维度切分，需 all_reduce）
    - ParallelEmbedding     分布式词嵌入层
    - RMSNorm               Root Mean Square 层归一化
"""

from typing import Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from .kernel import act_quant, weight_dequant, fp8_gemm
from .RuntimeConfig import RuntimeConfig


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale_fmt: Optional[str] = None,
    block_size: Optional[int] = None,
    gemm_impl: Optional[str] = None,
) -> torch.Tensor:
    """
    统一线性变换入口，根据权重量化状态和 GEMM 实现方式自动选择计算路径。

    三种计算路径：
        1. 标准 BF16 线性：权重未量化（element_size > 1），直接调用 F.linear。
        2. 反量化 + BF16 线性：权重已量化但 gemm_impl == "bf16"，先反量化再计算。
        3. FP8 GEMM：对输入做分块量化后，使用 fp8_gemm 进行矩阵乘法。

    可拔插设计：
        - block_size 和 gemm_impl 为可选参数，未传入时自动从 RuntimeConfig 读取。
        - 调用时可临时覆盖全局配置，单次调用不影响其他代码路径。

    Args:
        x:          输入张量，形状 (..., in_features)。
        weight:     权重矩阵，形状 (out_features, in_features)。
                    若已量化（element_size() == 1），则需附带 .scale 属性。
        bias:       可选偏置，形状 (out_features,)。
        scale_fmt:  分块量化的缩放因子格式（fp8 模式使用）。
        block_size: 分块量化的块大小，默认从 RuntimeConfig 获取。
        gemm_impl:  GEMM 实现模式 ("bf16" 或 "fp8")，默认从 RuntimeConfig 获取。

    Returns:
        torch.Tensor: 线性变换结果，形状 (..., out_features)。
    """
    if block_size is None:
        block_size = RuntimeConfig.default().block_size
    if gemm_impl is None:
        gemm_impl = RuntimeConfig.default().gemm_impl

    if weight.element_size() > 1:
        # 路径 1：标准 BF16 线性
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # 路径 2：反量化后 BF16 线性
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # 路径 3：FP8 GEMM
        x, scale = act_quant(x, block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    可拔插的自定义线性层，支持量化权重和可选偏置。

    当权重的 element_size 为 1（即 fp8 量化格式）时，自动创建 scale 参数
    用于后续反量化。运行时不依赖任何全局变量，所有动态参数通过 RuntimeConfig 注入。

    属性:
        dtype (torch.dtype):   权重默认数据类型，类变量，默认为 torch.bfloat16。
        scale_fmt (str|None):  分块量化缩放格式，类变量，fp8 模式下使用。
        in_features (int):     输入特征维度。
        out_features (int):    输出特征维度（并行层中为本地持有维度）。
        weight (Parameter):    权重矩阵，形状 (out_features, in_features)。
        scale (Parameter|None):量化缩放因子，非量化层为 None。
        bias (Parameter|None): 偏置向量，无偏置时为 None。

    Args:
        in_features:  输入特征维度。
        out_features: 输出特征维度。
        bias:         是否使用偏置，默认 False。
        dtype:        权重数据类型，默认使用类变量 Linear.dtype。
    """
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype or Linear.dtype)
        )

        # 量化权重处理：element_size() == 1 表示已量化（如 fp8），需创建 scale 参数
        if self.weight.element_size() == 1:
            block_size = RuntimeConfig.default().block_size
            # 按块大小计算 scale 矩阵的维度（向上取整）
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight.element_size() > 1:
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.weight)
        if self.scale is not None:
            nn.init.ones_(self.scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，委托给 linear() 函数处理。"""
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    列并行线性层，按输出维度（out_features）在多个 GPU 间均匀切分。

    每个 GPU 只持有 1/world_size 的输出维度权重，前向计算后不进行 all_reduce，
    因此输出为局部结果。适合接在 RowParallelLinear 之后，或不需要完整输出的场景。

    继承自 Linear，额外属性：
        part_out_features (int): 当前 GPU 持有的输出特征数 = out_features / world_size。

    Args:
        in_features:  输入特征维度（全局值，不切分）。
        out_features: 输出特征维度（全局值，必须能被 world_size 整除）。
        bias:         是否使用偏置。
        dtype:        权重数据类型。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        world_size = RuntimeConfig.default().world_size
        assert out_features % world_size == 0, (
            f"Output features ({out_features}) must be divisible by world_size ({world_size})"
        )
        self.part_out_features = out_features // world_size
        # 父类用切分后的 part_out_features 创建权重
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回局部结果（不聚合）。"""
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    行并行线性层，按输入维度（in_features）在多个 GPU 间均匀切分。

    每个 GPU 只持有 1/world_size 的输入维度权重，前向计算后必须通过 all_reduce
    聚合各 GPU 的部分和，才能得到完整的输出。适合接在 ColumnParallelLinear 之后。

    继承自 Linear，额外属性：
        part_in_features (int): 当前 GPU 持有的输入特征数 = in_features / world_size。

    Args:
        in_features:  输入特征维度（全局值，必须能被 world_size 整除）。
        out_features: 输出特征维度（全局值，不切分）。
        bias:         是否使用偏置。
        dtype:        权重数据类型。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        world_size = RuntimeConfig.default().world_size
        assert in_features % world_size == 0, (
            f"Input features ({in_features}) must be divisible by world_size ({world_size})"
        )
        self.part_in_features = in_features // world_size
        # 父类用切分后的 part_in_features 创建权重
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，先计算局部结果，再通过 all_reduce 聚合。

        偏置在所有 GPU 上都相同，因此在 all_reduce 之后才加上，
        避免被重复累加。
        """
        y = linear(x, self.weight)
        if RuntimeConfig.default().world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class ParallelEmbedding(nn.Module):
    """
    分布式词嵌入层，按 world_size 将词表维度均匀切分到各 GPU。

    每个 GPU 只存储 vocab_size // world_size 个 token 的嵌入向量，
    前向传播时对超出本 GPU 范围的 token 做 masked embedding，
    最后通过 all_reduce 聚合所有 GPU 的嵌入结果。

    属性:
        vocab_size (int):       全局词表大小（必须能被 world_size 整除）。
        dim (int):              嵌入向量的维度。
        part_vocab_size (int):  当前 GPU 持有的词表大小 = vocab_size // world_size。
        vocab_start_idx (int):  当前 GPU 对应的词表起始索引。
        vocab_end_idx (int):    当前 GPU 对应的词表结束索引（不包含）。
        weight (Parameter):     嵌入权重矩阵，形状 (part_vocab_size, dim)。

    Args:
        vocab_size: 全局词表大小。
        dim:        嵌入向量的维度。
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        world_size = RuntimeConfig.default().world_size
        rank = RuntimeConfig.default().rank
        assert vocab_size % world_size == 0, (
            f"Vocabulary size ({vocab_size}) must be divisible by world size ({world_size})"
        )
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        对于不在本 GPU 词表范围内的 token，先将其索引偏移到有效范围
        并 mask 掉其输出，最后通过 all_reduce 从其他 GPU 获取正确嵌入。

        Args:
            x: 输入 token 索引，形状 (batch_size, seq_len)，dtype 为整型。

        Returns:
            torch.Tensor: 嵌入向量，形状 (batch_size, seq_len, dim)。
        """
        world_size = RuntimeConfig.default().world_size
        if world_size > 1:
            # 标记不在本 GPU 范围内的 token
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 将索引偏移到本 GPU 的局部范围
            x = x - self.vocab_start_idx
            x[mask] = 0  # 越界 token 临时设为 0，后续会被 mask
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0  # 清除越界 token 的嵌入
            dist.all_reduce(y)  # 从各 GPU 聚合正确嵌入
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization（RMS 层归一化）。

    与 LayerNorm 相比，RMSNorm 去掉了均值中心化，只做缩放归一化，
    计算更高效，在 Transformer 模型中广泛使用。

    属性:
        dim (int):         归一化的维度。
        eps (float):       数值稳定性参数，防止除零。
        weight (Parameter): 可学习的缩放参数，形状 (dim,)。

    Args:
        dim:  归一化的维度（通常为隐藏层维度）。
        eps:  数值稳定性参数，默认 1e-6。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状 (..., dim)。

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同。
        """
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).type_as(x) * self.weight
