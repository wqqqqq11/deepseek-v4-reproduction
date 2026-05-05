"""
运行时环境配置（实例化设计）

每个 Transformer 实例应持有独立的 RuntimeConfig，避免多模型在
同一进程中因共享可变类变量而相互干扰。

迁移指南：
    旧写法:  RuntimeConfig.world_size / RuntimeConfig.block_size
    新写法:  RuntimeConfig.default().world_size        （快速兼容）
             self.runtime.world_size                    （推荐：构造注入）

    旧写法:  RuntimeConfig.set_distributed(8, 0)
    新写法:  RuntimeConfig.default().set_distributed(8, 0)   （兼容）
             rt = RuntimeConfig(world_size=8, rank=0)          （推荐）
"""

from typing import Literal, Optional


class RuntimeConfig:
    """
    运行时配置，控制分布式拓扑、量化参数和注意力实现模式。

    每个字段都是实例属性，不再使用可变类变量。通过 default() 获取
    全局单例以兼容旧代码；新代码应显式创建实例并注入到模块中。
    """

    # 模块级默认单例（惰性创建）
    _default: Optional["RuntimeConfig"] = None

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        block_size: int = 128,
        gemm_impl: Literal["bf16", "fp8"] = "bf16",
        attn_impl: Literal["naive", "absorb"] = "absorb",
    ):
        """
        Args:
            world_size: 分布式训练的 GPU 总数，单机为 1。
            rank:       当前 GPU 在全局的排名（0 ~ world_size-1）。
            block_size: 分块量化的块大小（fp8 路径使用）。
            gemm_impl:  GEMM 实现模式：
                         - "bf16"  标准 bfloat16 线性
                         - "fp8"   模拟 float8 量化推理
            attn_impl:  注意力实现模式：
                         - "naive"  标准多头注意力，缓存完整 K/V
                         - "absorb" 低秩吸收模式，缓存潜变量以节省显存
        """
        self.world_size = world_size
        self.rank = rank
        self.block_size = block_size
        self.gemm_impl = gemm_impl
        self.attn_impl = attn_impl

    # ── 默认单例（向后兼容桥接） ────────────────────────────

    @classmethod
    def default(cls) -> "RuntimeConfig":
        """
        返回全局默认 RuntimeConfig 实例。

        旧代码中直接访问 RuntimeConfig.world_size 等类变量的路径
        应改为 RuntimeConfig.default().world_size。
        """
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def reset_default(cls) -> None:
        """
        重置默认实例并释放引用。

        主要用于测试环境，确保各测试用例之间配置隔离。
        """
        cls._default = None

    # ── 快捷工厂 ────────────────────────────────────────────

    @classmethod
    def from_distributed(cls) -> "RuntimeConfig":
        """
        从当前分布式环境自动创建配置。

        读取 torch.distributed 的 world_size 和 rank；
        若分布式未初始化则回退为单机配置。
        """
        import torch.distributed as dist

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        return cls(world_size=world_size, rank=rank)

    # ── 便捷设置方法 ───────────────────────────────────────

    def set_distributed(self, world_size: int, rank: int) -> None:
        """设置分布式参数（实例方法，不影响其他实例）。"""
        self.world_size = world_size
        self.rank = rank

    def set_gemm_impl(self, impl: Literal["bf16", "fp8"]) -> None:
        """设置 GEMM 实现模式。"""
        self.gemm_impl = impl

    def set_attn_impl(self, impl: Literal["naive", "absorb"]) -> None:
        """设置注意力实现模式。"""
        self.attn_impl = impl

    # ── 向后兼容的类方法（Deprecated，操作 default 单例） ──

    @classmethod
    def _set_distributed(cls, world_size: int, rank: int) -> None:
        """
        [Deprecated] 类方法版本，操作 default 单例。

        请改用 RuntimeConfig(world_size=..., rank=...)
        或 RuntimeConfig.default().set_distributed(...)。
        """
        cls.default().set_distributed(world_size, rank)

    @classmethod
    def _set_gemm_impl(cls, impl: Literal["bf16", "fp8"]) -> None:
        """[Deprecated] 请改用 RuntimeConfig.default().set_gemm_impl(impl)。"""
        cls.default().set_gemm_impl(impl)

    @classmethod
    def _set_attn_impl(cls, impl: Literal["naive", "absorb"]) -> None:
        """[Deprecated] 请改用 RuntimeConfig.default().set_attn_impl(impl)。"""
        cls.default().set_attn_impl(impl)