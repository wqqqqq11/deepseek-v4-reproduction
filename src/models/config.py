from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelArgs:
    """
    模型参数数据类，定义所有模型结构和超参数。

    默认使用 Tiny（方案A）参数配置，总参数量约 25M，
    适合单 GPU（12GB 显存）训练和快速迭代。
    可通过 ModelArgs.original() 获取完整 DeepSeek V4 配置。
    """

    # ── 运行环境 / 部署相关 ────────────────────────────────────────────
    max_batch_size: int = 4
    """最大批次大小，用于预分配 KV 缓存缓冲区。"""

    max_seq_len: int = 4096
    """模型能处理的最大序列长度（同时决定 RoPE 预计算范围）。"""

    dtype: Literal["bf16", "fp8"] = "bf16"
    """计算数据类型：bf16（bfloat16）或 fp8（float8，需自定义 kernel 支持）。"""

    scale_fmt: Optional[str] = None
    """分块量化的缩放因子格式，None 表示不使用量化；仅在 dtype='fp8' 时生效。"""

    # ── 词表 ──────────────────────────────────────────────────────────
    vocab_size: int = 32000
    """词表大小（嵌入矩阵尺寸：vocab_size × dim）。
    原始值为 102400，此处降至 32000 以减少显存占用，便于测试。"""

    # ── 核心模型维度 ──────────────────────────────────────────────────
    dim: int = 512
    """隐藏层维度（模型宽度 / 残差流维度），所有嵌入、注意力投影和 FFN 层均以此为基准。"""

    inter_dim: int = 1536
    """密集 MLP 前馈网络的中间层（隐藏）维度。
    通常约为 dim 的 3 倍；原始比例约为 5.34 倍（10944/2048）。"""

    moe_inter_dim: int = 192
    """每个 MoE 专家前馈网络的中间层维度。
    通常比 inter_dim 小，因为专家更专业化；此处约为 dim 的一半。"""

    n_layers: int = 8
    """Transformer 总层数（Block 个数）。"""

    n_dense_layers: int = 1
    """使用密集 MLP（而非 MoE）的前几层数量。
    第 0 到 n_dense_layers-1 层使用 MLP，其余层使用 MoE。"""

    n_heads: int = 4
    """MLA（多头潜在注意力）的注意力头数。
    每个头的维度为 dim // n_heads = 128。"""

    # ── 专家混合（MoE） ──────────────────────────────────────────────
    n_routed_experts: int = 16
    """每个 MoE 层中路由专家的总数。
    当 world_size > 1 时，专家会在各 GPU 间切分。"""

    n_shared_experts: int = 2
    """共享专家数量（所有 token 都会经过，不参与路由）。
    共享专家的输出会加到路由专家输出之上。"""

    n_activated_experts: int = 4
    """每个 token 激活的专家数（top-k 路由）。
    每个 token 将被路由到门控得分最高的 k 个专家。"""

    n_expert_groups: int = 4
    """分层路由中的专家分组数。
    当 > 1 时，token 先路由到某个组，再在组内选择专家。"""

    n_limited_groups: int = 2
    """分层路由中每个 token 可选择的组数上限。"""

    score_func: Literal["softmax", "sigmoid"] = "softmax"
    """MoE 门控的评分函数：'softmax'（归一化）或 'sigmoid'（非归一化）。
    sigmoid 模式会额外用所选权重之和重新归一化。"""

    route_scale: float = 1.0
    """路由权重的缩放因子，在专家选择后乘以原始得分，用于控制专家输出大小。"""

    # ── 多头潜在注意力（MLA） ────────────────────────────────────────
    q_lora_rank: int = 0
    """查询投影的低秩压缩维度。
    为 0 时，查询投影为直接线性层（无低秩分解）；
    大于 0 时，Q 先压缩到该维度再扩展到各头维度。"""

    kv_lora_rank: int = 128
    """键值投影的低秩压缩维度。
    KV 状态先压缩到此维度，可将 KV 缓存内存降低约 dim / kv_lora_rank 倍。"""

    qk_nope_head_dim: int = 64
    """查询和键中非位置编码部分的每头维度（内容部分）。"""

    qk_rope_head_dim: int = 32
    """查询和键中位置编码（RoPE）部分的每头维度。
    总 Q/K 头维度 = qk_nope_head_dim + qk_rope_head_dim = 96。"""

    v_head_dim: int = 64
    """值投影的每头维度。"""

    # ── YaRN 位置编码（长序列扩展） ──────────────────────────────────
    original_seq_len: int = 4096
    """预训练的原始序列长度，作为 YaRN 缩放的基准。
    序列长度 ≤ 此值时使用标准 RoPE。"""

    rope_theta: float = 10000.0
    """旋转位置编码（RoPE）的基础频率。
    值越大，频率衰减越慢，有利于改善长程衰减稳定性。"""

    rope_factor: float = 1.0
    """YaRN 扩展因子，用于在原序列长度基础上扩展。
    当 max_seq_len > original_seq_len 时，频率会除以此因子并配合平滑插值。
    设为 1.0 表示不需要长序列扩展。"""

    beta_fast: int = 32
    """YaRN 维度混合的快速 beta 修正参数。
    控制插值斜坡函数的下界（以旋转圈数为单位）。"""

    beta_slow: int = 1
    """YaRN 维度混合的慢速 beta 修正参数。
    控制插值斜坡函数的上界（以旋转圈数为单位）。"""

    mscale: float = 1.0
    """长序列扩展时注意力 softmax 缩放系数的乘数。
    当 max_seq_len > original_seq_len 时：
    softmax_scale *= 0.1 * mscale * ln(rope_factor) + 1.0。"""

    # ── 便捷预设 ──────────────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "ModelArgs":
        """返回 Tiny（方案A）配置：约 25M 参数，适合单 GPU 训练。"""
        return cls()

    @classmethod
    def original(cls) -> "ModelArgs":
        """返回原始 DeepSeek V4 配置：约 671B 总参数，约 37B 激活参数。"""
        return cls(
            max_batch_size=8,
            max_seq_len=16384,
            vocab_size=102400,
            dim=2048,
            inter_dim=10944,
            moe_inter_dim=1408,
            n_layers=27,
            n_heads=16,
            n_routed_experts=64,
            n_shared_experts=2,
            n_activated_experts=6,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rope_factor=40,
        )