"""
Transformer 顶层模型模块

组装完整的 DeepSeek-V4 Transformer 模型，包含：
    - 词嵌入层（ParallelEmbedding）
    - 多个 Block（注意力 + MoE）
    - 最终 RMSNorm + 输出头
    - 旋转位置编码预计算（freqs_cis）
"""

import torch
import torch.distributed as dist
from torch import nn
from .config import ModelArgs
from .layers import Linear, ColumnParallelLinear, ParallelEmbedding, RMSNorm
from .block import Block
from .rotary_embedding import precompute_freqs_cis
from .RuntimeConfig import RuntimeConfig


class Transformer(nn.Module):
    """
    DeepSeek-V4 Transformer 模型。

    结构：
        Embedding → [Block × n_layers] → RMSNorm → Linear Head → logits

    支持分布式推理：Embedding 和 Head 按词表切分，Block 内部的 MLP/MoE/Attention
    按各自并行策略切分。分布式环境由 RuntimeConfig 注入，不使用全局变量。

    属性:
        max_seq_len (int):          最大序列长度。
        embed (ParallelEmbedding):  分布式词嵌入层。
        layers (ModuleList[Block]): Transformer Block 列表。
        norm (RMSNorm):             最终层归一化。
        head (ColumnParallelLinear):输出投影（按词表列切分），映射回词表空间。
        freqs_cis (Tensor):         预计算的 RoPE 复指数位置编码（缓冲区）。
    """

    def __init__(self, args: ModelArgs):
        """
        初始化 Transformer 模型。

        完成以下关键动作：
        1. 从分布式环境读取 world_size / rank，写入 RuntimeConfig。
        2. 根据 dtype 参数设置 Linear 的权重类型和量化格式。
        3. 构建完整的模型图。

        Args:
            args: 模型参数。
        """
        # ---- 创建运行时配置实例（替代原全局类变量写入） ----
        self.runtime = RuntimeConfig.from_distributed()

        # ---- 同步到默认单例，确保 layers/moe/attention 等子模块能用 default() 读到正确配置 ----
        RuntimeConfig._default = self.runtime

        # ---- 根据 dtype 配置 Linear 的类属性 ----
        # 1. 根据 args.dtype 确定目标数据类型
        target_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        # 2. 设置 PyTorch 全局默认 dtype，确保 Embedding 等也在同精度下创建
        torch.set_default_dtype(target_dtype)
        Linear.dtype =  target_dtype
        Linear.scale_fmt = args.scale_fmt

        super().__init__()
        self.max_seq_len = args.max_seq_len

        # 词嵌入层（分布式切分词表）
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)

        # Block 堆叠
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))

        # 最终归一化
        self.norm = RMSNorm(args.dim)

        # 输出头（列并行：按词表切分，推理时 all_gather 聚合）
        self.head = ColumnParallelLinear(
            args.dim, args.vocab_size, dtype=torch.get_default_dtype()
        )

        # 预计算 RoPE 位置编码，注册为持久化缓冲区
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(args),
            persistent=False,
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        前向传播。

        Args:
            tokens:     输入 token ID，形状 (batch_size, seq_len)。
            start_pos:  序列起始位置（自回归生成时使用），默认 0。

        Returns:
            logits: 形状 (batch_size, vocab_size) 的输出 logits。
        """
        seqlen = tokens.size(1)

        # 第1步：词嵌入
        h = self.embed(tokens)                                      # (bsz, seqlen, dim)

        # 第2步：取当前位置对应的 RoPE 频率
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 第3步：构建因果 mask（仅在 seqlen > 1 时需要，即训练/预填充阶段）
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            ).triu_(1)

        # 第4步：逐层传递（Block 内部完成 Pre-Norm + ResNet + 并行通信）
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # 第5步：最终归一化（只取最后一个 token 用于预测下一个）
        h = self.norm(h)[:, -1]                                     # (bsz, dim)

        # 第6步：输出头投影到词表空间
        logits = self.head(h)                                       # (bsz, part_vocab_size)

        # 第7步：分布式聚合各 GPU 的 logits（词表列并行）
        if self.runtime.world_size > 1:
            all_logits = [
                torch.empty_like(logits)
                for _ in range(self.runtime.world_size)
            ]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)                  # (bsz, vocab_size)

        return logits
    


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    logits = model(x)
    print(f"输入形状: (2, 128)")
    print(f"输出形状: {logits.size()}")          # 应为 (2, vocab_size)
    print(f"输出 dtype: {logits.dtype}")
    print(f"显卡大小: {model.runtime.world_size}")
    print(f"排名: {model.runtime.rank}")
    print("✅ Transformer 前向传播通过！")
