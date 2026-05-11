"""
Transformer 顶层模型模块

组装完整的 DeepSeek-V4 Transformer 模型，包含：
    - 词嵌入层（ParallelEmbedding）
    - 多个 Block（MLA + MoE + HC）
    - 最终 RMSNorm + 输出头
    - Hyper-Connections: 维护 hc_mult 个并行残差流

与标准 Transformer 不同，V4 使用 HC 结构：
    - 嵌入扩展为 hc_mult 个副本
    - 每层输入/输出: [batch, seq, hc_mult, dim]
    - Block 内部通过 hc_pre/hc_post 混合残差流
"""

import torch
import torch.distributed as dist
from torch import nn
from .config import ModelArgs
from .layers import Linear, ColumnParallelLinear, ParallelEmbedding, RMSNorm
from .block import Block
from .RuntimeConfig import RuntimeConfig


class Transformer(nn.Module):
    """
    DeepSeek-V4 Transformer 模型，带 Hyper-Connections。

    结构：
        Embedding → Expand to hc_mult copies → [Block × n_layers]
                  → Merge hc copies → RMSNorm → Head → logits

    Hyper-Connections 流程：
        1. 嵌入后扩展: [b,s,d] → [b,s,hc,d]
        2. 每层 Block: 输入/输出均为 [b,s,hc,d]
        3. 最终合并: [b,s,hc,d] → [b,s,d]

    属性:
        max_seq_len (int):          最大序列长度。
        hc_mult (int):              Hyper-Connections 倍数。
        embed (ParallelEmbedding):  分布式词嵌入层。
        layers (ModuleList[Block]): Transformer Block 列表。
        norm (RMSNorm):             最终层归一化。
        head (ColumnParallelLinear):输出投影（按词表列切分）。
        hc_head_fn, hc_head_scale, hc_head_base: 最终 HC 合并参数。
    """

    def __init__(self, args: ModelArgs):
        """
        初始化 Transformer 模型。

        Args:
            args: 模型参数，包含 hc_mult 等 HC 配置。
        """
        # ---- 创建运行时配置 ----
        self.runtime = RuntimeConfig.from_distributed()
        RuntimeConfig._default = self.runtime

        # ---- 设置数据类型 ----
        target_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        torch.set_default_dtype(target_dtype)
        Linear.dtype = target_dtype
        Linear.scale_fmt = args.scale_fmt

        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.hc_mult

        # 词嵌入层
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)

        # Block 堆叠
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))

        # 最终归一化
        self.norm = RMSNorm(args.dim, args.norm_eps)

        # 输出头
        self.head = ColumnParallelLinear(
            args.dim, args.vocab_size, dtype=torch.get_default_dtype()
        )

        # 最终 HC 合并参数（将 [b,s,hc,d] 合并为 [b,s,d]）
        hc_dim = args.hc_mult * args.dim
        with torch.no_grad():
            self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, hc_dim, dtype=torch.float32))
            self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        nn.init.normal_(self.hc_head_fn, std=0.02)
        nn.init.zeros_(self.hc_head_base)
        nn.init.ones_(self.hc_head_scale)

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        最终 HC 合并：将 hc_mult 个副本合并为 1 个。

        Args:
            x: [batch, seq, hc_mult, dim]

        Returns:
            y: [batch, seq, dim]
        """
        shape, dtype = x.size(), x.dtype  # [b, s, hc, d]
        x_flat = x.flatten(2).float()  # [b, s, hc*d]

        # RMSNorm
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + 1e-6)
        mixes = torch.matmul(x_flat, self.hc_head_fn.t()) * rsqrt  # [b, s, hc]

        # Sigmoid 归一化（简化版，不使用完整 Sinkhorn）
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + 1e-6
        pre = pre / pre.sum(dim=-1, keepdim=True)  # [b, s, hc]

        # 加权合并
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)  # [b, s, d]

        return y.to(dtype)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, return_all_logits: bool = False):
        """
        前向传播。

        Args:
            tokens: 输入 token ID [batch_size, seq_len]
            start_pos: 序列起始位置（自回归生成时使用）

        Returns:
            logits: [batch_size, vocab_size]
        """
        # 第1步：词嵌入
        h = self.embed(tokens)  # [b, s, d]

        # 第2步：扩展为 hc_mult 个副本
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)  # [b, s, hc, d]

        # 第3步：逐层传递（Block 内部完成 HC 混合）
        # 注意：Block 不再需要外部传入 freqs_cis 和 mask，MLA 内部处理
        for layer in self.layers:
            h = layer(h, start_pos, tokens)  # [b, s, hc, d]

        # 第4步：HC 合并为 1 个副本
        h = self.hc_head(h)  # [b, s, d]

        # 修改--适配训练模式
        h = self.norm(h)

        # 第5步：根据状态判定是否进行训练
        if not return_all_logits:
            h = h[:, -1]  # [b, dim]

        # 第6步：输出头投影到词表空间
        logits = self.head(h)  # [b, part_vocab_size]

        # 第7步：分布式聚合
        if self.runtime.world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.runtime.world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)  # [b, vocab_size]

        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    args = ModelArgs(hc_mult=4)
    x = torch.randint(0, args.vocab_size, (2, 128))

    model = Transformer(args)
    logits = model(x)

    print(f"输入形状: (2, 128)")
    print(f"输出形状: {logits.size()}")  # 应为 (2, vocab_size)
    print(f"输出 dtype: {logits.dtype}")
    print(f"HC 倍数: {model.hc_mult}")
    print(f"显卡数量: {model.runtime.world_size}")
    print(f"当前排名: {model.runtime.rank}")
    print("✅ Transformer 前向传播通过！")
