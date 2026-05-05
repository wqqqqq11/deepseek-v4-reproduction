"""
多头潜在注意力（MLA）模块

实现了 DeepSeek-V4 中的 Multi-Head Latent Attention 机制。
该模块使用低秩联合压缩（LoRA）来减少 KV 缓存的内存占用，
并通过两种注意力实现模式（naive 与 absorb）在计算效率和内存之间取得平衡。

核心思想：
1. 对 Key 和 Value 进行低秩联合压缩，缓存压缩后的潜在表示而非展开的完整 K/V。
2. 将旋转位置编码（RoPE）与 Key 的非位置部分解耦，避免 RoPE 破坏低秩压缩。
3. 在 "absorb" 模式下，将 Value 投影矩阵的前半部分（nope 部分）"吸收" 进 Query，
   从而在推理时直接使用低秩缓存计算注意力，大幅减少 KV 缓存内存。
"""

import math
from typing import Optional
import torch
from torch import nn
from .config import ModelArgs
from .layers import Linear, ColumnParallelLinear, RowParallelLinear, RMSNorm
from .kernel import weight_dequant
from .rotary_embedding import apply_rotary_emb
from .RuntimeConfig import RuntimeConfig


class MLA(nn.Module):
    """
    多头潜在注意力（MLA）层。

    通过低秩投影压缩 KV 表示，降低推理时的 KV 缓存开销。
    支持两种注意力实现模式：
        - "naive"： 标准多头注意力，缓存展开后的完整 K 和 V。
        - "absorb"： 将 RoPE 后的 Key 投影吸收到 Query 和 Value 计算中，
                     缓存低秩 KV 潜变量和位置编码（pe_cache），
                     极大降低缓存内存占用。

    属性:
        dim (int):                 输入特征维度。
        n_heads (int):             注意力头总数（全局）。
        n_local_heads (int):       当前 GPU 持有的本地头数 = n_heads / world_size。
        q_lora_rank (int):         Query 低秩投影的秩，0 表示不使用 LoRA（直接投影）。
        kv_lora_rank (int):        Key-Value 联合低秩压缩的秩。
        qk_nope_head_dim (int):    Query/Key 非位置部分的每头维度。
        qk_rope_head_dim (int):    Query/Key RoPE 部分的每头维度。
        qk_head_dim (int):         Query/Key 总维度 = qk_nope_head_dim + qk_rope_head_dim。
        v_head_dim (int):          Value 的每头维度。
        softmax_scale (float):     注意力 softmax 缩放因子。当序列长度超原始长度时，
                                   自动乘以 YaRN 的 mscale^2 因子。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        # ---- 基本维度保存 ----
        self.dim = args.dim
        self.n_heads = args.n_heads
        # 本地头数 = 总头数 / 并行 GPU 数（列并行切分头维度）
        self.n_local_heads = args.n_heads // RuntimeConfig.default().world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # ---- Query 投影（支持低秩 LoRA） ----
        if self.q_lora_rank == 0:
            # 不使用 LoRA：直接做一次大的列并行线性投影
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 使用 LoRA：先降维投影，再归一化，最后升维投影到多头 Q
            self.wq_a = Linear(self.dim, self.q_lora_rank)          # 降维
            self.q_norm = RMSNorm(self.q_lora_rank)                 # 归一化
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # 升维到头的维度

        # ---- Key-Value 联合低秩投影 ----
        # wkv_a: 将输入投影到 kv_lora_rank + qk_rope_head_dim 维（其中 kv_lora_rank 是 KV 压缩部分,
        #         qk_rope_head_dim 是单独用于位置编码的 Key 部分）
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        # kv_norm: 对压缩后的 KV 潜变量做归一化
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # wkv_b: 将压缩后的 KV 潜变量展开到所有头，包含 nope 部分 和 value 部分
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )

        # ---- 输出投影（行并行，所有 GPU 结果聚合） ----
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # ---- softmax 缩放因子（含 YaRN 扩展调整） ----
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            # 当需要推理比训练更长的序列时，使用 YaRN 的 mscale 因子调整 attention 温度
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # ---- KV 缓存（根据注意力实现模式建立不同形状的缓存） ----
        if RuntimeConfig.default().attn_impl == "naive":
            # 朴素模式：缓存展开后的完整 K 和 V，每个头的 K/V 单独存储
            self.register_buffer(
                "k_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len,
                            self.n_local_heads, self.qk_head_dim),
                persistent=False,
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len,
                            self.n_local_heads, self.v_head_dim),
                persistent=False,
            )
        else:
            # "absorb" 模式：只缓存低秩 KV 潜变量 (kv_cache) 和位置编码 (pe_cache)
            # 这样可以大幅减少 KV 缓存的内存占用
            self.register_buffer(
                "kv_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len,
                            self.kv_lora_rank),
                persistent=False,
            )
            self.register_buffer(
                "pe_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len,
                            self.qk_rope_head_dim),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            x:          输入张量，形状 (batch_size, seq_len, dim)。
            start_pos:  当前序列的起始位置（用于 KV 缓存写入位置）。
            freqs_cis:  预计算的 RoPE 复指数位置编码。
            mask:       注意力 mask（因果 mask 或其他），形状 (seq_len, seq_len) 或可广播。

        Returns:
            torch.Tensor: 注意力输出，形状 (batch_size, seq_len, dim)。
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen  # 当前批次在缓存中的结束位置

        # ============================================================
        # 第1步：Query 投影，并拆分为 nope 和 rope 两部分
        # ============================================================
        if self.q_lora_rank == 0:
            # 直接投影：将输入 x 映射到所有头的 Query 维度
            q = self.wq(x)                              # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            # LoRA 投影：先降维、归一化、再升维，减少参数量
            q = self.wq_b(self.q_norm(self.wq_a(x)))    # (bsz, seqlen, n_heads * qk_head_dim)
        # 重塑为 (bsz, seqlen, n_local_heads, qk_head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # 沿最后一维拆分为非位置部分 (q_nope) 和位置编码部分 (q_pe)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # 对位置编码部分应用旋转位置编码
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # ============================================================
        # 第2步：Key-Value 联合投影，并获取 Key 的 rope 部分
        # ============================================================
        # wkv_a 将输入映射到 (低秩 KV 潜变量) + (位置编码 Key 部分)
        kv = self.wkv_a(x)                              # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        # 拆分出压缩后的 KV 潜变量 和 独立的 position-aware Key
        kv, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # 对 position-aware Key 应用旋转位置编码（增加一个头维度用于广播）
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)   # (bsz, seqlen, 1, qk_rope_head_dim)

        # ============================================================
        # 第3步：计算注意力分数（区分 "naive" 和 "absorb" 两种模式）
        # ============================================================
        if RuntimeConfig.default().attn_impl == "naive":
            # ----- 朴素模式：标准多头注意力 -----
            # 拼接 q_nope 和 q_pe 得到完整的 Query
            q = torch.cat([q_nope, q_pe], dim=-1)               # (bsz, seqlen, n_local_heads, qk_head_dim)
            # 对 KV 潜变量先归一化，再通过 wkv_b 展开到所有头的 k_nope 和 v
            kv_out = self.wkv_b(self.kv_norm(kv))               # (bsz, seqlen, n_local_heads, qk_nope_head_dim + v_head_dim)
            kv_out = kv_out.view(
                bsz, seqlen, self.n_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            # 拆分为 k_nope 和 v
            k_nope, v = torch.split(
                kv_out, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            # 拼接 k_nope 和 k_pe 得到完整的 Key
            k = torch.cat(
                [k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)],
                dim=-1,
            )                                                   # (bsz, seqlen, n_local_heads, qk_head_dim)
            # 写入缓存（后续 token 会用到）
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 计算 Q 与缓存中所有 K 的点积注意力分数
            scores = (
                torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
                * self.softmax_scale
            )
        else:
            # ----- Absorb 模式：将投影矩阵部分 "吸收" 进 Query，直接使用缓存计算 -----
            # 获取 wkv_b 的权重（若已量化则先反量化）
            wkv_b_weight = self.wkv_b.weight
            if self.wkv_b.scale is not None:
                wkv_b_weight = weight_dequant(
                    self.wkv_b.weight, self.wkv_b.scale, RuntimeConfig.default().block_size
                )
            # 重塑为 (n_local_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
            wkv_b_weight = wkv_b_weight.view(
                self.n_local_heads, -1, self.kv_lora_rank
            )
            # 将 wkv_b 的 nope 部分（前 qk_nope_head_dim 个通道）与 q_nope 相乘，
            # 相当于把 K 的 nope 投影 "吸收" 进 Q，直接产生分数的一部分
            q_nope = torch.einsum(
                "bshd,hdc->bshc",
                q_nope,
                wkv_b_weight[:, : self.qk_nope_head_dim],
            )                                                   # (bsz, seqlen, n_local_heads, kv_lora_rank)
            # 缓存 KV 潜变量（归一化后）和位置编码
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)      # (bsz, seqlen, kv_lora_rank)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)       # (bsz, seqlen, qk_rope_head_dim)
            # 注意力分数由两部分组成：nope 部分的点积 + rope 部分的点积
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

        # ============================================================
        # 第4步：应用 mask 和 softmax 得到注意力权重
        # ============================================================
        if mask is not None:
            # 加上 mask（因果 mask 为 -inf，消除未来位置的影响）
            scores += mask.unsqueeze(1)                             # mask 形状通常 (seqlen, seqlen)
        # softmax 归一化（在 float32 下计算以保证数值稳定性）
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # ============================================================
        # 第5步：用注意力权重加权聚合 Value，得到输出
        # ============================================================
        if RuntimeConfig.default().attn_impl == "naive":
            # 朴素模式：直接用 scores 权重对缓存的 V 进行加权求和
            x = torch.einsum(
                "bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos]
            )                                               # (bsz, seqlen, n_local_heads, v_head_dim)
        else:
            # Absorb 模式：先用 scores 对缓存中的 KV 潜变量进行加权求和
            x = torch.einsum(
                "bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos]
            )                                               # (bsz, seqlen, n_local_heads, kv_lora_rank)
            # 再通过 wkv_b 的 v 部分（后 v_head_dim 个通道）将潜变量映射回 Value 空间
            x = torch.einsum(
                "bshc,hdc->bshd", x, wkv_b_weight[:, -self.v_head_dim:]
            )                                               # (bsz, seqlen, n_local_heads, v_head_dim)

        # ============================================================
        # 第6步：输出投影，将所有头拼接后通过行并行线性层
        # ============================================================
        # 将头维度展平并与 seqlen 维度合并： (bsz, seqlen, n_local_heads * v_head_dim)
        # 然后通过 RowParallelLinear 投影回 dim 维度，内部自动完成 all_reduce 聚合
        x = self.wo(x.flatten(2))                          # (bsz, seqlen, dim)
        return x
