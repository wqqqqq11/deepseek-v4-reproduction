# 目标：复现deepseek-v4模型（10M 参数），不追求参数规模和性能一致，重点复现以下机制：
        1. DeepSeekMoE：Hash-routed MoE + learned-gate MoE
        2. Hybrid Attention：CSA/HCA + sliding window KV
        3. mHC：多残差流 + Sinkhorn 约束
        4. 训练流程：预训练、扩窗续训、分领域专家 SFT/GRPO、统一蒸馏、最终对齐
        5. 推理流程：稀疏专家激活、压缩 KV、长上下文缓存

# 复现概述
分为 架构层、训练全阶段（5阶段）、推理层 三部分。

# 架构层面
## DeepSeekMoE 混合专家架构
- 前若干层：Hash-routed MoE
- 后续层：learned-gate DeepSeekMoE
- 每层 FFN 子层均采用 MoE FFN
- 专家结构：routed experts + shared expert
- 路由打分：sqrt(softplus)
- 负载均衡：aux-loss-free + 轻量 sequence-wise balance loss
## 混合稀疏注意力
    CSA 压缩稀疏注意力 + HCA 重度压缩稠密注意
    搭配滑动窗口注意力（swa）+ attention sink 支持 4k token 上下文 结构设计
## 流形约束超连接（mHC）
    残差链路特殊约束设计，解决超深 MoE 模型梯度消失
## 基础组件固定不变
    归一化：RMSNorm
    激活函数：SwiGLU
    位置编码：YaRN 扩展部分旋转位置编码（Partial RoPE），仅最后 64 维施加
    算子依赖：强制依赖 DeepGEMM 融合 CUDA 算子
## 数值精度架构
    训练 / 推理统一：FP8 × FP4 混合精度 结构设计
    主模块用 Muon 二阶优化，嵌入 / 分类头用 AdamW

# 训练策略
## 阶段 1：通用基座大规模预训练
    定位：打底通用语言、百科、常识、基础文理知识
    数据总量：120M Tokens
    数据集： 中文（60%）：https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
            英文（40%）：https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2

## 阶段 2：超长上下文扩窗续训（简化版）
    上下文：从 4K → 8K
    数据：使用阶段 1 的通用普通文本（10M tokens 数据量）
    训练：只续训1000步

## 阶段 3：分领域独立专家训练（SFT + GRPO）
    独立拆分 5 大领域（为了简化训练，这里采用3大领域），每个领域用专属数据单独微调训练：
    数据总量：48M tokens
    数据：每个领域只用自己的垂直语料

    训练方式：固定 V4 主干，只精调对应领域专家分支
    目的：让每个专家在自己领域达到最优能力

    专家类型	总 Token	SFT（70%，基础能力）	GRPO（30%，专家强化）	数据集与组合比例
    数学推理	16M	        11.2M	                4.8M	                SFT：
                                                                                1. openbmb/UltraData-Math：50% = 5.60M tokens
                                                                                2. AI-MO/NuminaMath-CoT：30% = 3.36M tokens
                                                                                3. EleutherAI/hendrycks_math：20% = 2.24M tokens

                                                                            GRPO：
                                                                                1. EleutherAI/hendrycks_math：40% = 1.92M tokens
                                                                                2. openai/gsm8k：35% = 1.68M tokens
                                                                                3. AI-MO/NuminaMath-CoT：25% = 1.20M tokens
    代码生成	16M	        11.2M	                4.8M	                SFT：
                                                                                1. bigcode/the-stack-v2 instruction 化子集：45% = 5.04M tokens
                                                                                2. ise-uiuc/Magicoder-OSS-Instruct-75K：30% = 3.36M tokens
                                                                                3. m-a-p/CodeFeedback-Filtered-Instruction：25% = 2.80M tokens

                                                                            GRPO：
                                                                                1. codeparrot/apps：50% = 2.40M tokens
                                                                                2. google-research-datasets/mbpp：25% = 1.20M tokens
                                                                                3. Vezora/Code-Preference-Pairs：25% = 1.20M tokens
    科研学术	16M	        11.2M	                4.8M	                 SFT：
                                                                                1. ccdv/arxiv-summarization：50% = 5.60M tokens
                                                                                    - 不直接塞全文
                                                                                    - 使用 title + abstract + section chunks
                                                                                    - 对超长论文做 chunking
                                                                                    - 控制单样本长度 <= 当前上下文窗口
                                                                                2. allenai/SciRIFF：30% = 3.36M tokens
                                                                                3. qiaojin/PubMedQA：20% = 2.24M tokens

                                                                            GRPO：
                                                                                1. allenai/SciRIFF：50% = 2.40M tokens
                                                                                2. qiaojin/PubMedQA：30% = 1.44M tokens
                                                                                3. allenai/scifact：20% = 0.96M tokens
    
    注意：SFT 和 GRPO 可以使用同源数据集，但必须切分不同样本池。
            test split 不参与训练。

## 阶段 4：OPD 最优路径蒸馏合并
    把阶段 3 训练好的3 个独立领域专家，通过 OPD 蒸馏策略，合并融合进同一个主模型
    训练方式：知识蒸馏 + 路径权重学习
    效果：一个模型同时拥有所有专家的领域能力，且互相不干扰

    数据集构成（总共20M tokens）：
    数据类型	        Token 量级	        占比	        数据来源
    通用数据	         11M	            55%	            阶段 1 的通用高质量子集
    代码 + 数学数据	     6M	                30%	            阶段 3 数学 / 代码专家的垂直数据核心子集
    科研数据	         3M	                15%	            阶段 3 科研专家的垂直数据核心子集

## 阶段 5：最终对齐（轻量 SFT）
### 有监督微调 SFT（1.5M）：通用指令、对话、任务对齐
    领域	    占比	数据量	        数据集
    通用指令	60%	    0.9M tokens 	https://huggingface.co/datasets/tatsu-lab/alpaca_cleaned（经典通用指令集）
    数学推理	25%	    0.45M tokens 	https://huggingface.co/datasets/openbmb/UltraMath-Instruct（数学指令）
    代码生成	15%	    0.15M tokens 	https://huggingface.co/datasets/HuggingFaceH4/code_alpaca_2k（代码指令）

# 项目代码结构 (src/models/)

```
src/models/
├── config.py                # 模型配置集中管理，定义 ModelArgs 数据类
├── RuntimeConfig.py         # 运行时环境配置（分布式、量化参数、实现模式）
├── layers.py                # 基础层组件：RMSNorm + 各类并行线性层
├── rotary_embedding.py      # YaRN 扩展的 RoPE 位置编码实现
├── kernel.py                # 量化 kernel fallback 实现（FP8/FP4 模拟）
├── mla_attention.py         # MLA 多头潜在注意力（含 CSA + HCA 稀疏注意力）
├── moe.py                   # MoE 混合专家架构（Gate + Expert + Shared Expert）
├── block.py                 # Transformer Decoder Block（MLA + MoE + Hyper-Connections）
└── transformer.py           # DeepSeek-V4 主模型入口，组装完整模型
```

## 各文件详细职责

### config.py
- **作用**：集中管理所有模型超参数
- **内容**：`ModelArgs` 数据类，包含维度、层数、头数、MoE 专家数、上下文长度、MLA 配置、YaRN 配置、HC 配置等

### RuntimeConfig.py
- **作用**：运行时环境配置（实例化设计，避免全局状态）
- **包含**：
  - `RuntimeConfig`：分布式参数（world_size, rank）、量化参数（block_size, gemm_impl）、注意力实现模式
  - 提供 `default()` 单例模式和 `from_distributed()` 工厂方法

### layers.py
- **作用**：基础层组件库（可拔插设计，不依赖全局变量）
- **包含**：
  - `linear()`：统一线性变换入口（支持 BF16/FP8 路径）
  - `Linear`：自定义线性层（支持量化权重）
  - `ColumnParallelLinear`：列并行线性层（输出维度切分）
  - `RowParallelLinear`：行并行线性层（输入维度切分，需 all_reduce）
  - `ParallelEmbedding`：分布式词嵌入层（按 world_size 切分词表）
  - `RMSNorm`：根均方层归一化

### rotary_embedding.py
- **作用**：YaRN 扩展的旋转位置编码（RoPE）
- **包含**：
  - `precompute_freqs_cis()`：预计算旋转位置编码的复指数值（带 LRU 缓存）
  - `apply_rotary_emb()`：将 RoPE 应用到输入张量

### kernel.py
- **作用**：FP8/FP4 量化 kernel 的纯 PyTorch fallback 实现
- **包含**：
  - `act_quant()`：激活值分块量化（模拟）
  - `fp4_act_quant()`：FP4 激活值量化（模拟）
  - `rotate_activation()`：Hadamard 旋转移位（模拟）
  - `hc_split_sinkhorn()`：Hyper-Connections 的 Sinkhorn 正则化分块
  - `weight_dequant()`：权重量化反量化（模拟）
  - `fp8_gemm()`：FP8 矩阵乘法（退化为 BF16）

### mla_attention.py
- **作用**：MLA (Multi-Head Latent Attention) 多头潜在注意力完整实现
- **包含组件**：
  - `Compressor`：KV 缓存压缩器（门控池化压缩为低频表示）
  - `Indexer`：压缩 KV 索引器（学习选择最相关的压缩 KV 位置）
  - `get_window_indices()`：生成滑动窗口索引（CSA 局部注意力）
  - `get_compress_indices()`：生成压缩 KV 固定间隔采样索引（HCA 全局稀疏）
  - `MLA`：主类，集成低秩 KV 压缩 + CSA 滑动窗口 + HCA 压缩稀疏检索
- **特性**：
  - 低秩联合压缩（q_lora_rank / kv_lora_rank）减少 KV 缓存内存
  - CSA (Context-Sparse Attention)：局部滑动窗口处理短距离依赖
  - HCA (Hierarchical Context Attention)：全局压缩稀疏处理长距离依赖

### moe.py
- **作用**：DeepSeekMoE 混合专家架构实现
- **包含模块**：
  - `MLP`：多层感知机（SwiGLU 激活，用于共享专家和密集层）
  - `Gate`：门控路由机制（支持 softmax/sigmoid/sqrt(softplus) 评分，可选 hash 路由）
  - `Expert`：单个专家（小型 MLP）
  - `MoE`：混合专家模块（整合 Gate + 多个 Expert + Shared Experts）
- **特性**：
  - 支持 hash 路由（前 n_hash_layers 层）和 learned gate 路由
  - 分层路由（n_expert_groups + n_limited_groups）
  - 共享专家（shared_experts）：所有 token 都经过
  - 分布式专家切分：专家均匀分配到各 GPU

### block.py
- **作用**：单个 Transformer Decoder 层，带 Hyper-Connections (HC) 流形超连接
- **结构**：
  - 输入/输出：`[B, S, hc_mult, D]`（维护 hc_mult 个并行残差流）
  - Attention 子层：`hc_pre` → RMSNorm → MLA → `hc_post`
  - FFN 子层：`hc_pre` → RMSNorm → MoE → `hc_post`
- **Hyper-Connections 实现**：
  - `hc_pre()`：通过 Sinkhorn 正则化将 hc_mult 个副本混合为 1 个输入子层
  - `hc_post()`：将子层输出扩展回 hc_mult 个副本，并与残差流混合

### transformer.py
- **作用**：DeepSeek-V4 主模型
- **数据流**：
  ```
  tokens → ParallelEmbedding → [B,S,D]
         → expand to hc_mult → [B,S,hc,D]
         → Block × N (每层 HC 混合) → [B,S,hc,D]
         → hc_head (合并 hc) → [B,S,D]
         → RMSNorm → LM Head → logits
  ```
- **Hyper-Connections 流程**：
  1. 嵌入后扩展：`[b,s,d]` → `[b,s,hc,d]`
  2. 每层 Block：输入/输出均为 `[b,s,hc,d]`
  3. 最终合并：`[b,s,hc,d]` → `[b,s,d]`

## 模块依赖关系

```
transformer.py
    ├── ParallelEmbedding (layers.py, 词表分布式切分)
    ├── Block × N
    │   ├── RMSNorm (layers.py)              # Pre-Norm
    │   ├── MLA (mla_attention.py)           # 含 CSA + HCA + 低秩压缩
    │   │   ├── Compressor                   # KV 压缩器
    │   │   ├── Indexer (可选)               # 压缩索引检索
    │   │   └── apply_rotary_emb             # RoPE 应用
    │   ├── hc_pre/hc_post                   # Hyper-Connections 混合
    │   ├── RMSNorm (layers.py)              # Pre-Norm
    │   └── MoE (moe.py)                     # 混合专家
    │       ├── Gate                         # 门控路由
    │       ├── Expert × n_local_experts    # 本地专家
    │       └── MLP (shared_experts)         # 共享专家
    ├── hc_head                              # 最终 HC 合并
    ├── RMSNorm (layers.py)                  # 最终归一化
    └── ColumnParallelLinear (head)          # 输出头（词表分布式切分）

依赖注入:
    ├── ModelArgs (config.py)                # 模型结构参数
    └── RuntimeConfig (RuntimeConfig.py)     # 运行时配置（分布式、量化）

底层支持:
    └── kernel.py                            # 量化/HC 计算 fallback
    └── rotary_embedding.py                  # RoPE 预计算
```

## 关键接口约定

| 模块 | 输入 | 输出 | 备注 |
|------|------|------|------|
| Block | [B, S, hc_mult, D] | [B, S, hc_mult, D] | 带 Hyper-Connections 的 decoder 层 |
| MLA | [B, S, D] | [B, S, D] | 含 CSA(局部窗口) + HCA(全局压缩) |
| MoE | [B, S, D] | [B, S, D] | Gate 路由 + Expert 计算 + Shared Expert |
| Transformer | [B, S] | [B, V] | 输入 token ids，输出 logits |
| hc_pre | [B, S, hc, D] | ([B, S, D], [B, S, hc], [B, S, hc, hc]) | 混合为1个 + 返回 post/comb 权重 |
| hc_post | ([B, S, D], [B, S, hc, D], [B, S, hc], [B, S, hc, hc]) | [B, S, hc, D] | 扩展并混合残差流 |

