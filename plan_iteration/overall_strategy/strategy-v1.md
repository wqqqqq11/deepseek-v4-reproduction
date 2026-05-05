# 目标：复现deepseek-v4模型，不追求百分之百的复现，需要尽量保证架构以及训练推理方式与deepseek-v4完全一致,参数为10M。专家训练减少为3大领域

# 复现概述
分为 架构层、训练全阶段（5阶段）、推理层 三部分。

# 架构层面
## MegaMoE 混合专家架构
    稀疏激活路由、多专家分层结构
    哈希路由层 + 条件记忆模块
    专家负载均衡约束、专家 Wave 调度机制
## mHC 混合稀疏注意力
    CSA 稠密局部注意力 + HCA 稀疏全局注意力 双分支融合
    原生支持 4k token 上下文 结构设计
## 流形约束超连接
    残差链路特殊约束设计，解决超深 MoE 模型梯度消失
## 基础组件固定不变
    归一化：RMSNorm
    激活函数：SwiGLU
    位置编码：V4 自研新式位置编码
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
    训练目标：标准自回归 LM + FIM 填空训练
    上下文：初始 4K
    精度：FP8 训练
    优化器：主体 Muon + 头尾 AdamW
    并行策略：TP+PP+EP+DP 四合一混合并行
    训练要求：训练至 loss 完全收敛，不提前停止

## 阶段 2：分领域独立专家单独训练
    独立拆分 5 大领域（为了简化训练，这里采用3大领域），每个领域用专属数据单独微调训练：
    数据总量：48M tokens
    数学推理专家（数据集：https://huggingface.co/datasets/openbmb/UltraData-Math）（16M Tokens）
    代码生成专家（数据集：https://huggingface.co/datasets/bigcode/the-stack-v2）（16M Tokens）
    科研学术专家（数据集：https://huggingface.co/datasets/ccdv/arxiv-summarization）（16M Tokens）
    数据：每个领域只用自己的垂直语料
    训练方式：固定 V4 主干，只精调对应领域专家分支
    目的：让每个专家在自己领域达到最优能力

## 阶段 3：OPD 最优路径蒸馏合并
    把阶段 2 训练好的3 个独立领域专家，通过 OPD 蒸馏策略，合并融合进同一个主模型
    训练方式：知识蒸馏 + 路径权重学习
    效果：一个模型同时拥有所有专家的领域能力，且互相不干扰

    数据集构成（总共20M tokens）：
    数据类型	        Token 量级	        占比	        数据来源
    通用数据	         11M	            55%	            阶段 1 的通用高质量子集
    代码 + 数学数据	     6M	                30%	            阶段 2 数学 / 代码专家的垂直数据核心子集
    科研数据	         3M	                15%	            阶段 2 科研专家的垂直数据核心子集

## 阶段 4：超长上下文扩窗续训（简化版）
    上下文：从 4K → 8K
    数据：使用阶段 1 的通用普通文本（10M tokens 数据量）
    训练：只续训1000步

## 阶段 5：对齐阶段（SFT + GRPO 强化学习）
### 有监督微调 SFT（1.5M）：通用指令、对话、任务对齐
    领域	    占比	数据量	        数据集
    通用指令	60%	    0.9M tokens 	https://huggingface.co/datasets/tatsu-lab/alpaca_cleaned（经典通用指令集）
    数学推理	25%	    0.45M tokens 	https://huggingface.co/datasets/openbmb/UltraMath-Instruct（数学指令）
    代码生成	15%	    0.15M tokens 	https://huggingface.co/datasets/HuggingFaceH4/code_alpaca_2k（代码指令）
### GRPO 强化学习（1M）：偏好对齐、逻辑纠错、拒绝有害输出
    完成人类偏好对齐
    领域	    占比	数据量	        数据集
    通用偏好	60%	    0.6M tokens 	https://huggingface.co/datasets/openbmb/UltraChat-Preference（通用对话偏好）
    数学偏好	25%	    0.25M tokens 	https://huggingface.co/datasets/allenai/math-preference-v1（数学解题偏好）
    代码偏好	15%	    0.15M tokens   https://huggingface.co/datasets/lmsys/code-preference-small（代码质量偏好）


# 训练工程底层必须对齐的细节
## 优化器严格二分
    Transformer 主干 / MoE 专家：Muon 二阶优化器
    嵌入层、分类头：AdamW
## 使用 DeepGEMM 算子
    训练前向 / 反向全部基于官方 DeepGEMM 融合内核，不能自己写普通 MatMul
## 混合并行不变
    张量并行 TP、流水线 PP、专家并行 EP、数据并行 DP 全套拉满
## 负载均衡
    MoE 路由复刻 V4 官方负载均衡损失
## 数据配比
    通用 55% + 代码数学 30% + 科研 15%，所有训练阶段都遵循这个底层分布

# 推理层面：必须 100% 对齐的要点
    复用 DeepGEMM 推理内核
    mHC 双分支稀疏注意力推理路由逻辑
    MegaMoE 动态专家激活、Wave 调度推理
    FP8/FP4 推理量化格式和官方一致
    16k token 超长上下文 KV 缓存分片策略
    推理时专家路由哈希算法和训练完全一致

# 项目代码结构 (src/models/)

```
src/models/
├── config.py                # 模型配置集中管理，定义 V4Config 类
├── layers.py                # 基础层组件：RMSNorm + SwiGLU + 流形约束超连接
├── rotary_embedding.py      # V4 自研位置编码，替代标准 RoPE
├── mhc_attention.py         # mHC 混合稀疏注意力：CSA稠密局部 + HCA稀疏全局 + 双分支融合
├── moe.py                   # MegaMoE 完整实现：哈希路由 + 专家池 + 负载均衡 + Wave调度
├── block.py                 # Transformer Decoder Block：整合所有子模块
└── transformer.py           # DeepSeek-V4 主模型入口，组装完整模型
```

## 各文件详细职责

### config.py
- **作用**：集中管理所有模型超参数
- **内容**：V4Config 类，包含维度、层数、头数、MoE 专家数、上下文长度、精度配置等

### layers.py
- **作用**：基础层组件库
- **包含**：
  - `RMSNorm`：根均方层归一化
  - `SwiGLU`：激活函数门控单元
  - `ManifoldHyperConnection`：流形约束超连接（V4 核心创新，解决超深 MoE 梯度消失）

### rotary_embedding.py
- **作用**：V4 自研位置编码实现
- **区别于标准 RoPE**：根据 V4 论文描述的特殊位置编码方案

### mhc_attention.py
- **作用**：mHC 混合稀疏注意力完整实现
- **包含组件**：
  - `CSA` (Context-Sparse Attention)：稠密局部注意力，处理短距离依赖，计算密集但精度高
  - `HCA` (Hierarchical Context Attention)：稀疏全局注意力，处理长距离依赖，稀疏计算节省内存
  - `MHCFusion`：双分支融合层，加权合并 CSA 和 HCA 输出
- **输出**：融合后的注意力表示

### moe.py
- **作用**：MegaMoE 混合专家架构完整实现
- **包含模块**：
  - `HashRouter`：哈希路由层，决定 token 分配给哪些专家
  - `ExpertPool`：专家池管理，包含 N 个领域专家
  - `LoadBalancer`：专家负载均衡约束损失
  - `WaveScheduler`：专家 Wave 调度机制，优化并行效率

### block.py
- **作用**：单个 Transformer Decoder 层
- **数据流**：`Input → RMSNorm → mHCAttention → 流形超连接 → RMSNorm → MoE → 流形超连接 → Output`

### transformer.py
- **作用**：DeepSeek-V4 主模型
- **数据流**：`Embeddings → Block × N → RMSNorm → LM Head`
- **优化器绑定**：主干用 Muon，Embeddings/LMHead 用 AdamW

## 模块依赖关系

```
transformer.py
    ├── Embeddings (AdamW 优化)
    ├── Block × N
    │   ├── RMSNorm (layers.py)
    │   ├── mhc_attention.py (包含CSA+HCA+融合)
    │   ├── ManifoldHyperConnection (layers.py)
    │   ├── RMSNorm (layers.py)
    │   └── moe.py
    │       ├── HashRouter
    │       ├── ExpertPool
    │       ├── LoadBalancer
    │       └── WaveScheduler
    ├── RMSNorm (layers.py)
    └── LMHead (AdamW 优化)

All modules depend on:
    └── config.py (V4Config)
```

## 关键接口约定

| 模块 | 输入 | 输出 | 备注 |
|------|------|------|------|
| mHC | [B, N, D] | [B, N, D] | 融合 CSA(局部) + HCA(全局) |
| MoE | [B, N, D] | [B, N, D] | 路由后 expert 计算 |
| Block | [B, N, D] | [B, N, D] | 完整 decoder 层 |
| Transformer | [B, N] | [B, N, V] | 输入 token ids，输出 logits |
