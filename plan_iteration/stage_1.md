# 阶段 1：通用基座大规模预训练
    定位：打底通用语言、百科、常识、基础文理知识
    数据总量：650M Tokens
    数据集： 中文（60%）：https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
            英文（40%）：https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
## 数据集预处理
1. 下载 Fineweb 中文数据、RedPajama 英文数据

2. 原始字段抽取
   Fineweb 只取 text 字段。 原始字段：['text', 'source', 'source_from_meta']
   RedPajama 只取raw_content 字段并且映射为text。原始字段：['raw_content', 'doc_id', 'meta', 'quality_signals']

3. 基础清洗
   去 HTML
   去 URL
   去广告
   去异常符号
   去空文本
   去过短文本
   去超长异常文本

4. 质量过滤
   中文数据：中文字符比例过低的丢弃
   英文数据：英文字符比例过低的丢弃
   标点/数字/乱码比例异常的丢弃

5. 去重
   近似去重

6. 文档级划分
   先分别将chinese和english数据集划分为 train / val / test （8:1:1）后再两两合并 train / val / test (8:1:1)

7. tokenize
   每篇文档 encode 成 token ids
   每篇文档末尾加 eos_token_id

8. 拼接
   train 文档拼成 train_token_stream
   val 文档拼成 val_token_stream
   test 文档拼成 test_token_stream

9. 转为.bin格式
   读取每个 split 的 jsonl，把 input_ids 展平写入连续的 bin 文件，保持原有切分结构不变。

10. 训练
   model(x)
   causal LM loss

11. 实现思路

### 11.1 项目代码结构 (src/data/)

```
src/data/
├── config.py                    # 数据预处理配置（路径、比例、阈值等）
├── downloader.py                # 数据集下载模块（HuggingFace 流式下载）
├── cleaners/
│   ├── html_cleaner.py          # HTML 标签清洗
│   ├── url_cleaner.py           # URL 清洗
│   ├── text_cleaner.py          # 异常符号、乱码、空文本、长短文本过滤
│   └── quality_filter.py        # 质量过滤（语言比例、标点/数字比例）
├── deduplicator.py              # 去重
├── tokenizer.py                 # Byte-level BPE Tokenize + eos_token 处理
├── chunker.py                   # 文档拼接 + 切块（4097→x[4096]/y[4096]）
├── pipeline.py                  # 完整预处理流水线编排（支持断点续处理）
└── utils/
    └── io_utils.py              # 文件读写、流式处理工具
```

### 11.2 数据集目录划分 (datasets/stage_1_datasets/)

```
datasets/
└── stage_1_datasets/
    ├── README.md                    # 数据版本、来源、处理参数记录
    │
    ├── 0_raw/                       # 原始下载数据（HuggingFace 缓存）
    │   ├── fineweb_edu_chinese/     # 中文原始数据
    │   └── redpajama_v2/            # 英文原始数据
    │
    ├── 1_cleaned/                   # 清洗后数据（按语言分离）
    │   ├── chinese/
    │   │   └── part_{0000..N}.jsonl     # {"text": "...", "source": "fineweb"}
    │   └── english/
    │       └── part_{0000..N}.jsonl     # {"text": "...", "source": "redpajama"}
    │
    ├── 2_deduplicated/              # 去重后数据
    │   ├── chinese/
    │   └── english/
    │
    ├── 3_tokenized/                 # Tokenize 后数据（含 eos_token_id）
    │   ├── chinese/
    │   └── english/
    │
    ├── 4_merged/                    # 中英文按比例交错混合
    │   ├── train.jsonl              # 60% 中文 + 40% 英文 混合流
    │   ├── val.jsonl
    │   └── test.jsonl
    │
    └── 5_final/                     # 最终训练数据（.bin 格式）
        ├── train.bin                # x,y 对，连续内存块（约 240MB）
        ├── val.bin
        └── test.bin
```

### 11.3 关键设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| **处理方式** | 流式处理 | 使用 `datasets.streaming=True`，避免内存爆炸 |
| **中间存储格式** | .jsonl | 每行一个文档，便于调试和断点续处理 |
| **最终存储格式** | .bin | 二进制内存映射格式，PyTorch DataLoader 快速加载 |
| **去重粒度** | 文档级 | 精确去重（MD5 哈希），可选近似去重 |
| **Tokenizer** | Byte-level BPE | 与 DeepSeek 原版一致，可用 `transformers` 实现 |
| **混合策略** | 先分后合 | 中英文分别处理完成，最后在 chunker 按 60:40 交错拼接 |
| **数据切分** | 8:1:1 | 仅在最下游（4_merged 阶段）做 train/val/test 切分 |

### 11.4 流水线执行顺序

```
阶段 0 (downloader):   HuggingFace → 0_raw/              (流式下载，不存储完整副本)
阶段 1 (cleaners):     0_raw/ → 1_cleaned/               (清洗+质量过滤)
阶段 2 (deduplicator): 1_cleaned/ → 2_deduplicated/      (文档级精确去重)
阶段 3 (tokenizer):    2_deduplicated/ → 3_tokenized/  (BPE编码+加eos)
阶段 4 (chunker):      3_tokenized/ → 4_merged/         (中英文 60:40 混合+8:1:1切分)
阶段 5 (binarize):     4_merged/ → 5_final/              (转为.bin二进制格式)
```

### 11.6 断点续处理机制

每个阶段完成后写入 `checkpoint.json`：
```json
{
  "stage": 3,
  "output_dir": "3_tokenized/chinese/",
  "processed_docs": 1250000,
  "skipped_docs": 23000,
  "md5_hash": "...",
  "timestamp": "2025-05-13T16:30:00"
}
```

重新运行 pipeline 时检测 checkpoint，自动跳过已完成阶段。

### 11.7 配置集中管理 (config.py)

```python
@dataclass
class DataConfig:
    # 路径
    base_data_dir: str = "datasets/stage_1_datasets"
    
    # 数据比例
    chinese_ratio: float = 0.6
    english_ratio: float = 0.4
    split_ratio: Tuple[float] = (0.8, 0.1, 0.1)  # train/val/test
    
    # 处理参数
    context_size: int = 4096
    chunk_stride: int = 4097  # x=4096, y=4096
    
    # 清洗阈值
    min_text_length: int = 100        # 字符
    max_text_length: int = 100000     # 字符
    min_chinese_ratio: float = 0.3    # 中文数据中文比例
    min_english_ratio: float = 0.5    # 英文数据英文比例
    
    # Tokenizer
    vocab_size: int = 16000
    eos_token_id: int = 2
```

## 预训练

### 12. 训练代码项目架构

基于模块化设计原则，训练代码分为**入口层**、**配置层**、**核心层**和**阶段层**。

#### 12.1 完整项目结构树

```
deepseek-v4-reproduction/
├── mian.py                              # 训练入口：解析参数、启动对应阶段训练
│
├── src/
│   ├── models/                          # 模型架构
│   │   ├── config.py                    # ModelArgs 模型配置
│   │   ├── transformer.py               # 主模型入口
│   │   ├── block.py                     # Transformer Block (HC + MLA + MoE)
│   │   ├── mla_attention.py             # 多头潜在注意力 (CSA + HCA)
│   │   ├── moe.py                       # 混合专家架构
│   │   ├── layers.py                    # 基础层 (RMSNorm, Linear)
│   │   ├── rotary_embedding.py          # YaRN RoPE 位置编码
│   │   ├── kernel.py                    # FP8/FP4 量化 kernel fallback
│   │   └── RuntimeConfig.py             # 运行时配置 (分布式、量化)
│   │
│   ├── data/                            # 数据处理
│   │   ├── config.py                    # DataConfig 数据配置
│   │   ├── pipeline.py                  # 数据流水线编排
│   │   ├── binarizer.py                 # JSONL → BIN 转换
│   │   ├── tokenizer.py                 # Tokenizer 实现
│   │   ├── cleaners/                    # 清洗模块
│   │   └── utils/                       # 工具函数
│   │
│   ├── training/                        # 训练核心模块
│   │   ├── __init__.py
│   │   ├── config.py                    # TrainArgs 训练配置 (学习率、批次等)
│   │   ├── dataset.py                   # TokenDataset 加载 .bin 文件
│   │   ├── optimizer.py                 # Muon + AdamW 混合优化器
│   │   ├── trainer.py                   # BaseTrainer 训练循环基类
│   │   ├── checkpoint.py                # 检查点保存/加载
│   │   ├── logger.py                    # 日志记录 (wandb/tensorboard)
│   │   ├── lr_scheduler.py              # 学习率调度 (cosine decay)
│   │   ├── utils.py                     # 训练工具函数
│   │   │
│   │   └── stages/                      # 各阶段训练实现
│   │       ├── stage1_pretrain.py       # 阶段1：通用预训练
│   │       ├── stage2_context_extend.py # 阶段2：扩窗续训 (1K→2K)
│   │       ├── stage3_domain_sft.py     # 阶段3：分领域 SFT
│   │       ├── stage3_domain_grpo.py    # 阶段3：分领域 GRPO
│   │       ├── stage4_opd_distill.py    # 阶段4：OPD 最优路径蒸馏
│   │       └── stage5_final_align.py    # 阶段5：最终对齐 (轻量SFT)
│   │
│   └── configs/                         # JSON 配置存储
│       ├── stage1_config.json           # 阶段1配置
│       ├── stage2_config.json
│       └── ...
├── checkpoints/                         # 【运行生成】模型检查点
│   ├── stage1/                          # 阶段1检查点
│   │   ├── best.pt
│   │   └── step_50000.pt
│   ├── stage2/
│   └── ...
│
├── logs/                                # 【运行生成】训练日志
│   ├── stage1/
│   └── ...
│
├── datasets/                            # 数据集
│   └── stage_1_datasets/
│       ├── 6_binarize/                  # 二进制数据
│       └── checkpoint.json              # 预处理进度
│
└── plan_iteration/                      # 项目规划
    ├── overall_strategy/
    │   └── strategy-v1.md
    └── stage_1.md                       
```

#### 12.2 核心模块职责

| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `mian.py` | 训练入口，解析命令行参数，分发到对应阶段 | `--stage`, `--config`, `--resume` | 启动训练器 |
| `TrainArgs` | 训练超参配置 | YAML/JSON | 配置对象 |
| `TokenDataset` | 高效加载 .bin 数据 | `*.bin` 路径 | `(x, y)` Tensor 对 |
| `Muon` | 二阶优化器（主模块） | 梯度 | 参数更新 |
| `AdamW` | 一阶优化器（嵌入/头） | 梯度 | 参数更新 |
| `BaseTrainer` | 训练循环基类 | 模型、数据、配置 | 训练好的模型 |
| `StageXTrainer` | 各阶段特化逻辑 | 配置 | 阶段输出 |

#### 12.3 使用方式

```bash
# 阶段1：通用预训练
python mian.py --stage 1 --config configs/stage1_pretrain.yaml

```

---

## 阶段1 预训练实现方案（最终版）

### 13. 模型架构设计（Dense + MLA）

阶段1采用 **Dense 架构**（暂不叠加 MoE 和 mHC），保留完整的 MLA 注意力机制。

| 配置项 | 值 | 说明 |
|--------|-----|------|
| vocab_size | < 6400 | minimind tokenizer 适配 |
| hidden_size | 256 | 小参数模型 |
| num_layers | 8 | 8层 Transformer |
| num_heads | 8 | 8 头注意力 |
| kv_lora_rank | 32 | KV 压缩到 32 维（1/8 压缩）|
| q_lora_rank | 64 | Query 压缩到 64 维 |
| intermediate_size | 688 | SwiGLU FFN 中间层（约 2/3 * 4 * hidden）|
| max_seq_len | 1024 | 阶段1序列长度 |
| rope_dim | 32 | YaRN Partial RoPE 仅最后32维 |
| dropout | 0.0 | 预训练不使用 dropout |

#### 13.1 MLA 注意力配置

- **CSA（Compressed Sparse Attention）**：局部窗口注意力
- **HCA（Heavily Compressed Attention）**：全局压缩稠密注意力
- **低秩压缩**：通过可学习的投影矩阵压缩 KV 到 latent 空间
- **推理优化**：仅缓存 latent 向量，不缓存完整 KV

#### 13.2 FFN 配置

- **类型**：Dense SwiGLU（非 MoE）
- **公式**：`SwiGLU(x) = (xW_1 * SiLU(xW_2))W_3`
- **中间维度**：688（标准 SwiGLU 2/3 规则）

#### 13.3 精度策略

| 模块 | 精度 | 实现方式 |
|------|------|----------|
| 前向 Linear | FP8 | `torch.float8_e4m3fn` 模拟（BF16计算 + per-tensor scale）|
| 反向激活/权重 | FP8 | 同上 |
| 优化器状态 | FP16 | momentum 保持足够精度 |
| 嵌入/头 | BF16 | 稳定优先 |

---

### 14. 优化器设计（官方 Muon Algorithm 1）

严格遵循 DeepSeek-V4 论文 Algorithm 1 实现。

#### 14.1 适用范围

| 参数组 | 优化器 | 说明 |
|--------|--------|------|
| Attention 矩阵 (WQ, WK, WV, WO) | **Muon** | 主模块用二阶优化 |
| FFN 矩阵 (W1, W2, W3) | **Muon** | 主模块用二阶优化 |
| 所有投影矩阵 | **Muon** | 主模块用二阶优化 |
| Embedding 层 | AdamW | 稳定优先 |
| Prediction Head | AdamW | 稳定优先 |
| RMSNorm 权重 | AdamW | 稳定优先 |
| 所有 Bias | AdamW | 稳定优先 |

#### 14.2 Muon 超参（固定）

```python
mu = 0.95              # 动量系数
lambda_wd = 0.1        # 权重衰减
gamma = 0.18           # 更新重缩放因子（对齐 AdamW）
nesterov = True        # 启用 Nesterov 动量
learning_rate = 3e-4   # 学习率
```

#### 14.3 核心算法（Algorithm 1）

```
Algorithm 1 Muon Optimizer for DeepSeek-V4
Require: 学习率 η, 动量 μ, 权重衰减 λ, 更新重缩放因子 γ

for each training step t do
  for each 独立权重矩阵 W ∈ R^{n×m} do
    G_t = ∇^W L_t(W_{t-1})                      # 计算梯度
    M_t = μ · M_{t-1} + G_t                      # 动量累积
    O'_t = HybridNewtonSchulz(μ · M_t + G_t)     # Nesterov + 混合牛顿-舒尔茨正交化
    O_t = O'_t · √max(n,m) · γ                   # 重缩放更新矩阵的 RMS
    W_t = W_{t-1} · (1 - ηλ) - η · O_t           # 权重衰减 + 参数更新
  end for
end for
```

#### 14.4 混合 Newton-Schulz 迭代（10次）

分两个阶段执行，使用不同收敛系数：

- **前 8 次迭代**：快速收敛系数 `(3.4445, -4.7750, 2.0315)`
  ```python
  T = a * Z - b * Z @ Y @ Z + c * Z @ Y @ Z @ Y @ Z
  Y = a * Y - b * Y @ Y @ Y + c * Y @ Y @ Y @ Y @ Y
  ```

- **后 2 次迭代**：稳定精调系数 `(2, -1.5, 0.5)`
  ```python
  T = 2 * Z - 1.5 * Z @ Y @ Z + 0.5 * Z @ Y @ Z @ Y @ Z
  Y = 2 * Y - 1.5 * Y @ Y @ Y + 0.5 * Y @ Y @ Y @ Y @ Y
  ```

#### 14.5 稳定性设计

- **不使用 QK-Clip**：因 CSA/HCA 已对 Q/K 做 RMSNorm，避免注意力 logit 爆炸
- **梯度裁剪**：全局统一裁剪到 1.0

---

### 15. 损失函数设计

#### 15.1 预训练阶段损失

| 损失类型 | 权重 | 说明 |
|----------|------|------|
| **LM Loss** | 1.0 | 标准自回归交叉熵损失 |
| **MTP Loss** | 0.3（衰减阶段→0.1） | 多令牌预测损失，预测未来2个token |

#### 15.2 MTP（Multi-Token Prediction）实现

- **预测深度**：2 个 token（主输出 + 未来1位 + 未来2位）
- **实现方式**：主模型输出层旁增加轻量级 MTP 头（独立 Linear）
- **权重共享**：与主输出共享 embedding，但使用独立预测头
- **权重衰减策略**：
  - 预热+峰值阶段：mtp_weight = 0.3
  - 余弦衰减阶段：mtp_weight 从 0.3 线性降到 0.1

#### 15.3 关于 MoE 负载均衡损失

当前为 Dense 版本，暂不实现。后续叠加 MoE 时再加入（weight=0.0001）。

---

### 16. 学习率调度（三段式）

```python
# 调度参数
max_lr = 3e-4
min_lr = 3e-5
warmup_steps = 2000       # ~10% of 20K
peak_steps = 6000        # ~30% of 20K
cosine_steps = 12000     # ~60% of 20K
```

#### 16.1 三段式逻辑

```python
if step < warmup_steps:
    # 阶段1：线性预热
    lr = max_lr * (step / warmup_steps)
elif step < warmup_steps + peak_steps:
    # 阶段2：恒定峰值
    lr = max_lr
else:
    # 阶段3：余弦衰减
    progress = (step - warmup_steps - peak_steps) / cosine_steps
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

---

### 17. 训练配置汇总

| 配置项 | 值 |
|--------|-----|
| Epochs | 2 |
| Batch size | 32 |
| Gradient accumulation | 1 |
| Total steps/epoch | ~20,000 |
| Grad clip | 1.0（全局统一） |
| Random seed | 42 |
| torch.compile | **不使用** |
| EMA | **不使用** |

#### 17.1 数据采样策略

- **策略**：无放回随机打乱（每个 epoch shuffle）
- **实现**：每个 epoch 用不同但确定的种子 `epoch_seed = 42 + epoch`
- **访问方式**：随机访问 .bin 的索引，非顺序读取

#### 17.2 检查点与验证

| 配置 | 值 |
|------|-----|
| 保存策略 | 只保存 best.pt（覆盖式，最优验证 loss） |
| 验证频率 | 每个 epoch 结束（完整验证集） |
| Early stopping | 无（固定2 epochs） |
| Resume | 不支持 |

---

### 18. 日志监控设计（TensorBoard + tqdm）

#### 18.1 TensorBoard 指标

| 指标类别 | 具体指标 | 记录频率 |
|----------|----------|----------|
| **损失** | `loss/total`, `loss/lm`, `loss/mtp` | every step |
| **学习率** | `lr/muon`, `lr/adamw` | every step |
| **吞吐量** | `speed/tokens_per_sec`, `speed/samples_per_sec` | every step |
| **梯度** | `grad_norm/global` | every 10 steps |
| **硬件** | `gpu/memory_mb`, `gpu/utilization` | every 10 steps |
| **分布** | 参数/梯度直方图（hist） | every 10 steps |
| **验证** | `val/loss`, `val/ppl` | every epoch |

#### 18.2 tqdm 进度条格式

```
Epoch 1/2 [00:15<02:30] Step 1250/20000 | loss: 2.847 (lm: 2.541 + mtp: 1.020) | lr: 2.8e-4 | tok/s: 12500 | gpu: 4.2GB
```

---

### 19. 代码文件结构（更新版）

```
deepseek-v4-reproduction/
├── main.py                          # 统一入口: --stage 1/2/3/4/5
├── src/
│   ├── models/
│   │   ├── config.py                # ModelArgs 模型配置
│   │   ├── transformer.py           # DeepSeekV4 (Dense + MLA + MTP)
│   │   ├── mla_attention.py         # MLA 实现 (CSA + HCA + 低秩压缩)
│   │   ├── dense_mlp.py             # Dense SwiGLU FFN
│   │   ├── rotary_embedding.py      # YaRN Partial RoPE
│   │   ├── fp8_quant.py             # FP8/FP4 模拟 kernel
│   │   └── layers.py                # RMSNorm 等基础层
│   │
│   ├── training/
│   │   ├── config.py                # TrainConfig 训练配置
│   │   ├── muon.py                  # 官方 Muon 优化器 (Algorithm 1)
│   │   ├── lr_scheduler.py          # 三段式学习率调度器
│   │   ├── trainer.py               # 阶段1训练器
│   │   ├── dataset.py               # TokenDataset (mmap .bin + shuffle)
│   │   ├── logger.py                # TensorBoard + tqdm 封装
│   │   └── utils/
│   │       └── util.py              # 训练工具函数
│   │
│   └── configs/
│       └── stage1_pretrain.yaml     # 阶段1完整配置
│
├── datasets/
│   └── stage_1_datasets/
│       └── 5_final/
│           ├── train.bin
│           ├── val.bin
│           └── metadata.json        # vocab_size, num_tokens, eos_id 等
│
└── plan_iteration/
    ├── overall_strategy/
    │   └── strategy-v1.md
    └── stage_1.md                   # 本文件
```

---

### 20. 阶段1配置示例（stage1_pretrain.yaml）

```yaml
# =====================================
# 阶段1：通用基座预训练配置
# =====================================

# 模型参数
model:
  vocab_size: 6400              # minimind tokenizer (< 6400)
  hidden_size: 256
  num_layers: 8
  num_heads: 8
  kv_lora_rank: 32              # KV 压缩维度
  q_lora_rank: 64                # Query 压缩维度
  intermediate_size: 688         # SwiGLU 中间层
  max_seq_len: 1024
  rope_dim: 32                   # Partial RoPE 维度
  dropout: 0.0
  tie_word_embeddings: true      # 嵌入与输出头共享

# MTP 配置
mtp:
  num_future_tokens: 2           # 预测未来2个token
  loss_weight: 0.3               # 初始权重
  loss_weight_decay_end: 0.1     # 衰减阶段最终权重

# 训练参数
training:
  num_epochs: 2
  batch_size: 32
  gradient_accumulation: 1
  max_lr: 3.0e-4
  min_lr: 3.0e-5
  warmup_steps: 2000
  peak_steps: 6000
  cosine_steps: 12000
  weight_decay: 0.1
  grad_clip: 1.0
  seed: 42

# 优化器分组
optimizer:
  muon_groups: ["attn", "mlp"]
  adamw_groups: ["embed", "head", "norm", "bias"]
  muon_lr: 3.0e-4
  adamw_lr: 3.0e-4
  muon_momentum: 0.95
  muon_gamma: 0.18
  muon_nesterov: true

# 精度配置
precision:
  fp8_enabled: true              # 启用 FP8 模拟
  compute_dtype: "bfloat16"        # 实际计算精度
  optimizer_state_dtype: "float16"

# 数据路径
data:
  train_bin: "datasets/stage_1_datasets/5_final/train.bin"
  val_bin: "datasets/stage_1_datasets/5_final/val.bin"
  context_size: 1024

# 日志与检查点
logging:
  log_dir: "logs/stage1"
  checkpoint_dir: "checkpoints/stage1"
  save_best_only: true
  eval_every_epoch: true
  log_every_steps: 10
  tensorboard_port: 6006

# 硬件
hardware:
  device: "cuda"
  num_gpus: 1
  use_compile: false             # 不使用 torch.compile
```

---

### 21. 运行方式

```bash
# 阶段1预训练
python main.py --stage 1 --config configs/stage1_pretrain.yaml

# TensorBoard 查看
 tensorboard --logdir logs/stage1 --port 6006
```

---

**方案确定日期**：2026-05-17
**下一阶段**：代码实现
