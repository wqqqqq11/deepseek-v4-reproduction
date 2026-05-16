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

### 11.5 核心模块接口设计

**pipeline.py** - 主入口：
```python
class DataPipeline:
    def run_stage(self, stage_num: int, resume: bool = True)
    def run_all(self, start_stage: int = 0)
    def validate_stage(self, stage_num: int) -> bool
```

**downloader.py** - 流式下载：
```python
def download_dataset(dataset_name: str, output_dir: str, streaming: bool = True) -> Iterator[Dict]
```

**cleaners/** - 清洗链：
```python
class CleaningChain:
    def add_step(cleaner: BaseCleaner)
    def process(doc: Dict) -> Optional[Dict]  # None 表示丢弃
```

**tokenizer.py** - 编码：
```python
class TokenizerProcessor:
    def __init__(self, vocab_path: str)
    def encode_with_eos(text: str) -> List[int]
```

**chunker.py** - 拼接切块：
```python
def create_sliding_chunks(token_streams: List[Iterator], 
                         mix_ratio: List[float],
                         context_size: int = 4096) -> Iterator[Tuple[List[int], List[int]]]
# 返回 (x, y) 对，x=前4096 tokens, y=后4096 tokens
```

**io_utils.py** - 工具：
```python
def write_jsonl(filepath: str, records: Iterator[Dict])
def read_jsonl(filepath: str) -> Iterator[Dict]
def write_bin(filepath: str, xy_pairs: Iterator[Tuple[List, List]], dtype=np.uint16)
def memory_map_bin(filepath: str) -> np.memmap
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

# 从检查点恢复训练
python mian.py --stage 1 --config configs/stage1_pretrain.yaml \
    --resume checkpoints/stage1/step_10000.pt

# 阶段3：数学领域 SFT
python mian.py --stage 3 --domain math --config configs/stage3_math.yaml

# 阶段3：代码领域 GRPO
python mian.py --stage 3 --domain code --config configs/stage3_code.yaml \
    --use_grpo
```

#### 12.4 关键设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| **入口文件** | `mian.py` | 统一入口，通过 `--stage` 分发 |
| **配置格式** | YAML + JSON | YAML 人工编辑，JSON 运行时存储 |
| **数据集** | `np.memmap` | 高效读取大文件，支持随机访问 |
| **优化器** | Muon + AdamW 混合 | 主模块用 Muon，嵌入/头用 AdamW |
| **训练基类** | `BaseTrainer` | 各阶段继承，复用核心逻辑 |
| **精度** | BF16 | 平衡训练速度和数值稳定性 |
| **损失函数** | 自回归因果 LM | 标准 next-token prediction |

#### 12.5 配置继承关系

```
BaseConfig (通用参数)
    ├── ModelArgs (模型架构)
    ├── TrainArgs (训练流程)
    └── DataConfig (数据处理)

各阶段特化:
    Stage1Config ← TrainArgs + seq_len=1024
    Stage2Config ← TrainArgs + seq_len=2048 (扩窗)
    Stage3Config ← TrainArgs + domain-specific data
```

#### 12.6 阶段1预训练配置示例

```yaml
# configs/stage1_pretrain.yaml

model:
  vocab_size: 16000
  dim: 512
  n_layers: 8
  n_heads: 4
  inter_dim: 1536
  moe_inter_dim: 192
  n_routed_experts: 16
  n_shared_experts: 2
  n_activated_experts: 4
  hc_mult: 4
  max_seq_len: 1024
  dtype: "bf16"

training:
  stage: 1
  data_path: "datasets/stage_1_datasets/6_binarize"
  output_dir: "checkpoints/stage1"
  
  max_steps: 50000
  warmup_steps: 2000
  learning_rate: 3e-4
  min_lr: 3e-5
  weight_decay: 0.1
  gradient_clip: 1.0
  
  batch_size: 4
  gradient_accumulation: 8
  seq_len: 1024
  num_workers: 4
  
  log_interval: 10
  eval_interval: 1000
  save_interval: 5000
  
  use_wandb: false
  wandb_project: "deepseek-v4-stage1"
```

### 13. 阶段1预训练执行流程

```
1. 数据准备
   └─> 确认 datasets/stage_1_datasets/6_binarize/*.bin 存在
   └─> 读取 checkpoint.json 确认 tokens 数量

2. 配置加载
   └─> mian.py 解析参数
   └─> 加载 configs/stage1_pretrain.yaml
   └─> 合并为 TrainArgs + ModelArgs

3. 组件初始化
   └─> 创建 Transformer(model_args)
   └─> 创建 TokenDataset(data_path)
   └─> 创建 Muon + AdamW 优化器
   └─> 创建 DataLoader

4. 训练循环
   for step in range(max_steps):
       batch = next(dataloader)           # [B, S]
       logits = model(batch.x)            # [B, V]
       loss = CrossEntropy(logits, batch.y[:, -1])
       loss.backward()
       clip_grad_norm_(model.parameters(), 1.0)
       optimizer.step()
       
       if step % eval_interval == 0:
           val_loss = evaluate(model, val_loader)
           save_checkpoint('best' if val_loss < best)
       
       if step % save_interval == 0:
           save_checkpoint(f'step_{step}')

5. 输出
   └─> checkpoints/stage1/best.pt (用于阶段2)
```

### 14. 检查点格式

```python
{
    'stage': 1,
    'step': 50000,
    'model_state_dict': {...},      # 模型权重
    'muon_state_dict': {...},        # Muon 优化器状态
    'adam_state_dict': {...},         # AdamW 优化器状态
    'best_val_loss': 3.42,
    'current_lr': 3e-5,
    'train_args': {...},              # 序列化的配置
    'model_args': {...},
    'timestamp': '2025-05-16T21:00:00'
}
```