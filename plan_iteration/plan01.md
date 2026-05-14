# 阶段 1：通用基座大规模预训练
    定位：打底通用语言、百科、常识、基础文理知识
    数据总量：120M Tokens
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

9. 切块
   每 4097 个 token 切一块
   x = 前 4096
   y = 后 4096

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
