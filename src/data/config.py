"""数据预处理配置管理"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """数据预处理全流程配置"""

    base_data_dir: str = "datasets/stage_1_datasets"
    chinese_dataset: str = "opencsg/Fineweb-Edu-Chinese-V2.1"
    english_dataset: str = "togethercomputer/RedPajama-Data-V2"

    chinese_ratio: float = 0.6
    english_ratio: float = 0.4
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    context_size: int = 4096
    chunk_stride: int = 4097

    min_text_length: int = 10
    max_text_length: int = 100000
    min_chinese_ratio: float = 0.3
    min_english_ratio: float = 0.5

    vocab_size: int = 32000
    eos_token_id: int = 2
    tokenizer_name: str = "XiaoduoAILab/Xmodel_LM"

    total_target_tokens: int = 120_000_000
    chinese_target_tokens: int = 72_000_000
    english_target_tokens: int = 48_000_000

    log_interval: int = 10000
    batch_size: int = 5000
    num_workers: int = 4
    io_queue_size: int = 3
    random_seed: int = 42

    def __post_init__(self):
        self._validate_ratios()
        self._ensure_dirs()

    def _validate_ratios(self):
        total = self.chinese_ratio + self.english_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"语言比例之和必须等于1，当前为{total}")

        split_total = sum(self.split_ratio)
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError(f"数据切分比例之和必须等于1，当前为{split_total}")

    def _ensure_dirs(self):
        base = Path(self.base_data_dir)
        for stage in ["0_raw", "1_cleaned", "2_deduplicated", "3_split",
                      "4_tokenized", "5_merged", "6_binarize"]:
            (base / stage).mkdir(parents=True, exist_ok=True)

        (base / "1_cleaned" / "chinese").mkdir(parents=True, exist_ok=True)
        (base / "1_cleaned" / "english").mkdir(parents=True, exist_ok=True)
        (base / "2_deduplicated" / "chinese").mkdir(parents=True, exist_ok=True)
        (base / "2_deduplicated" / "english").mkdir(parents=True, exist_ok=True)
        (base / "3_split" / "chinese").mkdir(parents=True, exist_ok=True)
        (base / "3_split" / "english").mkdir(parents=True, exist_ok=True)
        (base / "4_tokenized").mkdir(parents=True, exist_ok=True)

    def get_stage_dir(self, stage: int, lang: Optional[str] = None) -> Path:
        stage_names = {
            0: "0_raw",
            1: "1_cleaned",
            2: "2_deduplicated",
            3: "3_split",
            4: "4_tokenized",
            5: "5_merged",
            6: "6_binarize"
        }
        if stage not in stage_names:
            raise ValueError(f"无效的阶段编号: {stage}")

        path = Path(self.base_data_dir) / stage_names[stage]
        if lang:
            path = path / lang
        return path

    def get_checkpoint_path(self) -> Path:
        return Path(self.base_data_dir) / "checkpoint.json"
