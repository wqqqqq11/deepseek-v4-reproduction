"""数据合并模块 - 文档级混合"""

import random
import logging
from typing import Iterator, Dict, List
from pathlib import Path
from multiprocessing import Pool, cpu_count

from .utils.io_utils import read_jsonl, write_jsonl


logger = logging.getLogger(__name__)


class DocumentShuffler:
    """文档级中英文混合器"""

    def __init__(self, chinese_ratio: float = 0.6, english_ratio: float = 0.4):
        self.chinese_ratio = chinese_ratio
        self.english_ratio = english_ratio
        self.ratio_sum = chinese_ratio + english_ratio

    def shuffle_and_merge(self,
                         chinese_docs: List[Dict],
                         english_docs: List[Dict]) -> Iterator[Dict]:
        """分别shuffle后按6:4数量比交错合并"""
        random.shuffle(chinese_docs)
        random.shuffle(english_docs)

        chinese_per_round = int(10 * self.chinese_ratio / self.ratio_sum)
        english_per_round = 10 - chinese_per_round

        chinese_idx = 0
        english_idx = 0
        chinese_count = len(chinese_docs)
        english_count = len(english_docs)

        while chinese_idx < chinese_count or english_idx < english_count:
            round_docs = []

            for _ in range(chinese_per_round):
                if chinese_idx < chinese_count:
                    round_docs.append(chinese_docs[chinese_idx])
                    chinese_idx += 1

            for _ in range(english_per_round):
                if english_idx < english_count:
                    round_docs.append(english_docs[english_idx])
                    english_idx += 1

            if round_docs:
                random.shuffle(round_docs)
                for doc in round_docs:
                    yield doc


def _load_all_docs(directory: str) -> List[Dict]:
    """加载目录下所有jsonl文件"""
    docs = []
    path = Path(directory)
    for pattern in ["*.jsonl", "*/*.jsonl"]:
        for jsonl_file in sorted(path.glob(pattern)):
            for doc in read_jsonl(str(jsonl_file)):
                docs.append(doc)
    return docs


def _docs_to_records(docs: Iterator[Dict]) -> Iterator[Dict]:
    """将文档转换为只含input_ids的记录"""
    for doc in docs:
        yield {'input_ids': doc.get('token_ids', [])}


def _merge_split(
    chinese_dir: str,
    english_dir: str,
    output_file: str,
    chinese_ratio: float,
    english_ratio: float,
    random_seed: int = 42
) -> int:
    """合并单个split的中英文文档"""
    logger.info(f"处理: {output_file}")
    random.seed(random_seed)

    chinese_docs = _load_all_docs(chinese_dir)
    english_docs = _load_all_docs(english_dir)

    logger.info(f"中文文档: {len(chinese_docs)}, 英文文档: {len(english_docs)}")

    shuffler = DocumentShuffler(chinese_ratio, english_ratio)
    mixed_docs = shuffler.shuffle_and_merge(chinese_docs, english_docs)

    records = _docs_to_records(mixed_docs)
    count = write_jsonl(output_file, records)

    logger.info(f"完成: {output_file} -> {count} 条")
    return count


def _process_split_worker(args) -> tuple:
    """多进程worker处理单个split"""
    chinese_dir, english_dir, output_file, chinese_ratio, english_ratio, random_seed = args
    count = _merge_split(
        chinese_dir, english_dir, output_file,
        chinese_ratio, english_ratio, random_seed
    )
    return (output_file, count)


def merge_documents(
    chinese_dir: str,
    english_dir: str,
    output_dir: str,
    chinese_ratio: float = 0.6,
    english_ratio: float = 0.4,
    num_workers: int = None,
    random_seed: int = 42,
    use_parallel: bool = True
) -> Dict:
    """合并中英文文档（文档级shuffle，不切块）"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 3)

    splits = ['train', 'val', 'test']
    stats = {}

    if use_parallel and num_workers > 1:
        tasks = []
        for split_name in splits:
            task = (
                str(Path(chinese_dir) / split_name),
                str(Path(english_dir) / split_name),
                str(output_path / f"{split_name}.jsonl"),
                chinese_ratio,
                english_ratio,
                random_seed + hash(split_name) % 1000
            )
            tasks.append(task)

        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_split_worker, tasks)

        for output_file, count in results:
            split_name = Path(output_file).stem
            stats[split_name] = count
    else:
        for split_name in splits:
            chinese_split_dir = str(Path(chinese_dir) / split_name)
            english_split_dir = str(Path(english_dir) / split_name)
            output_file = str(output_path / f"{split_name}.jsonl")

            count = _merge_split(
                chinese_split_dir, english_split_dir, output_file,
                chinese_ratio, english_ratio,
                random_seed + hash(split_name) % 1000
            )
            stats[split_name] = count

    total = sum(stats.values())
    logger.info(f"合并完成: 总计 {total} 条, {stats}")
    return stats
