"""数据合并模块 - 文档级混合切块"""

import random
import logging
from typing import Iterator, Dict, List
from pathlib import Path
from multiprocessing import Pool, cpu_count

from .utils.io_utils import read_jsonl, write_jsonl


logger = logging.getLogger(__name__)


class DocumentMixer:
    """文档级中英文混合器"""

    def __init__(self, chinese_ratio: float = 0.6, english_ratio: float = 0.4):
        self.chinese_ratio = chinese_ratio
        self.english_ratio = english_ratio
        self.ratio_sum = chinese_ratio + english_ratio

    def mix_documents(self,
                     chinese_docs: List[Dict],
                     english_docs: List[Dict]) -> Iterator[Dict]:
        """按比例混合两个文档列表，返回文档流（每轮6:4交错）"""
        chinese_count = len(chinese_docs)
        english_count = len(english_docs)

        chinese_idx = 0
        english_idx = 0

        batch_size = 10
        chinese_per_batch = int(batch_size * self.chinese_ratio / self.ratio_sum)
        english_per_batch = batch_size - chinese_per_batch

        while chinese_idx < chinese_count or english_idx < english_count:
            batch = []

            for _ in range(chinese_per_batch):
                if chinese_idx < chinese_count:
                    batch.append(chinese_docs[chinese_idx])
                    chinese_idx += 1

            for _ in range(english_per_batch):
                if english_idx < english_count:
                    batch.append(english_docs[english_idx])
                    english_idx += 1

            if batch:
                random.shuffle(batch)
                for doc in batch:
                    yield doc


class ChunkBuilder:
    """训练样本切块构建器（输出 input_ids 格式）"""

    def __init__(self, context_size: int = 4096):
        self.context_size = context_size

    def build_chunks(self, documents: Iterator[Dict]) -> Iterator[Dict]:
        """构建 chunks，输出格式: {'input_ids': [tokens]}"""
        buffer = []
        count = 0

        for doc in documents:
            token_ids = doc.get('token_ids', [])
            buffer.extend(token_ids)

            while len(buffer) >= self.context_size:
                chunk_tokens = buffer[:self.context_size]
                yield {'input_ids': chunk_tokens}
                buffer = buffer[self.context_size:]
                count += 1

                if count % 10000 == 0:
                    logger.info(f"已生成 {count} 个 chunk")

        if len(buffer) >= self.context_size // 2:
            chunk_tokens = buffer[:self.context_size]
            chunk_tokens.extend([0] * (self.context_size - len(chunk_tokens)))
            yield {'input_ids': chunk_tokens}
            count += 1

        logger.info(f"共生成 {count} 个 chunk")


def _load_all_docs(directory: str) -> List[Dict]:
    """加载目录下所有 jsonl 文件"""
    docs = []
    path = Path(directory)
    for pattern in ["*.jsonl", "*/*.jsonl"]:
        for jsonl_file in sorted(path.glob(pattern)):
            for doc in read_jsonl(str(jsonl_file)):
                docs.append(doc)
    return docs


def _merge_and_chunk_split(
    chinese_dir: str,
    english_dir: str,
    output_file: str,
    chinese_ratio: float,
    english_ratio: float,
    context_size: int,
    random_seed: int = 42
) -> int:
    """合并单个 split 的中英文数据并切块"""
    logger.info(f"处理: {output_file}")
    random.seed(random_seed)

    chinese_docs = _load_all_docs(chinese_dir)
    english_docs = _load_all_docs(english_dir)

    random.shuffle(chinese_docs)
    random.shuffle(english_docs)

    logger.info(f"中文文档: {len(chinese_docs)}, 英文文档: {len(english_docs)}")

    mixer = DocumentMixer(chinese_ratio, english_ratio)
    mixed_docs = mixer.mix_documents(chinese_docs, english_docs)

    chunker = ChunkBuilder(context_size)
    chunks = chunker.build_chunks(mixed_docs)

    count = write_jsonl(output_file, chunks)
    logger.info(f"完成: {output_file} -> {count} 个 chunk")
    return count


def _process_split_worker(args) -> tuple:
    """多进程 worker 处理单个 split"""
    chinese_dir, english_dir, output_file, chinese_ratio, english_ratio, context_size, random_seed = args
    count = _merge_and_chunk_split(
        chinese_dir, english_dir, output_file,
        chinese_ratio, english_ratio, context_size, random_seed
    )
    return (output_file, count)


def merge_and_chunk(
    chinese_dir: str,
    english_dir: str,
    output_dir: str,
    chinese_ratio: float = 0.6,
    english_ratio: float = 0.4,
    context_size: int = 4096,
    num_workers: int = None,
    random_seed: int = 42,
    use_parallel: bool = True
) -> Dict:
    """合并中英文数据并切块（文档级混合）"""
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
                context_size,
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

            count = _merge_and_chunk_split(
                chinese_split_dir, english_split_dir, output_file,
                chinese_ratio, english_ratio, context_size,
                random_seed + hash(split_name) % 1000
            )
            stats[split_name] = count

    total = sum(stats.values())
    logger.info(f"合并完成: 总计 {total} 个 chunk, {stats}")
    return stats
