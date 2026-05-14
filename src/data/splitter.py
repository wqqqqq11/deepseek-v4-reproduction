"""数据集划分模块 - 按文档级别划分 train/val/test"""

import random
import logging
import asyncio
from typing import Iterator, Dict, List, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count

from .utils.io_utils import (
    read_jsonl, write_jsonl,
    async_read_jsonl_batches, async_write_jsonl_batches
)


logger = logging.getLogger(__name__)


def _load_document_batch(filepath: str) -> List[Dict]:
    """加载单个文件的文档批次"""
    return list(read_jsonl(filepath))


def _assign_split_index(index: int, total: int,
                        split_ratio: Tuple[float, float, float]) -> str:
    """根据索引和比例分配 split"""
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    if index < train_end:
        return 'train'
    elif index < val_end:
        return 'val'
    else:
        return 'test'


class AsyncDocumentSplitter:
    """异步文档划分器（支持流式处理）"""

    def __init__(self,
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 random_seed: int = 42,
                 batch_size: int = 5000):
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.split_names = ['train', 'val', 'test']

    async def split_file_async(self,
                               input_file: Path,
                               output_dir: Path,
                               total_docs: int) -> Dict[str, int]:
        """异步划分单个文件"""
        random.seed(self.random_seed + hash(str(input_file)) % 10000)

        indices = list(range(total_docs))
        random.shuffle(indices)

        split_indices = {
            'train': set(indices[:int(total_docs * self.split_ratio[0])]),
            'val': set(indices[int(total_docs * self.split_ratio[0]):
                              int(total_docs * (self.split_ratio[0] + self.split_ratio[1]))]),
            'test': set(indices[int(total_docs * (self.split_ratio[0] + self.split_ratio[1])):])
        }

        split_buffers = {name: [] for name in self.split_names}
        split_counts = {name: 0 for name in self.split_names}

        doc_index = 0
        async for batch in async_read_jsonl_batches(str(input_file), self.batch_size):
            for doc in batch:
                for split_name in self.split_names:
                    if doc_index in split_indices[split_name]:
                        split_buffers[split_name].append(doc)

                        if len(split_buffers[split_name]) >= self.batch_size:
                            await self._write_batch(output_dir, split_name,
                                                   split_buffers[split_name])
                            split_counts[split_name] += len(split_buffers[split_name])
                            split_buffers[split_name] = []
                        break
                doc_index += 1

        for split_name in self.split_names:
            if split_buffers[split_name]:
                await self._write_batch(output_dir, split_name,
                                       split_buffers[split_name])
                split_counts[split_name] += len(split_buffers[split_name])

        return split_counts

    async def _write_batch(self, output_dir: Path, split_name: str,
                           documents: List[Dict]) -> None:
        """异步写入批次到对应 split"""
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        output_file = split_dir / "part_0000.jsonl"

        async def single_batch():
            yield documents

        await async_write_jsonl_batches(str(output_file), single_batch(), append=True)


class ParallelDocumentSplitter:
    """并行文档划分器（多进程加速）"""

    def __init__(self,
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 random_seed: int = 42,
                 num_workers: int = None):
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.num_workers = num_workers or min(cpu_count(), 4)

    def split_documents_parallel(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """并行划分文档列表"""
        if not documents:
            return {name: [] for name in ['train', 'val', 'test']}

        random.seed(self.random_seed)
        shuffled_indices = list(range(len(documents)))
        random.shuffle(shuffled_indices)

        total = len(documents)
        train_end = int(total * self.split_ratio[0])
        val_end = train_end + int(total * self.split_ratio[1])

        train_indices = set(shuffled_indices[:train_end])
        val_indices = set(shuffled_indices[train_end:val_end])
        test_indices = set(shuffled_indices[val_end:])

        with Pool(processes=self.num_workers) as pool:
            train_docs = pool.map(_get_doc_by_index,
                                 [(documents, i) for i in train_indices])
            val_docs = pool.map(_get_doc_by_index,
                               [(documents, i) for i in val_indices])
            test_docs = pool.map(_get_doc_by_index,
                                [(documents, i) for i in test_indices])

        return {
            'train': [d for d in train_docs if d is not None],
            'val': [d for d in val_docs if d is not None],
            'test': [d for d in test_docs if d is not None]
        }


def _get_doc_by_index(args: Tuple[List[Dict], int]) -> Dict:
    """辅助函数：通过索引获取文档"""
    documents, index = args
    return documents[index] if 0 <= index < len(documents) else None


class DocumentSplitter:
    """基础文档划分器"""

    def __init__(self,
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 random_seed: int = 42):
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.split_names = ['train', 'val', 'test']

    def split_documents(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """将文档列表划分为 train/val/test"""
        if not documents:
            return {name: [] for name in self.split_names}

        random.seed(self.random_seed)
        shuffled = documents.copy()
        random.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * self.split_ratio[0])
        val_end = train_end + int(total * self.split_ratio[1])

        return {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }


def _load_all_documents(input_dir: Path) -> List[Dict]:
    """加载目录下所有 jsonl 文件"""
    documents = []
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        for doc in read_jsonl(str(jsonl_file)):
            documents.append(doc)
    return documents


def _count_documents(input_dir: Path) -> int:
    """统计目录下文档总数"""
    total = 0
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        for _ in read_jsonl(str(jsonl_file)):
            total += 1
    return total


def _write_split_documents(output_dir: Path, split_name: str,
                           documents: List[Dict]) -> int:
    """写入划分后的文档到对应目录"""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    output_file = split_dir / "part_0000.jsonl"
    count = write_jsonl(str(output_file), iter(documents))

    logger.info(f"{split_name}: 写入 {count} 条到 {output_file}")
    return count


def split_language_documents(
    input_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_seed: int = 42,
    use_async: bool = True
) -> Dict:
    """划分单种语言的文档"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始划分: {input_dir}")

    total_docs = _count_documents(input_path)

    if total_docs == 0:
        logger.warning(f"没有文档需要划分: {input_dir}")
        return {'total': 0, 'train': 0, 'val': 0, 'test': 0}

    if use_async and len(list(input_path.glob("*.jsonl"))) == 1:
        return _split_single_file_async(input_path, output_path,
                                       total_docs, split_ratio, random_seed)
    else:
        return _split_multiple_files_sync(input_path, output_path,
                                         split_ratio, random_seed)


def _split_single_file_async(input_path: Path, output_path: Path,
                             total_docs: int,
                             split_ratio: Tuple[float, float, float],
                             random_seed: int) -> Dict:
    """异步划分单个文件"""
    input_file = next(input_path.glob("*.jsonl"))

    splitter = AsyncDocumentSplitter(split_ratio, random_seed)
    split_counts = asyncio.run(splitter.split_file_async(input_file,
                                                         output_path, total_docs))
    split_counts['total'] = total_docs

    logger.info(f"异步划分完成: {split_counts}")
    return split_counts


def _split_multiple_files_sync(input_path: Path, output_path: Path,
                               split_ratio: Tuple[float, float, float],
                               random_seed: int) -> Dict:
    """同步划分多个文件"""
    documents = _load_all_documents(input_path)
    total_docs = len(documents)

    splitter = DocumentSplitter(split_ratio, random_seed)
    splits = splitter.split_documents(documents)

    stats = {'total': total_docs}
    for split_name in ['train', 'val', 'test']:
        count = _write_split_documents(output_path, split_name, splits[split_name])
        stats[split_name] = count

    logger.info(f"划分完成: {stats}")
    return stats


def split_files(
    chinese_input_dir: str,
    english_input_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_seed: int = 42,
    use_async: bool = True
) -> Dict:
    """划分中英文数据集（分别划分后再合并到同一目录结构）"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {'chinese': {}, 'english': {}, 'combined': {}}

    logger.info("开始划分中文数据...")
    chinese_output = output_path / "chinese"
    result['chinese'] = split_language_documents(
        chinese_input_dir, str(chinese_output), split_ratio, random_seed, use_async
    )

    logger.info("开始划分英文数据...")
    english_output = output_path / "english"
    result['english'] = split_language_documents(
        english_input_dir, str(english_output), split_ratio, random_seed, use_async
    )

    result['combined']['total'] = result['chinese']['total'] + result['english']['total']
    for split_name in ['train', 'val', 'test']:
        result['combined'][split_name] = (
            result['chinese'][split_name] + result['english'][split_name]
        )

    logger.info(f"数据集划分完成: {result['combined']}")
    return result
