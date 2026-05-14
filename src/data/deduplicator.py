"""文档去重模块 - 多进程并行版"""

import hashlib
import logging
import asyncio
from typing import Iterator, Dict, Set, List, Tuple, Optional
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from pybloom_live import BloomFilter
    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False

from .utils.io_utils import async_read_jsonl_batches, async_write_jsonl_batches


logger = logging.getLogger(__name__)


class BloomDeduplicator:
    """布隆过滤器去重器（支持近似去重）"""

    def __init__(self, expected_items: int = 100000, fp_rate: float = 0.001):
        self.expected_items = expected_items
        self.fp_rate = fp_rate
        self.duplicate_count = 0
        self.unique_count = 0

        if BLOOM_AVAILABLE:
            self.bloom = BloomFilter(capacity=expected_items, error_rate=fp_rate)
            self.seen_hashes: Optional[Set[str]] = set()
        else:
            self.bloom = None
            self.seen_hashes: Set[str] = set()

    def is_duplicate(self, doc_hash: str) -> bool:
        if self.bloom is not None:
            if doc_hash in self.bloom:
                if doc_hash in self.seen_hashes:
                    self.duplicate_count += 1
                    return True
            else:
                self.bloom.add(doc_hash)
                self.seen_hashes.add(doc_hash)
                self.unique_count += 1
                return False
        else:
            if doc_hash in self.seen_hashes:
                self.duplicate_count += 1
                return True
            self.seen_hashes.add(doc_hash)
            self.unique_count += 1
            return False

    def get_stats(self) -> Dict:
        total = self.unique_count + self.duplicate_count
        return {
            'total': total,
            'unique': self.unique_count,
            'duplicates': self.duplicate_count,
            'duplicate_ratio': self.duplicate_count / total if total > 0 else 0.0
        }


class ExactDeduplicator:
    """精确去重器（基于 MD5 哈希）"""

    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0
        self.unique_count = 0

    def deduplicate(self, documents: Iterator[Dict]) -> Iterator[Dict]:
        for doc in documents:
            doc_hash = self._calc_hash(doc)

            if doc_hash in self.seen_hashes:
                self.duplicate_count += 1
                continue

            self.seen_hashes.add(doc_hash)
            self.unique_count += 1
            yield doc

            if (self.unique_count + self.duplicate_count) % 10000 == 0:
                logger.info(f"去重进度: 唯一 {self.unique_count}, 重复 {self.duplicate_count}")

    def _calc_hash(self, doc: Dict) -> str:
        text = doc.get('text', '')
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def get_stats(self) -> Dict:
        total = self.unique_count + self.duplicate_count
        return {
            'total': total,
            'unique': self.unique_count,
            'duplicates': self.duplicate_count,
            'duplicate_ratio': self.duplicate_count / total if total > 0 else 0.0
        }


class NearDeduplicator:
    """近似去重器（简化版 MinHash）"""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.seen_signatures: List[Set] = []

    def deduplicate(self, documents: Iterator[Dict]) -> Iterator[Dict]:
        for doc in documents:
            signature = self._calc_signature(doc)

            if self._is_duplicate(signature):
                continue

            self.seen_signatures.append(signature)
            yield doc

    def _calc_signature(self, doc: Dict) -> Set[str]:
        text = doc.get('text', '')
        words = text.split()
        return set(frozenset(words[i:i+5]) for i in range(len(words)-4))

    def _is_duplicate(self, signature: Set[str]) -> bool:
        for seen_sig in self.seen_signatures[-1000:]:
            jaccard = self._jaccard_similarity(signature, seen_sig)
            if jaccard > self.threshold:
                return True
        return False

    def _jaccard_similarity(self, a: Set, b: Set) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0


def _calc_doc_hash(doc: Dict) -> Tuple[Dict, str]:
    """计算单个文档的 MD5 哈希"""
    text = doc.get('text', '')
    normalized = text.strip().lower()
    doc_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
    return (doc, doc_hash)


def _calc_hashes_batch(docs: List[Dict]) -> List[Tuple[Dict, str]]:
    """批量计算文档哈希（用于多进程）"""
    return [_calc_doc_hash(doc) for doc in docs]


async def _process_file_async(
    input_file: Path,
    output_file: Path,
    dedup: BloomDeduplicator,
    num_workers: int,
    batch_size: int
) -> Dict:
    """异步处理单个文件的去重"""
    read_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    write_queue: asyncio.Queue = asyncio.Queue(maxsize=3)

    async def reader():
        """异步读取批次"""
        async for batch in async_read_jsonl_batches(str(input_file), batch_size):
            await read_queue.put(batch)
        await read_queue.put(None)

    async def processor():
        """多进程处理批次"""
        loop = asyncio.get_event_loop()
        processed = 0
        unique_written = 0

        with Pool(processes=num_workers) as pool:
            while True:
                batch = await read_queue.get()
                if batch is None:
                    break

                results = await loop.run_in_executor(
                    None, _calc_hashes_batch, batch
                )

                unique_batch = []
                for doc, doc_hash in results:
                    if not dedup.is_duplicate(doc_hash):
                        unique_batch.append(doc)
                        unique_written += 1

                if unique_batch:
                    await write_queue.put(unique_batch)

                processed += len(batch)
                if processed % 20000 == 0:
                    logger.info(f"处理进度: {processed} 条, 唯一 {unique_written}")

        await write_queue.put(None)
        return processed, unique_written

    async def writer():
        """异步写入批次"""
        count = 0

        async def batch_gen():
            nonlocal count
            while True:
                batch = await write_queue.get()
                if batch is None:
                    break
                count += len(batch)
                yield batch

        await async_write_jsonl_batches(str(output_file), batch_gen())
        return count

    reader_task = asyncio.create_task(reader())
    processor_task = asyncio.create_task(processor())
    writer_task = asyncio.create_task(writer())

    await asyncio.gather(reader_task, processor_task, writer_task)

    _, unique_written = await processor_task
    final_count = await writer_task

    return {
        'file': input_file.name,
        'unique': final_count
    }


def deduplicate_files(
    input_dir: str,
    output_dir: str,
    use_approx: bool = False,
    num_workers: int = None,
    batch_size: int = 3000,
    bloom_size: int = 500000,
    bloom_fp_rate: float = 0.001
) -> Dict:
    """去重目录下的所有 jsonl 文件（多进程并行版）"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 4)

    dedup = BloomDeduplicator(expected_items=bloom_size, fp_rate=bloom_fp_rate)

    if use_approx:
        near_dedup = NearDeduplicator()

    total_stats = {
        'files_processed': 0,
        'total_docs': 0,
        'unique_docs': 0,
        'duplicates': 0
    }

    input_files = sorted(input_path.glob("*.jsonl"))
    logger.info(f"开始去重: {len(input_files)} 个文件, {num_workers} 个进程")

    async def process_all():
        tasks = []
        for input_file in input_files:
            output_file = output_path / input_file.name
            task = _process_file_async(
                input_file, output_file, dedup, num_workers, batch_size
            )
            tasks.append(task)
        return await asyncio.gather(*tasks)

    try:
        results = asyncio.run(process_all())

        for result in results:
            total_stats['files_processed'] += 1
            total_stats['unique_docs'] += result['unique']
            logger.info(f"去重完成: {result['file']} -> {result['unique']} 条")

    except Exception as e:
        logger.error(f"去重过程出错: {e}")
        raise

    stats = dedup.get_stats()
    total_stats['total_docs'] = stats['total']
    total_stats['duplicates'] = stats['duplicates']

    logger.info(f"去重统计: {total_stats}")
    return total_stats
