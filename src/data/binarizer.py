"""数据二进制化模块 - 将 jsonl 转为连续 token 流 bin 文件"""

import logging
import numpy as np
from typing import Iterator, List
from pathlib import Path
from multiprocessing import Pool, cpu_count

from .utils.io_utils import read_jsonl


logger = logging.getLogger(__name__)


def _extract_tokens(doc: dict) -> List[int]:
    """从文档中提取 token ids"""
    return doc.get('input_ids', [])


def _jsonl_to_token_stream(filepath: str) -> Iterator[int]:
    """从 jsonl 文件生成 token 流"""
    for doc in read_jsonl(filepath):
        tokens = _extract_tokens(doc)
        for token in tokens:
            yield token


def _write_tokens_to_bin(input_file: str, output_file: str, dtype=np.uint16) -> int:
    """将 token 流写入二进制文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 清空或创建文件
    output_path.write_bytes(b'')

    token_buffer = []
    count = 0
    chunk_size = 100000

    def flush_buffer():
        if not token_buffer:
            return
        arr = np.array(token_buffer, dtype=dtype)
        with open(output_path, 'ab') as f:
            f.write(arr.tobytes())

    for token in _jsonl_to_token_stream(input_file):
        token_buffer.append(token)
        count += 1

        if len(token_buffer) >= chunk_size:
            flush_buffer()
            token_buffer = []

    flush_buffer()

    logger.info(f"写入完成: {output_file}, 共 {count} 个 tokens")
    return count


def _binarize_split_worker(args) -> tuple:
    """多进程 worker 处理单个 split"""
    input_file, output_file, dtype = args
    count = _write_tokens_to_bin(input_file, output_file, dtype)
    return (output_file, count)


def binarize_files(
    input_dir: str,
    output_dir: str,
    splits: List[str] = None,
    dtype=np.uint16,
    num_workers: int = None,
    use_parallel: bool = True
) -> dict:
    """将 jsonl 文件转为二进制 token 流"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ['train', 'val', 'test']

    if num_workers is None:
        num_workers = min(cpu_count(), len(splits))

    stats = {}

    if use_parallel and num_workers > 1:
        tasks = []
        for split_name in splits:
            input_file = str(input_path / f"{split_name}.jsonl")
            output_file = str(output_path / f"{split_name}.bin")
            tasks.append((input_file, output_file, dtype))

        with Pool(processes=num_workers) as pool:
            results = pool.map(_binarize_split_worker, tasks)

        for output_file, count in results:
            split_name = Path(output_file).stem
            stats[split_name] = count
    else:
        for split_name in splits:
            input_file = str(input_path / f"{split_name}.jsonl")
            output_file = str(output_path / f"{split_name}.bin")

            count = _write_tokens_to_bin(input_file, output_file, dtype)
            stats[split_name] = count

    total = sum(stats.values())
    logger.info(f"二进制化完成: 总计 {total} 个 tokens, {stats}")
    return stats
