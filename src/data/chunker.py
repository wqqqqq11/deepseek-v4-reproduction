"""数据切块模块"""

import random
import logging
from typing import Iterator, Dict, List, Tuple
from pathlib import Path

from .utils.io_utils import read_jsonl, write_jsonl


logger = logging.getLogger(__name__)


class Chunker:
    """将 token 流切分为固定长度的训练样本"""

    def __init__(self,
                 context_size: int = 4096,
                 chunk_stride: int = 4097,
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        self.context_size = context_size
        self.chunk_stride = chunk_stride
        self.split_ratio = split_ratio

    def create_chunks(self,
                      chinese_docs: Iterator[Dict],
                      english_docs: Iterator[Dict],
                      chinese_ratio: float = 0.6,
                      english_ratio: float = 0.4) -> Dict[str, Iterator[Tuple[List[int], List[int]]]]:
        """创建训练样本切块"""
        chinese_stream = self._doc_stream_to_token_stream(chinese_docs)
        english_stream = self._doc_stream_to_token_stream(english_docs)

        mixed_stream = self._mix_streams(
            chinese_stream, english_stream,
            chinese_ratio, english_ratio
        )

        chunks = self._build_chunks(mixed_stream)
        return self._split_chunks(chunks)

    def _doc_stream_to_token_stream(self, docs: Iterator[Dict]) -> Iterator[int]:
        """将文档流转换为 token 流"""
        for doc in docs:
            token_ids = doc.get('token_ids', [])
            for tid in token_ids:
                yield tid

    def _mix_streams(self,
                     chinese_stream: Iterator[int],
                     english_stream: Iterator[int],
                     chinese_ratio: float,
                     english_ratio: float) -> Iterator[int]:
        """按比例混合两个 token 流"""
        chinese_buffer = []
        english_buffer = []

        chunk_size = 10000
        ratio_sum = chinese_ratio + english_ratio

        while True:
            try:
                for _ in range(int(chunk_size * chinese_ratio / ratio_sum)):
                    chinese_buffer.append(next(chinese_stream))
            except StopIteration:
                pass

            try:
                for _ in range(int(chunk_size * english_ratio / ratio_sum)):
                    english_buffer.append(next(english_stream))
            except StopIteration:
                pass

            if not chinese_buffer and not english_buffer:
                break

            combined = chinese_buffer + english_buffer
            random.shuffle(combined)

            for token in combined:
                yield token

            chinese_buffer = []
            english_buffer = []

    def _build_chunks(self, token_stream: Iterator[int]) -> Iterator[Tuple[List[int], List[int]]]:
        """构建 (x, y) 切块"""
        buffer = []
        count = 0

        for token in token_stream:
            buffer.append(token)

            if len(buffer) >= self.chunk_stride:
                x = buffer[:self.context_size]
                y = buffer[1:self.context_size + 1]
                yield (x, y)
                buffer = buffer[self.context_size:]
                count += 1

                if count % 10000 == 0:
                    logger.info(f"已生成 {count} 个 chunk")

        logger.info(f"共生成 {count} 个 chunk")

    def _split_chunks(self,
                      chunks: Iterator[Tuple[List[int], List[int]]]
                      ) -> Dict[str, Iterator[Tuple[List[int], List[int]]]]:
        """按比例切分为 train/val/test"""
        train_buffer = []
        val_buffer = []
        test_buffer = []

        for chunk in chunks:
            r = random.random()
            if r < self.split_ratio[0]:
                train_buffer.append(chunk)
            elif r < self.split_ratio[0] + self.split_ratio[1]:
                val_buffer.append(chunk)
            else:
                test_buffer.append(chunk)

            if len(train_buffer) >= 1000:
                for c in train_buffer:
                    yield c
                train_buffer = []

        def _yield_buffered(buffer):
            for c in buffer:
                yield c

        return {
            'train': _yield_buffered(train_buffer),
            'val': _yield_buffered(val_buffer),
            'test': _yield_buffered(test_buffer)
        }


def process_and_chunk(chinese_dir: str,
                      english_dir: str,
                      output_dir: str,
                      chinese_ratio: float = 0.6,
                      english_ratio: float = 0.4,
                      context_size: int = 4096) -> Dict:
    """处理并切块数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunker = Chunker(context_size=context_size)

    chinese_docs = _load_all_docs(chinese_dir)
    english_docs = _load_all_docs(english_dir)

    splits = chunker.create_chunks(
        chinese_docs, english_docs,
        chinese_ratio, english_ratio
    )

    stats = {}
    for split_name, chunks in splits.items():
        output_file = output_path / f"{split_name}.jsonl"
        count = _write_chunks(str(output_file), chunks)
        stats[split_name] = count
        logger.info(f"{split_name}: {count} 个 chunk")

    return stats


def _load_all_docs(directory: str) -> Iterator[Dict]:
    """加载目录下所有 jsonl 文件"""
    path = Path(directory)
    for jsonl_file in sorted(path.glob("*.jsonl")):
        yield from read_jsonl(str(jsonl_file))


def _write_chunks(filepath: str,
                  chunks: Iterator[Tuple[List[int], List[int]]]) -> int:
    """写入 chunk 文件"""
    records = ({'x': x, 'y': y} for x, y in chunks)
    return write_jsonl(filepath, records)
