"""文档去重模块"""

import hashlib
import logging
from typing import Iterator, Dict, Set
from pathlib import Path

from .utils.io_utils import read_jsonl, write_jsonl


logger = logging.getLogger(__name__)


class ExactDeduplicator:
    """精确去重器（基于 MD5 哈希）"""

    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0
        self.unique_count = 0

    def deduplicate(self, documents: Iterator[Dict]) -> Iterator[Dict]:
        """去重文档流"""
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
        """计算文档哈希"""
        text = doc.get('text', '')
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def get_stats(self) -> Dict:
        """返回去重统计"""
        total = self.unique_count + self.duplicate_count
        return {
            'total': total,
            'unique': self.unique_count,
            'duplicates': self.duplicate_count,
            'duplicate_ratio': self.duplicate_count / total if total > 0 else 0
        }


class NearDeduplicator:
    """近似去重器（简化版 MinHash）"""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.seen_signatures: list = []

    def deduplicate(self, documents: Iterator[Dict]) -> Iterator[Dict]:
        """近似去重文档流"""
        for doc in documents:
            signature = self._calc_signature(doc)

            if self._is_duplicate(signature):
                continue

            self.seen_signatures.append(signature)
            yield doc

    def _calc_signature(self, doc: Dict) -> Set[str]:
        """计算文档的 n-gram 签名"""
        text = doc.get('text', '')
        words = text.split()
        return set(frozenset(words[i:i+5]) for i in range(len(words)-4))

    def _is_duplicate(self, signature: Set[str]) -> bool:
        """检查是否为近似重复"""
        for seen_sig in self.seen_signatures[-1000:]:
            jaccard = self._jaccard_similarity(signature, seen_sig)
            if jaccard > self.threshold:
                return True
        return False

    def _jaccard_similarity(self, a: Set, b: Set) -> float:
        """计算 Jaccard 相似度"""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0


def deduplicate_files(input_dir: str,
                      output_dir: str,
                      use_approx: bool = False) -> Dict:
    """去重目录下的所有 jsonl 文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    deduplicator = ExactDeduplicator()
    if use_approx:
        near_dedup = NearDeduplicator()

    total_stats = {
        'files_processed': 0,
        'total_docs': 0,
        'unique_docs': 0,
        'duplicates': 0
    }

    for input_file in sorted(input_path.glob("*.jsonl")):
        try:
            documents = read_jsonl(str(input_file))

            if use_approx:
                documents = near_dedup.deduplicate(documents)

            documents = deduplicator.deduplicate(documents)

            output_file = output_path / input_file.name
            count = write_jsonl(str(output_file), documents)

            total_stats['files_processed'] += 1
            total_stats['unique_docs'] += count

            logger.info(f"去重完成: {input_file.name} -> {count} 条")
        except Exception as e:
            logger.error(f"去重失败 {input_file.name}: {e}")

    stats = deduplicator.get_stats()
    total_stats['total_docs'] = stats['total']
    total_stats['duplicates'] = stats['duplicates']

    logger.info(f"去重统计: {total_stats}")
    return total_stats
