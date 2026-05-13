"""Tokenizer 模块"""

import logging
from typing import Iterator, Dict, List
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from .utils.io_utils import read_jsonl, write_jsonl


logger = logging.getLogger(__name__)


class TokenizerProcessor:
    """BPE Tokenizer 处理器"""

    def __init__(self,
                 tokenizer_name: str = "gpt2",
                 eos_token_id: int = 2,
                 vocab_size: int = 32000):
        self.tokenizer_name = tokenizer_name
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """加载 tokenizer"""
        if AutoTokenizer is None:
            logger.warning("transformers 未安装，使用模拟 tokenizer")
            self.tokenizer = None
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True
            )
            if self.tokenizer.eos_token_id is None:
                self.tokenizer.eos_token_id = self.eos_token_id
            logger.info(f"加载 tokenizer: {self.tokenizer_name}")
        except Exception as e:
            logger.error(f"加载 tokenizer 失败: {e}")
            self.tokenizer = None

    def encode(self, text: str) -> List[int]:
        """编码文本为 token ids"""
        if not text:
            return []

        if self.tokenizer is None:
            return self._mock_encode(text)

        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.tokenizer.eos_token_id or self.eos_token_id)
            return ids
        except Exception as e:
            logger.warning(f"编码失败: {e}")
            return []

    def _mock_encode(self, text: str) -> List[int]:
        """模拟编码（用于测试）"""
        return [ord(c) % self.vocab_size for c in text[:1000]] + [self.eos_token_id]

    def process_documents(self, documents: Iterator[Dict]) -> Iterator[Dict]:
        """处理文档流"""
        processed = 0
        failed = 0

        for doc in documents:
            text = doc.get('text', '')
            token_ids = self.encode(text)

            if not token_ids:
                failed += 1
                continue

            processed += 1
            yield {
                'token_ids': token_ids,
                'length': len(token_ids),
                'source': doc.get('source', 'unknown'),
                'lang': doc.get('lang', 'unknown')
            }

            if (processed + failed) % 10000 == 0:
                logger.info(f"Tokenize: 成功 {processed}, 失败 {failed}")

        logger.info(f"Tokenize 完成: 成功 {processed}, 失败 {failed}")


def tokenize_files(input_dir: str,
                   output_dir: str,
                   tokenizer_name: str = "gpt2",
                   eos_token_id: int = 2) -> Dict:
    """Tokenize 目录下的所有 jsonl 文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor = TokenizerProcessor(
        tokenizer_name=tokenizer_name,
        eos_token_id=eos_token_id
    )

    stats = {
        'files_processed': 0,
        'total_docs': 0,
        'total_tokens': 0
    }

    for input_file in sorted(input_path.glob("*.jsonl")):
        try:
            documents = read_jsonl(str(input_file))
            tokenized = processor.process_documents(documents)

            output_file = output_path / input_file.name
            count = write_jsonl(str(output_file), tokenized)

            stats['files_processed'] += 1
            stats['total_docs'] += count

            logger.info(f"Tokenize 完成: {input_file.name} -> {count} 条")
        except Exception as e:
            logger.error(f"Tokenize 失败 {input_file.name}: {e}")

    return stats
