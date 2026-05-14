"""Tokenizer 模块"""

import logging
import asyncio
from typing import Iterator, Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context, cpu_count
from functools import partial

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

from .utils.io_utils import async_read_jsonl_batches, async_write_jsonl_batches


logger = logging.getLogger(__name__)


class TokenizerProcessor:
    """BPE Tokenizer 处理器"""

    def __init__(self,
                 tokenizer_name: str = "gpt2",
                 eos_token_id: int = 2,
                 vocab_size: int = 16000):
        self.tokenizer_name = tokenizer_name
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """加载 tokenizer"""
        if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
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

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """批量编码"""
        return [self.encode(text) for text in texts]


def _init_worker(tokenizer_name: str, eos_token_id: int):
    """多进程 worker 初始化"""
    global _worker_processor
    _worker_processor = TokenizerProcessor(
        tokenizer_name=tokenizer_name,
        eos_token_id=eos_token_id
    )


def _process_batch_worker(batch: List[Dict]) -> List[Dict]:
    """多进程处理批次"""
    global _worker_processor
    results = []

    for doc in batch:
        text = doc.get('text', '')
        token_ids = _worker_processor.encode(text)

        if token_ids:
            results.append({
                'token_ids': token_ids,
                'length': len(token_ids),
                'source': doc.get('source', 'unknown'),
                'lang': doc.get('lang', 'unknown')
            })

    return results


async def _tokenize_file_async(
    input_file: Path,
    output_file: Path,
    tokenizer_name: str,
    eos_token_id: int,
    num_workers: int,
    batch_size: int
) -> Dict:
    """异步并行 tokenize 单个文件"""
    read_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    write_queue: asyncio.Queue = asyncio.Queue(maxsize=3)

    total_in = 0
    total_out = 0
    total_tokens = 0

    mp_context = get_context("spawn")

    async def reader():
        """异步读取批次"""
        nonlocal total_in
        async for batch in async_read_jsonl_batches(str(input_file), batch_size):
            await read_queue.put(batch)
            total_in += len(batch)
        await read_queue.put(None)

    async def processor():
        """并行处理批次"""
        nonlocal total_out, total_tokens
        loop = asyncio.get_event_loop()

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(tokenizer_name, eos_token_id)
        ) as executor:
            while True:
                batch = await read_queue.get()
                if batch is None:
                    break

                results = await loop.run_in_executor(
                    executor, _process_batch_worker, batch
                )

                if results:
                    await write_queue.put(results)
                    total_out += len(results)
                    total_tokens += sum(r['length'] for r in results)

                if total_in % 20000 == 0:
                    logger.info(f"Tokenize 进度: 处理 {total_in}, 成功 {total_out}")

        await write_queue.put(None)

    async def writer():
        """异步写入批次"""
        async def batch_gen():
            while True:
                batch = await write_queue.get()
                if batch is None:
                    break
                yield batch

        await async_write_jsonl_batches(str(output_file), batch_gen())

    await asyncio.gather(
        asyncio.create_task(reader()),
        asyncio.create_task(processor()),
        asyncio.create_task(writer())
    )

    return {
        'input_count': total_in,
        'output_count': total_out,
        'total_tokens': total_tokens
    }


def _tokenize_file_sync(
    input_file: Path,
    output_file: Path,
    processor: TokenizerProcessor
) -> Dict:
    """同步 tokenize 单个文件（fallback）"""
    from .utils.io_utils import read_jsonl, write_jsonl

    documents = read_jsonl(str(input_file))
    tokenized = []

    total_in = 0
    total_out = 0
    total_tokens = 0

    for doc in documents:
        total_in += 1
        text = doc.get('text', '')
        token_ids = processor.encode(text)

        if token_ids:
            tokenized.append({
                'token_ids': token_ids,
                'length': len(token_ids),
                'source': doc.get('source', 'unknown'),
                'lang': doc.get('lang', 'unknown')
            })
            total_out += 1
            total_tokens += len(token_ids)

        if total_in % 10000 == 0:
            logger.info(f"Tokenize 进度: {total_in}, 成功 {total_out}")

    write_jsonl(str(output_file), iter(tokenized))

    return {
        'input_count': total_in,
        'output_count': total_out,
        'total_tokens': total_tokens
    }


def _get_all_jsonl_files(input_dir: Path) -> List[Path]:
    """获取所有 jsonl 文件路径"""
    files = []
    for pattern in ["*.jsonl", "*/*.jsonl", "*/*/*.jsonl"]:
        files.extend(input_dir.glob(pattern))
    return sorted(files)


def _get_relative_output_path(input_file: Path, input_base: Path) -> Path:
    """计算输出文件的相对路径"""
    try:
        rel_path = input_file.relative_to(input_base)
        return rel_path
    except ValueError:
        return input_file.name


def tokenize_files(
    input_dir: str,
    output_dir: str,
    tokenizer_name: str = "gpt2",
    eos_token_id: int = 2,
    num_workers: int = None,
    batch_size: int = 3000,
    use_async: bool = True
) -> Dict:
    """Tokenize 目录下的所有 jsonl 文件（异步并行版）"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 4)

    input_files = _get_all_jsonl_files(input_path)
    if not input_files:
        logger.warning(f"没有找到输入文件: {input_dir}")
        return {'files_processed': 0, 'total_docs': 0, 'total_tokens': 0}

    logger.info(f"开始 Tokenize: {len(input_files)} 个文件, {num_workers} 个进程")

    processor = TokenizerProcessor(
        tokenizer_name=tokenizer_name,
        eos_token_id=eos_token_id
    )

    total_stats = {
        'files_processed': 0,
        'total_docs': 0,
        'total_tokens': 0
    }

    async def process_all():
        tasks = []
        for input_file in input_files:
            rel_path = _get_relative_output_path(input_file, input_path)
            output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if use_async:
                task = _tokenize_file_async(
                    input_file, output_file,
                    tokenizer_name, eos_token_id,
                    num_workers, batch_size
                )
            else:
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(
                    None, _tokenize_file_sync,
                    input_file, output_file, processor
                )
            tasks.append((str(rel_path), task))

        results = await asyncio.gather(*[t[1] for t in tasks])
        return dict((t[0], results[i]) for i, t in enumerate(tasks))

    try:
        file_results = asyncio.run(process_all())

        for filename, stats in file_results.items():
            total_stats['files_processed'] += 1
            total_stats['total_docs'] += stats['output_count']
            total_stats['total_tokens'] += stats['total_tokens']
            logger.info(f"Tokenize 完成: {filename} -> {stats['output_count']} 条, "
                       f"{stats['total_tokens']} tokens")

    except Exception as e:
        logger.error(f"Tokenize 过程出错: {e}")
        raise

    logger.info(f"Tokenize 总计: {total_stats}")
    return total_stats
