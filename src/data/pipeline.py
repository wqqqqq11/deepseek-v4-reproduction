"""数据预处理流水线"""

import json
import time
import asyncio
import logging
from typing import Dict, Optional, List
from pathlib import Path
from multiprocessing import cpu_count

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    load_dataset = None
    AutoTokenizer = None

from .config import DataConfig
from .cleaners import CleaningChain, ParallelCleaner
from .deduplicator import deduplicate_files
from .tokenizer import tokenize_files
from .chunker import process_and_chunk
from .utils.io_utils import (
    load_checkpoint, save_checkpoint, write_jsonl,
    async_read_jsonl_batches, async_write_jsonl_batches
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """数据预处理流水线"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.checkpoint = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict:
        """加载检查点"""
        cp = load_checkpoint(self.config.base_data_dir)
        return cp or {'completed_stages': []}

    def _save_checkpoint(self, stage: int, data: Dict) -> None:
        """保存检查点"""
        self.checkpoint['completed_stages'].append(stage)
        self.checkpoint[f'stage_{stage}'] = data
        save_checkpoint(self.config.base_data_dir, self.checkpoint)

    def _is_stage_completed(self, stage: int) -> bool:
        """检查阶段是否已完成"""
        return stage in self.checkpoint.get('completed_stages', [])

    def run_stage(self, stage_num: int, resume: bool = True) -> Dict:
        """运行指定阶段"""
        if resume and self._is_stage_completed(stage_num):
            logger.info(f"阶段 {stage_num} 已完成，跳过")
            return self.checkpoint.get(f'stage_{stage_num}', {})

        logger.info(f"开始运行阶段 {stage_num}")
        start_time = time.time()

        handlers = {
            0: self._stage_download,
            1: self._stage_clean,
            2: self._stage_deduplicate,
            3: self._stage_tokenize,
            4: self._stage_chunk,
            5: self._stage_binarize,
        }

        if stage_num not in handlers:
            raise ValueError(f"无效的阶段编号: {stage_num}")

        try:
            result = handlers[stage_num]()
            elapsed = time.time() - start_time
            result['elapsed_time'] = elapsed

            self._save_checkpoint(stage_num, result)
            logger.info(f"阶段 {stage_num} 完成，耗时 {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"阶段 {stage_num} 失败: {e}")
            raise

    def run_all(self, start_stage: int = 0) -> Dict:
        """运行所有阶段"""
        results = {}
        for stage in range(start_stage, 6):
            results[stage] = self.run_stage(stage, resume=True)
        return results

    def _stage_download(self) -> Dict:
        """阶段 0: 下载数据"""
        if load_dataset is None:
            logger.warning("datasets 库未安装，跳过下载")
            return {'skipped': True}

        result = {
            'chinese': {'docs': 0, 'tokens': 0},
            'english': {'docs': 0, 'tokens': 0}
        }

        try:
            ds_zh = load_dataset(
                self.config.chinese_dataset,
                streaming=True,
                split='train'
            )
            output_zh = self.config.get_stage_dir(0, 'chinese')
            zh_docs, zh_tokens = self._save_stream(
                ds_zh, str(output_zh), 'chinese',
                self.config.chinese_target_tokens
            )
            result['chinese'] = {'docs': zh_docs, 'tokens': zh_tokens}
        except Exception as e:
            logger.error(f"下载中文数据失败: {e}")

        try:
            ds_en = load_dataset(
                self.config.english_dataset,
                name='sample-10B',
                streaming=True,
                split='train'
            )
            output_en = self.config.get_stage_dir(0, 'english')
            en_docs, en_tokens = self._save_stream(
                ds_en, str(output_en), 'english',
                self.config.english_target_tokens
            )
            result['english'] = {'docs': en_docs, 'tokens': en_tokens}
        except Exception as e:
            logger.error(f"下载英文数据失败: {e}")

        return result

    def _save_stream(self, dataset, output_dir: str, lang: str,
                     target_tokens: int) -> tuple:
        """保存流式数据集，达到目标 token 数后停止"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tokenizer = None
        if AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name
                )
            except Exception as e:
                logger.warning(f"加载 tokenizer 失败: {e}")

        token_count = 0
        doc_count = 0
        skipped_count = 0

        def get_text(item):
            if 'text' in item:
                return item['text']
            if 'raw_content' in item:
                return item['raw_content']
            return ''

        def get_source(item):
            if 'source' in item:
                return item['source']
            if 'meta' in item:
                try:
                    meta = json.loads(item['meta'])
                    return meta.get('url', '')
                except (json.JSONDecodeError, TypeError):
                    pass
            return lang

        def get_language(item):
            if 'meta' in item:
                try:
                    meta = json.loads(item['meta'])
                    return meta.get('language', '')
                except (json.JSONDecodeError, TypeError):
                    pass
            return ''

        def doc_generator():
            nonlocal token_count, doc_count, skipped_count

            for item in dataset:
                text = get_text(item)
                if not text:
                    continue

                if lang == 'english':
                    item_lang = get_language(item)
                    if item_lang != 'en':
                        skipped_count += 1
                        continue

                if tokenizer is not None:
                    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                else:
                    n_tokens = len(text) // 4

                if token_count + n_tokens > target_tokens:
                    logger.info(f"达到目标: {token_count}/{target_tokens}")
                    break

                token_count += n_tokens
                doc_count += 1

                yield {
                    'text': text,
                    'source': get_source(item)
                }

                if doc_count % 1000 == 0:
                    logger.info(f"{lang}: {doc_count} 篇, {token_count} tokens, 跳过 {skipped_count} 篇")

        output_file = Path(output_dir) / 'part_0000.jsonl'
        write_jsonl(str(output_file), doc_generator())

        logger.info(f"{lang} 完成: {doc_count} 篇, {token_count} tokens, 跳过非目标语言 {skipped_count} 篇")
        return doc_count, token_count

    def _stage_clean(self) -> Dict:
        """阶段 1: 清洗数据 (异步并行版本)"""
        result = {'chinese': 0, 'english': 0}

        for lang in ['chinese', 'english']:
            stats = self._clean_language_async(lang)
            result[lang] = stats

        return result

    def _clean_language_async(self, lang: str) -> int:
        """异步清洗指定语言的数据"""
        input_dir = self.config.get_stage_dir(0, lang)
        output_dir = self.config.get_stage_dir(1, lang)

        input_files = sorted(input_dir.glob("*.jsonl"))
        if not input_files:
            logger.warning(f"{lang}: 没有找到输入文件")
            return 0

        total_count = 0
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers

        with ParallelCleaner(self.config.__dict__, num_workers) as cleaner:
            for input_file in input_files:
                output_file = output_dir / input_file.name
                count = asyncio.run(self._process_file_async(
                    input_file, output_file, lang, cleaner, batch_size
                ))
                total_count += count
                logger.info(f"{lang}: {input_file.name} 处理完成，共 {count} 条")

        return total_count

    async def _process_file_async(
        self,
        input_file: Path,
        output_file: Path,
        lang: str,
        cleaner: ParallelCleaner,
        batch_size: int
    ) -> int:
        """异步处理单个文件"""
        read_queue = asyncio.Queue()
        write_queue = asyncio.Queue()

        reader_task = asyncio.create_task(
            self._read_batches(input_file, batch_size, read_queue)
        )

        processor_task = asyncio.create_task(
            self._process_batches(read_queue, write_queue, lang, cleaner)
        )

        writer_task = asyncio.create_task(
            self._write_batches(write_queue, output_file)
        )

        await asyncio.gather(reader_task, processor_task, writer_task)
        return writer_task.result()

    async def _read_batches(
        self,
        input_file: Path,
        batch_size: int,
        queue: asyncio.Queue
    ):
        """异步读取批次"""
        async for batch in async_read_jsonl_batches(str(input_file), batch_size):
            await queue.put(batch)
        await queue.put(None)

    async def _process_batches(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        lang: str,
        cleaner: ParallelCleaner
    ):
        """处理批次"""
        loop = asyncio.get_event_loop()
        processed_total = 0
        batch_num = 0

        while True:
            batch = await input_queue.get()
            if batch is None:
                break

            batch_num += 1
            results = await loop.run_in_executor(
                None, cleaner._process_batch_sync, batch, lang
            )
            await output_queue.put(results)

            processed_total += len(batch)
            if processed_total % 20000 == 0:
                logger.info(f"{lang}: 已处理 {processed_total} 条")

        logger.info(f"{lang}: 处理完成，共 {processed_total} 条，{batch_num} 个批次")
        await output_queue.put(None)

    async def _write_batches(
        self,
        input_queue: asyncio.Queue,
        output_file: Path
    ) -> int:
        """异步写入批次"""
        count = 0

        async def batch_generator():
            nonlocal count
            while True:
                batch = await input_queue.get()
                if batch is None:
                    break
                count += len(batch)
                yield batch

        await async_write_jsonl_batches(str(output_file), batch_generator())
        return count

    def _stage_deduplicate(self) -> Dict:
        """阶段 2: 去重"""
        result = {}

        for lang in ['chinese', 'english']:
            input_dir = self.config.get_stage_dir(1, lang)
            output_dir = self.config.get_stage_dir(2, lang)

            stats = deduplicate_files(str(input_dir), str(output_dir))
            result[lang] = stats

        return result

    def _stage_tokenize(self) -> Dict:
        """阶段 3: Tokenize"""
        result = {}

        for lang in ['chinese', 'english']:
            input_dir = self.config.get_stage_dir(2, lang)
            output_dir = self.config.get_stage_dir(3, lang)

            stats = tokenize_files(
                str(input_dir), str(output_dir),
                self.config.tokenizer_name,
                self.config.eos_token_id
            )
            result[lang] = stats

        return result

    def _stage_chunk(self) -> Dict:
        """阶段 4: 切块并混合"""
        chinese_dir = self.config.get_stage_dir(3, 'chinese')
        english_dir = self.config.get_stage_dir(3, 'english')
        output_dir = self.config.get_stage_dir(4)

        stats = process_and_chunk(
            str(chinese_dir), str(english_dir), str(output_dir),
            self.config.chinese_ratio, self.config.english_ratio,
            self.config.context_size
        )
        return stats

    def _stage_binarize(self) -> Dict:
        """阶段 5: 转为二进制格式"""
        from .utils.io_utils import read_jsonl, write_bin

        result = {}
        input_dir = self.config.get_stage_dir(4)
        output_dir = self.config.get_stage_dir(5)

        for split in ['train', 'val', 'test']:
            input_file = input_dir / f"{split}.jsonl"
            output_file = output_dir / f"{split}.bin"

            if not input_file.exists():
                logger.warning(f"输入文件不存在: {input_file}")
                result[split] = 0
                continue

            chunks = self._read_jsonl_safe(str(input_file))

            def extract_xy():
                for chunk in chunks:
                    x = chunk.get('x', [])
                    y = chunk.get('y', [])
                    if x and y:
                        yield (x, y)

            count = write_bin(str(output_file), extract_xy())
            result[split] = count

        return result

    def _read_jsonl_safe(self, filepath: str):
        """安全读取 jsonl"""
        from .utils.io_utils import read_jsonl
        try:
            yield from read_jsonl(filepath)
        except Exception as e:
            logger.error(f"读取失败 {filepath}: {e}")


__all__ = ['DataPipeline']
