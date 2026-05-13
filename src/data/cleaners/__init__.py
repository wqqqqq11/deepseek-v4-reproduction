"""数据清洗模块"""

import logging
import asyncio
from typing import Dict, Optional, Iterator, List
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from functools import partial

from .html_cleaner import HtmlCleaner
from .url_cleaner import UrlCleaner
from .text_cleaner import TextCleaner, TextValidator
from .quality_filter import QualityFilter, LanguageDetector

logger = logging.getLogger(__name__)


class BaseCleaner(ABC):
    """清洗器基类"""

    @abstractmethod
    def clean(self, text: str) -> Optional[str]:
        """清洗文本，返回 None 表示丢弃"""
        pass


class CleaningChain:
    """清洗链：按顺序执行多个清洗步骤"""

    def __init__(self, config: Dict):
        self.config = config
        self._init_cleaners()

    def _init_cleaners(self):
        """初始化清洗器"""
        self.html_cleaner = HtmlCleaner()
        self.url_cleaner = UrlCleaner()
        self.text_cleaner = TextCleaner(
            min_length=self.config.get('min_text_length', 100),
            max_length=self.config.get('max_text_length', 100000)
        )
        self.quality_filter = QualityFilter(
            min_chinese_ratio=self.config.get('min_chinese_ratio', 0.3),
            min_english_ratio=self.config.get('min_english_ratio', 0.5)
        )
        self.text_validator = TextValidator(
            min_length=self.config.get('min_text_length', 100),
            max_length=self.config.get('max_text_length', 100000)
        )

    def process_batch(self, docs: List[Dict], lang: str) -> List[Dict]:
        """批量处理文档"""
        if not docs:
            return []

        batch_id = id(docs) % 10000
        logger.info(f"批次 [{batch_id}] 开始: {len(docs)} 条 ({lang})")

        try:
            texts = [d.get('text', '') for d in docs]
            sources = [d.get('source', 'unknown') for d in docs]

            texts = self.html_cleaner.batch_clean(texts)
            texts = self.url_cleaner.batch_clean(texts)
            texts = self.text_cleaner.batch_clean(texts)

            # 只保留非空文本，跳过质量过滤
            results = []
            for i, text in enumerate(texts):
                if text and len(text.strip()) > 0:
                    results.append({
                        'text': text,
                        'source': sources[i],
                        'lang': lang
                    })

            logger.info(f"批次 [{batch_id}] 完成: {len(results)}/{len(docs)} 条保留")
            return results
        except Exception as e:
            logger.error(f"批次 [{batch_id}] 处理异常: {e}")
            return []

    def _batch_filter(self, texts: List[str], lang: str) -> List[int]:
        """批量质量过滤"""
        if lang == 'chinese':
            return self.quality_filter.batch_filter_chinese(texts)
        return self.quality_filter.batch_filter_english(texts)


def _init_worker(config: Dict):
    """工作进程初始化"""
    global _worker_chain
    _worker_chain = CleaningChain(config)


def _process_batch_worker(batch_lang: tuple) -> List[Dict]:
    """工作进程处理函数"""
    batch, lang = batch_lang
    return _worker_chain.process_batch(batch, lang)


class ParallelCleaner:
    """并行清洗器"""

    def __init__(self, config: Dict, num_workers: int = None):
        self.config = config
        self.num_workers = num_workers or cpu_count()
        self.batch_size = config.get('batch_size', 5000)
        self.pool = None

    def __enter__(self):
        self.pool = Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.config,)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            self.pool.join()

    async def process_stream(
        self,
        doc_iterator: Iterator[Dict],
        lang: str,
        output_queue: asyncio.Queue
    ):
        """并行处理文档流"""
        loop = asyncio.get_event_loop()

        batch = []
        for doc in doc_iterator:
            batch.append(doc)

            if len(batch) >= self.batch_size:
                results = await loop.run_in_executor(
                    None,
                    self._process_batch_sync,
                    batch,
                    lang
                )
                await output_queue.put(results)
                batch = []

        if batch:
            results = await loop.run_in_executor(
                None,
                self._process_batch_sync,
                batch,
                lang
            )
            await output_queue.put(results)

        await output_queue.put(None)

    def _process_batch_sync(self, batch: List[Dict], lang: str) -> List[Dict]:
        """同步批量处理（用于线程池）"""
        future = self.pool.apply_async(_process_batch_worker, ((batch, lang),))
        return future.get()


def process_documents_parallel(
    documents: Iterator[Dict],
    config: Dict,
    lang: str
) -> Iterator[Dict]:
    """多进程并行处理文档"""
    batch_size = config.get('batch_size', 5000)
    num_workers = config.get('num_workers', cpu_count())

    with Pool(processes=num_workers) as pool:
        initializer = partial(_init_worker, config)
        pool._initializer = initializer
        pool._initargs = (config,)

        chain = CleaningChain(config)
        batch = []

        for doc in documents:
            batch.append(doc)

            if len(batch) >= batch_size:
                results = chain.process_batch(batch, lang)
                yield from results
                batch = []

        if batch:
            results = chain.process_batch(batch, lang)
            yield from results


__all__ = [
    'HtmlCleaner',
    'UrlCleaner',
    'TextCleaner',
    'TextValidator',
    'QualityFilter',
    'LanguageDetector',
    'CleaningChain',
    'ParallelCleaner',
    'process_documents_parallel'
]
