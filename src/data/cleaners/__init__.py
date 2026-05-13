"""数据清洗模块"""

import logging
from typing import Dict, Optional, Iterator
from abc import ABC, abstractmethod

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
        self.steps = []
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

    def process_chinese(self, doc: Dict) -> Optional[Dict]:
        """处理中文文档"""
        return self._process(doc, lang='chinese')

    def process_english(self, doc: Dict) -> Optional[Dict]:
        """处理英文文档"""
        return self._process(doc, lang='english')

    def _process(self, doc: Dict, lang: str) -> Optional[Dict]:
        """执行清洗流程"""
        text = doc.get('text', '')
        if not text:
            return None

        try:
            text = self.html_cleaner.clean(text)
            text = self.url_cleaner.clean(text)
            text = self.text_cleaner.clean(text)

            if not text:
                return None

            if lang == 'chinese':
                text = self.quality_filter.filter_chinese(text)
            elif lang == 'english':
                text = self.quality_filter.filter_english(text)

            if not text:
                return None

            return {
                'text': text,
                'source': doc.get('source', 'unknown'),
                'lang': lang
            }
        except Exception as e:
            logger.warning(f"清洗流程异常: {e}")
            return None


def process_documents(documents: Iterator[Dict],
                     chain: CleaningChain,
                     lang: str) -> Iterator[Dict]:
    """批量处理文档"""
    processed = 0
    dropped = 0

    for doc in documents:
        if lang == 'chinese':
            result = chain.process_chinese(doc)
        else:
            result = chain.process_english(doc)

        if result:
            processed += 1
            yield result
        else:
            dropped += 1

        if (processed + dropped) % 10000 == 0:
            logger.info(f"处理: {processed}, 丢弃: {dropped}")

    logger.info(f"处理完成: 保留 {processed}, 丢弃 {dropped}")


__all__ = [
    'HtmlCleaner',
    'UrlCleaner',
    'TextCleaner',
    'TextValidator',
    'QualityFilter',
    'LanguageDetector',
    'CleaningChain',
    'process_documents',
    'BaseCleaner'
]
