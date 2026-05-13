"""质量过滤器"""

import re
import logging
import numpy as np
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class QualityFilter:
    """语言比例和质量过滤"""

    def __init__(self,
                 min_chinese_ratio: float = 0.3,
                 min_english_ratio: float = 0.5):
        self.min_chinese_ratio = min_chinese_ratio
        self.min_english_ratio = min_english_ratio

        self.chinese_regex = re.compile(r'[\u4e00-\u9fff]')
        self.english_regex = re.compile(r'[a-zA-Z]')
        self.punct_regex = re.compile(r'[^\w\s]')
        self.digit_regex = re.compile(r'\d')

    def filter_chinese(self, text: str) -> Optional[str]:
        """中文数据质量过滤"""
        if not text:
            return None

        try:
            stats = self._calc_stats(text)

            if stats['chinese_ratio'] < self.min_chinese_ratio:
                return None

            if stats['punct_ratio'] > 0.5:
                return None

            if stats['digit_ratio'] > 0.5:
                return None

            if self._is_garbage(text):
                return None

            return text
        except Exception as e:
            logger.warning(f"中文过滤异常: {e}")
            return None

    def filter_english(self, text: str) -> Optional[str]:
        """英文数据质量过滤"""
        if not text:
            return None

        try:
            stats = self._calc_stats(text)

            if stats['english_ratio'] < self.min_english_ratio:
                return None

            if stats['punct_ratio'] > 0.5:
                return None

            if stats['digit_ratio'] > 0.5:
                return None

            if self._is_garbage(text):
                return None

            return text
        except Exception as e:
            logger.warning(f"英文过滤异常: {e}")
            return None

    def batch_filter_chinese(self, texts: List[str]) -> List[int]:
        """批量中文过滤，返回通过的索引列表"""
        if not texts:
            return []

        try:
            if PANDAS_AVAILABLE and len(texts) > 100:
                return self._batch_filter_pandas(texts, 'chinese')
            return [i for i, t in enumerate(texts) if self.filter_chinese(t)]
        except Exception as e:
            logger.warning(f"批量中文过滤异常: {e}, fallback到单条处理")
            return [i for i, t in enumerate(texts) if self.filter_chinese(t)]

    def batch_filter_english(self, texts: List[str]) -> List[int]:
        """批量英文过滤，返回通过的索引列表"""
        if not texts:
            return []

        try:
            if PANDAS_AVAILABLE and len(texts) > 100:
                return self._batch_filter_pandas(texts, 'english')
            return [i for i, t in enumerate(texts) if self.filter_english(t)]
        except Exception as e:
            logger.warning(f"批量英文过滤异常: {e}, fallback到单条处理")
            return [i for i, t in enumerate(texts) if self.filter_english(t)]

    def _batch_filter_pandas(self, texts: List[str], lang: str) -> List[int]:
        """使用pandas批量过滤"""
        # 先过滤空字符串，记录原始索引
        non_empty_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) >= 10:
                non_empty_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return []

        s = pd.Series(valid_texts)
        total_chars = s.str.len().values

        chinese_counts = s.str.count(r'[一-龥]').values
        english_counts = s.str.count(r'[a-zA-Z]').values
        punct_counts = s.str.count(r'[^\w\s]').values
        digit_counts = s.str.count(r'\d').values

        chinese_ratio = chinese_counts / total_chars
        english_ratio = english_counts / total_chars
        punct_ratio = punct_counts / total_chars
        digit_ratio = digit_counts / total_chars

        valid_mask = (punct_ratio <= 0.5) & (digit_ratio <= 0.5)

        if lang == 'chinese':
            valid_mask &= chinese_ratio >= self.min_chinese_ratio
        else:
            valid_mask &= english_ratio >= self.min_english_ratio

        printable_ratio = s.apply(
            lambda x: sum(c.isprintable() for c in x) / len(x) if x else 0
        ).values
        valid_mask &= printable_ratio >= 0.8

        # 映射回原始索引
        valid_positions = np.where(valid_mask)[0].tolist()
        valid_indices = [non_empty_indices[i] for i in valid_positions]
        return valid_indices

    def _calc_stats(self, text: str) -> Dict[str, float]:
        """计算文本统计信息"""
        total_chars = len(text)
        if total_chars == 0:
            return {
                'chinese_ratio': 0,
                'english_ratio': 0,
                'punct_ratio': 0,
                'digit_ratio': 0
            }

        chinese_chars = len(self.chinese_regex.findall(text))
        english_chars = len(self.english_regex.findall(text))
        punct_chars = len(self.punct_regex.findall(text))
        digit_chars = len(self.digit_regex.findall(text))

        return {
            'chinese_ratio': chinese_chars / total_chars,
            'english_ratio': english_chars / total_chars,
            'punct_ratio': punct_chars / total_chars,
            'digit_ratio': digit_chars / total_chars
        }

    def _is_garbage(self, text: str) -> bool:
        """检测是否为乱码/垃圾文本"""
        if len(text) < 10:
            return True

        printable = sum(1 for c in text if c.isprintable())
        if printable / len(text) < 0.8:
            return True

        lines = text.split('\n')
        if len(lines) > 100:
            avg_line_len = len(text) / len(lines)
            if avg_line_len < 20:
                return True

        unique_chars = len(set(text.lower()))
        if unique_chars < 10:
            return True

        return False


class LanguageDetector:
    """简单语言检测器"""

    def __init__(self):
        self.chinese_regex = re.compile(r'[\u4e00-\u9fff]')
        self.english_regex = re.compile(r'[a-zA-Z]')

    def detect(self, text: str) -> str:
        """检测主要语言类型"""
        chinese_chars = len(self.chinese_regex.findall(text))
        english_chars = len(self.english_regex.findall(text))

        if chinese_chars > english_chars:
            return 'chinese'
        elif english_chars > chinese_chars:
            return 'english'
        else:
            return 'mixed'
