"""质量过滤器"""

import re
import unicodedata
import logging
from typing import Dict, Optional


logger = logging.getLogger(__name__)


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
