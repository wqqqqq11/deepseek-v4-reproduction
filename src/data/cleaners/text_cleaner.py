"""文本基础清洗"""

import re
import unicodedata
import logging


logger = logging.getLogger(__name__)


class TextCleaner:
    """基础文本清洗：异常符号、空文本、长短文本过滤"""

    INVALID_CHARS = [
        '\x00-\x08',
        '\x0b-\x0c',
        '\x0e-\x1f',
        '\x7f-\x9f',
    ]

    REPLACEMENTS = [
        (r'\.{3,}', '...'),
        (r'-{3,}', '---'),
        (r'_{3,}', '___'),
        (r'!{3,}', '!!'),
        (r'\?{3,}', '??'),
    ]

    def __init__(self, min_length: int = 100, max_length: int = 100000):
        self.min_length = min_length
        self.max_length = max_length
        self.invalid_regex = re.compile('[' + ''.join(self.INVALID_CHARS) + ']')
        self.replace_patterns = [(re.compile(p), r) for p, r in self.REPLACEMENTS]

    def clean(self, text: str) -> str:
        """清洗文本，返回空字符串表示丢弃"""
        if not text:
            return ""

        try:
            result = self._normalize_unicode(text)
            result = self._remove_invalid_chars(result)
            result = self._normalize_whitespace(result)
            result = self._apply_replacements(result)

            if not self._check_length(result):
                return ""

            return result
        except Exception as e:
            logger.warning(f"文本清洗异常: {e}")
            return ""

    def _normalize_unicode(self, text: str) -> str:
        """统一 Unicode 规范化"""
        return unicodedata.normalize('NFC', text)

    def _remove_invalid_chars(self, text: str) -> str:
        """去除控制字符"""
        return self.invalid_regex.sub('', text)

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白"""
        text = re.sub(r'[\t\n\r\f\v]+', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _apply_replacements(self, text: str) -> str:
        """应用替换规则"""
        result = text
        for pattern, replacement in self.replace_patterns:
            result = pattern.sub(replacement, result)
        return result

    def _check_length(self, text: str) -> bool:
        """检查文本长度"""
        return self.min_length <= len(text) <= self.max_length


class TextValidator:
    """文本质量验证器"""

    def __init__(self, min_length: int = 100, max_length: int = 100000):
        self.min_length = min_length
        self.max_length = max_length

    def is_valid(self, text: str) -> bool:
        """验证文本是否有效"""
        if not text or not isinstance(text, str):
            return False

        if len(text) < self.min_length:
            return False

        if len(text) > self.max_length:
            return False

        if text.count('\n') > len(text) / 10:
            return False

        return True
