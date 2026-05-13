"""HTML 标签清洗"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class HtmlCleaner:
    """去除 HTML 标签和实体"""

    HTML_TAGS = [
        (r'<script[^>]*>.*?</script>', ''),
        (r'<style[^>]*>.*?</style>', ''),
        (r'<[^>]+>', ' '),
    ]

    HTML_ENTITIES = [
        (r'&nbsp;', ' '),
        (r'&quot;', '"'),
        (r'&amp;', '&'),
        (r'&lt;', '<'),
        (r'&gt;', '>'),
        (r'&#\d+;', ''),
        (r'&[a-zA-Z]+;', ''),
    ]

    def __init__(self):
        self.tag_patterns = [(re.compile(p, re.DOTALL), r) for p, r in self.HTML_TAGS]
        self.entity_patterns = [(re.compile(p), r) for p, r in self.HTML_ENTITIES]

    def clean(self, text: str) -> str:
        """去除 HTML 标签和实体"""
        if not text:
            return ""

        try:
            result = text
            for pattern, replacement in self.tag_patterns:
                result = pattern.sub(replacement, result)
            for pattern, replacement in self.entity_patterns:
                result = pattern.sub(replacement, result)
            return self._normalize_whitespace(result)
        except Exception as e:
            logger.warning(f"HTML 清洗异常: {e}")
            return text

    def batch_clean(self, texts: List[str]) -> List[str]:
        """批量清洗 HTML"""
        if not texts:
            return []

        try:
            if PANDAS_AVAILABLE and len(texts) > 100:
                return self._batch_clean_pandas(texts)
            return [self.clean(t) for t in texts]
        except Exception as e:
            logger.warning(f"批量HTML清洗异常: {e}, fallback到单条处理")
            return [self.clean(t) for t in texts]

    def _batch_clean_pandas(self, texts: List[str]) -> List[str]:
        """使用pandas批量清洗"""
        s = pd.Series(texts)

        for pattern, replacement in self.tag_patterns:
            s = s.str.replace(pattern.pattern, replacement, regex=True)

        for pattern, replacement in self.entity_patterns:
            s = s.str.replace(pattern.pattern, replacement, regex=True)

        s = s.str.replace(r'\s+', ' ', regex=True)
        s = s.str.strip()

        return s.tolist()

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
