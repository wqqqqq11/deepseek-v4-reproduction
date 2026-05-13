"""HTML 标签清洗"""

import re
import logging


logger = logging.getLogger(__name__)


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

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
