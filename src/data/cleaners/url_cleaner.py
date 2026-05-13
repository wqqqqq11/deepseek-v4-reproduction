"""URL 和广告文本清洗"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class UrlCleaner:
    """去除 URL、邮箱、广告文本"""

    URL_PATTERNS = [
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        r'www\.[^\s<>"{}|\\^`\[\]]+',
        r'ftp://[^\s<>"{}|\\^`\[\]]+',
    ]

    EMAIL_PATTERN = r'\S+@\S+\.\S+'

    AD_PATTERNS = [
        r'点击.*?(?:查看|下载|购买)',
        r'(?:扫码|扫描).*?(?:二维码|关注)',
        r'(?:微信|微博|抖音|快手).*?关注',
        r'限时.*?优惠',
        r'.*?专享价.*?',
        r'点击关注.*?公众号',
    ]

    def __init__(self):
        self.url_regex = re.compile('|'.join(self.URL_PATTERNS), re.IGNORECASE)
        self.email_regex = re.compile(self.EMAIL_PATTERN)
        self.ad_regex_list = [re.compile(p) for p in self.AD_PATTERNS]

    def clean(self, text: str) -> str:
        """去除 URL、邮箱和广告文本"""
        if not text:
            return ""

        try:
            result = self.url_regex.sub('', text)
            result = self.email_regex.sub('', result)
            result = self._remove_ad_text(result)
            return self._normalize_whitespace(result)
        except Exception as e:
            logger.warning(f"URL 清洗异常: {e}")
            return text

    def batch_clean(self, texts: List[str]) -> List[str]:
        """批量清洗 URL 和广告"""
        if not texts:
            return []

        try:
            if PANDAS_AVAILABLE and len(texts) > 100:
                return self._batch_clean_pandas(texts)
            return [self.clean(t) for t in texts]
        except Exception as e:
            logger.warning(f"批量URL清洗异常: {e}, fallback到单条处理")
            return [self.clean(t) for t in texts]

    def _batch_clean_pandas(self, texts: List[str]) -> List[str]:
        """使用pandas批量清洗"""
        s = pd.Series(texts)

        s = s.str.replace(self.url_regex.pattern, '', regex=True)
        s = s.str.replace(self.email_regex.pattern, '', regex=True)

        for pattern in self.ad_regex_list:
            s = s.str.replace(pattern.pattern, '', regex=True)

        s = s.str.replace(r'\s+', ' ', regex=True)
        s = s.str.strip()

        return s.tolist()

    def _remove_ad_text(self, text: str) -> str:
        """去除广告文本"""
        result = text
        for pattern in self.ad_regex_list:
            result = pattern.sub('', result)
        return result

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
