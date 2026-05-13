"""URL 和广告文本清洗"""

import re
import logging


logger = logging.getLogger(__name__)


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
