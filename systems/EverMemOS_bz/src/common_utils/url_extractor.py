"""
URL内容提取工具
用于从网页中提取标题、描述、图片等元数据信息
"""

import re
import aiohttp
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Tag

from core.observation.logger import get_logger

logger = get_logger(__name__)

# 请求配置
DEFAULT_TIMEOUT = 10  # 秒
DEFAULT_MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; MemsysBot/1.0; +https://memsys.ai/bot)"


class URLExtractor:
    """URL内容提取器"""

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        """
        初始化URL提取器

        Args:
            timeout: 请求超时时间（秒）
            max_content_length: 最大内容长度
            user_agent: 用户代理字符串
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent

    async def extract_metadata(
        self, url: str, need_redirect: bool = True
    ) -> Dict[str, Any]:
        """
        提取URL的元数据信息

        Args:
            url: 要提取的URL
            need_redirect: 是否需要跟随重定向获取最终URL

        Returns:
            Dict[str, Any]: 提取的元数据信息
        """
        try:
            # 获取最终URL（如果需要重定向）
            final_url = url
            if need_redirect:
                final_url = await self._get_final_url(url)

            # 获取网页内容
            html_content = await self._fetch_html_content(final_url)
            if not html_content:
                return self._create_empty_metadata(url, final_url)

            # 解析HTML并提取元数据
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = self._extract_metadata_from_soup(soup, final_url)
            metadata['original_url'] = url
            metadata['final_url'] = final_url

            return metadata

        except Exception as e:
            logger.error("提取URL元数据失败: %s, 错误: %s", url, str(e))
            return self._create_error_metadata(url, str(e))

    async def _get_final_url(self, url: str) -> str:
        """
        获取重定向后的最终URL

        Args:
            url: 原始URL

        Returns:
            str: 最终URL
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {'User-Agent': self.user_agent}

            # 创建SSL上下文，跳过证书验证（用于内容提取，相对安全）
            import ssl

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers, connector=connector
            ) as session:
                # 只发送HEAD请求获取最终URL，不下载内容
                async with session.head(url, allow_redirects=True) as response:
                    return str(response.url)

        except Exception as e:
            logger.warning("获取最终URL失败: %s, 错误: %s", url, str(e))
            return url

    async def _fetch_html_content(self, url: str) -> Optional[str]:
        """
        获取HTML内容

        Args:
            url: 要获取的URL

        Returns:
            Optional[str]: HTML内容，失败时返回None
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            # 创建SSL上下文，跳过证书验证（用于内容提取，相对安全）
            import ssl

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers, connector=connector
            ) as session:
                async with session.get(url) as response:
                    # 检查内容类型
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type:
                        logger.warning(
                            "非HTML内容: %s, content-type: %s", url, content_type
                        )
                        return None

                    # 检查内容长度
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_content_length:
                        logger.warning("内容过大: %s, size: %s", url, content_length)
                        return None

                    # 读取内容
                    content = await response.text()
                    if len(content) > self.max_content_length:
                        logger.warning("内容过大: %s, size: %d", url, len(content))
                        return None

                    return content

        except Exception as e:
            logger.error("获取HTML内容失败: %s, 错误: %s", url, str(e))
            return None

    def _extract_metadata_from_soup(
        self, soup: BeautifulSoup, url: str
    ) -> Dict[str, Any]:
        """
        从BeautifulSoup对象中提取元数据

        Args:
            soup: BeautifulSoup对象
            url: 页面URL

        Returns:
            Dict[str, Any]: 提取的元数据
        """
        metadata = {
            'title': None,
            'description': None,
            'image': None,
            'site_name': None,
            'url': url,
            'type': None,
            'favicon': None,
            'og_tags': {},
            'twitter_tags': {},
            'meta_tags': {},
        }

        try:
            # 提取Open Graph标签
            og_tags = self._extract_og_tags(soup)
            metadata['og_tags'] = og_tags

            # 提取Twitter Card标签
            twitter_tags = self._extract_twitter_tags(soup)
            metadata['twitter_tags'] = twitter_tags

            # 提取基本meta标签
            meta_tags = self._extract_meta_tags(soup)
            metadata['meta_tags'] = meta_tags

            # 优先使用Open Graph信息，但跳过包含模板变量的值
            metadata['title'] = (
                self._get_safe_value(og_tags.get('title'))
                or self._get_safe_value(twitter_tags.get('title'))
                or self._get_safe_value(self._extract_title(soup))
                or self._get_safe_value(meta_tags.get('title'))
            )

            metadata['description'] = (
                self._get_safe_value(og_tags.get('description'))
                or self._get_safe_value(twitter_tags.get('description'))
                or self._get_safe_value(meta_tags.get('description'))
            )

            metadata['image'] = self._get_safe_value(
                og_tags.get('image')
            ) or self._get_safe_value(twitter_tags.get('image'))

            metadata['site_name'] = self._get_safe_value(og_tags.get('site_name'))
            metadata['type'] = self._get_safe_value(og_tags.get('type'))
            metadata['favicon'] = self._extract_favicon(soup, url)

            # 清理和验证数据
            metadata = self._clean_metadata(metadata)

        except Exception as e:
            logger.error("解析元数据失败: %s, 错误: %s", url, str(e))

        return metadata

    def _extract_og_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """提取Open Graph标签"""
        og_tags = {}

        for tag in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            if tag.get('content'):
                property_name = tag['property'][3:]  # 去掉'og:'前缀
                og_tags[property_name] = tag['content'].strip()

        return og_tags

    def _extract_twitter_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """提取Twitter Card标签"""
        twitter_tags = {}

        for tag in soup.find_all(
            'meta', attrs={'name': lambda x: x and x.startswith('twitter:')}
        ):
            if tag.get('content'):
                name = tag['name'][8:]  # 去掉'twitter:'前缀
                twitter_tags[name] = tag['content'].strip()

        return twitter_tags

    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """提取基本meta标签"""
        meta_tags = {}

        # 提取title
        title_tag = soup.find('meta', attrs={'name': 'title'})
        if title_tag and title_tag.get('content'):
            meta_tags['title'] = title_tag['content'].strip()

        # 提取description
        description_tag = soup.find('meta', attrs={'name': 'description'})
        if description_tag and description_tag.get('content'):
            meta_tags['description'] = description_tag['content'].strip()

        # 提取keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and keywords_tag.get('content'):
            meta_tags['keywords'] = keywords_tag['content'].strip()

        # 提取author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag and author_tag.get('content'):
            meta_tags['author'] = author_tag['content'].strip()

        return meta_tags

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """提取页面标题"""
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        return None

    def _extract_first_image(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """提取第一个有意义的图片"""
        # 查找img标签
        img_tags = soup.find_all('img', src=True)

        for img in img_tags:
            src = img['src'].strip()
            if not src:
                continue

            # 转换为绝对URL
            absolute_url = urljoin(base_url, src)

            # 简单过滤：跳过明显的装饰性图片
            if self._is_meaningful_image(img, src):
                return absolute_url

        return None

    def _is_meaningful_image(self, img_tag: Tag, src: str) -> bool:
        """判断图片是否有意义（非装饰性）"""
        # 跳过明显的装饰性图片
        skip_patterns = [
            'icon',
            'logo',
            'avatar',
            'button',
            'pixel',
            'spacer',
            'blank',
            'transparent',
            '1x1',
            'tracking',
        ]

        src_lower = src.lower()
        if any(pattern in src_lower for pattern in skip_patterns):
            return False

        # 检查图片尺寸属性
        width = img_tag.get('width')
        height = img_tag.get('height')

        if width and height:
            try:
                w, h = int(width), int(height)
                # 跳过太小的图片
                if w < 100 or h < 100:
                    return False
                # 跳过明显的装饰性尺寸
                if w == 1 or h == 1:
                    return False
            except (ValueError, TypeError):
                pass

        return True

    def _extract_favicon(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """提取网站图标"""
        # 查找link标签中的icon
        icon_links = soup.find_all('link', rel=lambda x: x and 'icon' in x.lower())

        for link in icon_links:
            href = link.get('href')
            if href:
                return urljoin(base_url, href.strip())

        # 默认favicon路径
        parsed_url = urlparse(base_url)
        default_favicon = f"{parsed_url.scheme}://{parsed_url.netloc}/favicon.ico"
        return default_favicon

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理和验证元数据"""
        # 清理字符串字段
        string_fields = [
            'title',
            'description',
            'image',
            'site_name',
            'type',
            'favicon',
            'url',
        ]
        for field in string_fields:
            if metadata.get(field):
                # 清理多余的空白字符
                cleaned_value = re.sub(r'\s+', ' ', str(metadata[field])).strip()
                metadata[field] = cleaned_value

                # 限制长度
                if field == 'title' and len(metadata[field]) > 200:
                    metadata[field] = metadata[field][:200] + '...'
                elif field == 'description' and len(metadata[field]) > 500:
                    metadata[field] = metadata[field][:500] + '...'

        # 验证URL格式
        url_fields = ['image', 'favicon', 'url']
        for field in url_fields:
            if metadata.get(field) and not self._is_valid_url(metadata[field]):
                metadata[field] = None

        return metadata

    def _contains_template_variables(self, text: str) -> bool:
        """
        检查文本是否包含模板变量

        检查以下模板变量格式：
        - ${variable}
        - {{variable}}
        - {variable}
        - #{variable}
        - @{variable}

        Args:
            text: 要检查的文本

        Returns:
            bool: 如果包含模板变量返回True，否则返回False
        """
        if not text or not isinstance(text, str):
            return False

        # 定义模板变量的正则表达式模式
        template_patterns = [
            r'\$\{[^}]+\}',  # ${variable}
            r'\{\{[^}]+\}\}',  # {{variable}}
            r'#\{[^}]+\}',  # #{variable}
            r'@\{[^}]+\}',  # @{variable}
            # {variable} - 只匹配包含字母、数字、点、下划线的变量名
            r'\{[a-zA-Z_][a-zA-Z0-9_.]*\}',
        ]

        # 检查每个模式
        for pattern in template_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _get_safe_value(self, value: str) -> Optional[str]:
        """
        获取安全的值，如果包含模板变量则返回None

        Args:
            value: 要检查的值

        Returns:
            Optional[str]: 如果值有效且不包含模板变量则返回原值，否则返回None
        """
        if not value or not isinstance(value, str):
            return None

        # 清理空白字符
        cleaned_value = value.strip()
        if not cleaned_value:
            return None

        # 检查是否包含模板变量
        if self._contains_template_variables(cleaned_value):
            return None

        return cleaned_value

    def _is_valid_url(self, url: str) -> bool:
        """验证URL格式"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _create_empty_metadata(
        self, original_url: str, final_url: str
    ) -> Dict[str, Any]:
        """创建空的元数据"""
        return {
            'title': None,
            'description': None,
            'image': None,
            'site_name': None,
            'url': final_url,
            'original_url': original_url,
            'final_url': final_url,
            'type': None,
            'favicon': None,
            'og_tags': {},
            'twitter_tags': {},
            'meta_tags': {},
            'error': None,
        }

    def _create_error_metadata(self, url: str, error: str) -> Dict[str, Any]:
        """创建错误元数据"""
        return {
            'title': None,
            'description': None,
            'image': None,
            'site_name': None,
            'url': url,
            'original_url': url,
            'final_url': url,
            'type': None,
            'favicon': None,
            'og_tags': {},
            'twitter_tags': {},
            'meta_tags': {},
            'error': error,
        }


# 全局实例
_url_extractor = URLExtractor()


async def extract_url_metadata(url: str, need_redirect: bool = True) -> Dict[str, Any]:
    """
    提取URL元数据的便捷函数

    Args:
        url: 要提取的URL
        need_redirect: 是否需要跟随重定向

    Returns:
        Dict[str, Any]: 元数据信息
    """
    return await _url_extractor.extract_metadata(url, need_redirect)
