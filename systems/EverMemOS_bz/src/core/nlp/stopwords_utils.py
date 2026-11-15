"""
停用词表工具类

提供停用词表的加载和管理功能，支持哈工大停用词表。
"""

import os
import logging
from typing import Set, Optional
from common_utils.project_path import CURRENT_DIR

logger = logging.getLogger(__name__)


class StopwordsManager:
    """停用词表管理器"""

    def __init__(self, stopwords_file_path: Optional[str] = None):
        """初始化停用词表管理器

        Args:
            stopwords_file_path: 停用词表文件路径，如果为None则使用默认路径
        """
        self.stopwords_file_path = (
            stopwords_file_path or self._get_default_stopwords_path()
        )
        self._stopwords: Optional[Set[str]] = None
        self.load_stopwords()

    def _get_default_stopwords_path(self) -> str:
        """获取默认停用词表文件路径"""
        return str(CURRENT_DIR / "config" / "stopwords" / "hit_stopwords.txt")

    def load_stopwords(self) -> Set[str]:
        """加载停用词表

        Returns:
            停用词集合
        """
        if self._stopwords is not None:
            return self._stopwords

        stopwords = set()

        # 检查文件是否存在
        if not os.path.exists(self.stopwords_file_path):
            logger.warning(f"停用词表文件不存在: {self.stopwords_file_path}")
            logger.info("将使用空的停用词表")
            self._stopwords = stopwords
            return stopwords

        try:
            with open(self.stopwords_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:  # 跳过空行
                        stopwords.add(word)

            logger.info(f"成功加载停用词表，共 {len(stopwords)} 个停用词")
            self._stopwords = stopwords
            return stopwords

        except Exception as e:
            logger.error(f"加载停用词表失败: {e}")
            logger.info("将使用空的停用词表")
            self._stopwords = set()
            return set()

    def is_stopword(self, word: str) -> bool:
        """判断是否为停用词

        Args:
            word: 要检查的词

        Returns:
            如果是停用词返回True，否则返回False
        """
        return word in self._stopwords

    def filter_stopwords(self, words: list, min_length: int = 1) -> list:
        """过滤停用词

        Args:
            words: 词列表
            min_length: 最小词长度，小于此长度的词也会被过滤

        Returns:
            过滤后的词列表
        """

        filtered_words = []
        for word in words:
            if (
                word not in self._stopwords and len(word) >= min_length and word.strip()
            ):  # 过滤空白字符
                filtered_words.append(word)

        return filtered_words


# 全局停用词管理器实例
_stopwords_manager: Optional[StopwordsManager] = StopwordsManager()


def filter_stopwords(words: list, min_length: int = 1) -> list:
    """便捷函数：过滤停用词

    Args:
        words: 词列表
        min_length: 最小词长度

    Returns:
        过滤后的词列表
    """
    return _stopwords_manager.filter_stopwords(words, min_length)
