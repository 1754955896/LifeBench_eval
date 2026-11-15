"""
文本处理工具模块

提供各种文本处理的通用工具函数，包括智能截取、格式化等功能。
"""

from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass


class TokenType(Enum):
    """Token类型枚举"""

    CJK_CHAR = "cjk_char"  # 中日韩字符
    ENGLISH_WORD = "english_word"  # 英文单词
    CONTINUOUS_NUMBER = "continuous_number"  # 连续数字
    PUNCTUATION = "punctuation"  # 标点符号
    WHITESPACE = "whitespace"  # 空白字符
    OTHER = "other"  # 其他字符


@dataclass
class Token:
    """文本Token"""

    type: TokenType
    content: str
    start_pos: int
    end_pos: int
    score: float = 0.0


@dataclass
class TokenConfig:
    """Token配置"""

    cjk_char_score: float = 1.0
    english_word_score: float = 1.5
    continuous_number_score: float = 0.8
    punctuation_score: float = 0.5
    whitespace_score: float = 0.3
    other_score: float = 0.5


class SmartTextParser:
    """智能文本解析器

    能够区分不同类型的token，支持可配置的分数计算，
    提供从左到右遍历和基于总分数的智能截断功能。
    """

    def __init__(self, config: TokenConfig = None):
        """初始化解析器

        Args:
            config: Token配置，如果为None则使用默认配置
        """
        self.config = config or TokenConfig()

        # 中日韩字符范围
        self._cjk_ranges = [
            (0x4E00, 0x9FFF),  # CJK统一表意文字
            (0x3400, 0x4DBF),  # CJK扩展A
            (0x20000, 0x2A6DF),  # CJK扩展B
            (0x2A700, 0x2B73F),  # CJK扩展C
            (0x2B740, 0x2B81F),  # CJK扩展D
            (0x2B820, 0x2CEAF),  # CJK扩展E
            (0x3040, 0x309F),  # 平假名
            (0x30A0, 0x30FF),  # 片假名
            (0xAC00, 0xD7AF),  # 韩文音节
        ]

    def _is_cjk_char(self, char: str) -> bool:
        """判断是否为中日韩字符"""
        if not char:
            return False
        code = ord(char)
        return any(start <= code <= end for start, end in self._cjk_ranges)

    def _is_english_char(self, char: str) -> bool:
        """判断是否为英文字符"""
        return char.isalpha() and ord(char) < 128

    def _is_punctuation(self, char: str) -> bool:
        """判断是否为标点符号"""
        # 常见的标点符号
        punctuation_chars = set('.,!?;:"\'()[]{}+-*/%=<>@#$&|~`^_\\/')
        return char in punctuation_chars or (
            0x2000 <= ord(char) <= 0x206F  # 通用标点
            or 0x3000 <= ord(char) <= 0x303F  # CJK符号和标点
            or 0xFF00 <= ord(char) <= 0xFFEF  # 全角ASCII和半角片假名
        )

    def parse_tokens(self, text: str, max_score: float = None) -> List[Token]:
        """解析文本为Token列表

        Args:
            text: 要解析的文本
            max_score: 最大分数限制，达到此分数时提前截断解析

        Returns:
            List[Token]: Token列表
        """
        if not text:
            return []

        tokens = []
        current_score = 0.0
        i = 0
        text_len = len(text)

        while i < text_len:
            char = text[i]
            start_pos = i

            # 处理中日韩字符
            if self._is_cjk_char(char):
                token = Token(
                    type=TokenType.CJK_CHAR,
                    content=char,
                    start_pos=start_pos,
                    end_pos=i + 1,
                    score=self.config.cjk_char_score,
                )
                tokens.append(token)
                current_score += token.score
                i += 1

                # 检查是否需要提前截断
                if max_score is not None and current_score > max_score:
                    # 移除刚添加的token，因为它使分数超出限制
                    tokens.pop()
                    break

            # 处理英文单词
            elif self._is_english_char(char):
                word_end = i
                while word_end < text_len and (
                    self._is_english_char(text[word_end]) or text[word_end] in "'-"
                ):
                    word_end += 1

                token = Token(
                    type=TokenType.ENGLISH_WORD,
                    content=text[i:word_end],
                    start_pos=start_pos,
                    end_pos=word_end,
                    score=self.config.english_word_score,
                )
                tokens.append(token)
                current_score += token.score
                i = word_end

                # 检查是否需要提前截断
                if max_score is not None and current_score > max_score:
                    # 移除刚添加的token，因为它使分数超出限制
                    tokens.pop()
                    break

            # 处理连续数字
            elif char.isdigit():
                num_end = i
                while num_end < text_len and (
                    text[num_end].isdigit() or text[num_end] in ".,"
                ):
                    num_end += 1

                token = Token(
                    type=TokenType.CONTINUOUS_NUMBER,
                    content=text[i:num_end],
                    start_pos=start_pos,
                    end_pos=num_end,
                    score=self.config.continuous_number_score,
                )
                tokens.append(token)
                current_score += token.score
                i = num_end

                # 检查是否需要提前截断
                if max_score is not None and current_score > max_score:
                    # 移除刚添加的token，因为它使分数超出限制
                    tokens.pop()
                    break

            # 处理标点符号
            elif self._is_punctuation(char):
                token = Token(
                    type=TokenType.PUNCTUATION,
                    content=char,
                    start_pos=start_pos,
                    end_pos=i + 1,
                    score=self.config.punctuation_score,
                )
                tokens.append(token)
                current_score += token.score
                i += 1

                # 检查是否需要提前截断
                if max_score is not None and current_score > max_score:
                    # 移除刚添加的token，因为它使分数超出限制
                    tokens.pop()
                    break

            # 处理空白字符
            elif char.isspace():
                # 合并连续的空白字符
                space_end = i
                while space_end < text_len and text[space_end].isspace():
                    space_end += 1

                token = Token(
                    type=TokenType.WHITESPACE,
                    content=text[i:space_end],
                    start_pos=start_pos,
                    end_pos=space_end,
                    score=self.config.whitespace_score,
                )
                tokens.append(token)
                current_score += token.score
                i = space_end

                # 检查是否需要提前截断
                if max_score is not None and current_score > max_score:
                    # 移除刚添加的token，因为它使分数超出限制
                    tokens.pop()
                    break

            # 处理其他字符
            else:
                token = Token(
                    type=TokenType.OTHER,
                    content=char,
                    start_pos=start_pos,
                    end_pos=i + 1,
                    score=self.config.other_score,
                )
                tokens.append(token)
                current_score += token.score
                i += 1

                # 检查是否需要提前截断
                if max_score is not None and current_score > max_score:
                    # 移除刚添加的token，因为它使分数超出限制
                    tokens.pop()
                    break

        return tokens

    def calculate_total_score(self, tokens: List[Token]) -> float:
        """计算Token列表的总分数

        Args:
            tokens: Token列表

        Returns:
            float: 总分数
        """
        return sum(token.score for token in tokens)

    def smart_truncate_by_score(
        self,
        text: str,
        max_score: float,
        suffix: str = "...",
        enable_fallback: bool = True,
    ) -> str:
        """基于分数智能截断文本

        Args:
            text: 要截断的文本
            max_score: 最大允许分数
            suffix: 截断后添加的后缀
            enable_fallback: 是否启用fallback模式，解析失败时按字符长度截断

        Returns:
            str: 截断后的文本
        """
        if not text:
            return text or ""

        if max_score <= 0:
            return text  # 保持向后兼容，限制<=0时返回原文

        try:
            # 首先解析完整文本
            all_tokens = self.parse_tokens(text)

            if not all_tokens:
                return text

            # 计算实际分数，如果不超过限制则无需截断
            total_score = self.calculate_total_score(all_tokens)
            if total_score <= max_score:
                return text

            # 使用完整的tokens进行截断计算
            tokens = all_tokens

            # 需要截断，找到合适的截断位置
            current_score = 0.0
            truncate_pos = len(text)

            for token in tokens:
                if current_score + token.score > max_score:
                    # 如果是英文单词或连续数字，且超出不太多，允许完整包含以避免截断
                    if (
                        token.type
                        in [TokenType.ENGLISH_WORD, TokenType.CONTINUOUS_NUMBER]
                        and current_score + token.score
                        <= max_score * 1.05  # 只允许5%超出
                        and current_score > 0
                    ):  # 必须已经有其他token，不能第一个token就超出
                        current_score += token.score
                        truncate_pos = token.end_pos
                    else:
                        truncate_pos = token.start_pos
                    break
                current_score += token.score
                truncate_pos = token.end_pos

            # 如果需要截断
            if truncate_pos < len(text):
                result = text[:truncate_pos].rstrip()
                return result + suffix if result else text

            return text

        except Exception as e:
            # Fallback模式：解析失败时使用简单的字符长度截断
            if enable_fallback:
                # 估算截断长度：假设平均每个字符1分
                estimated_length = int(max_score * 0.8)  # 保守估计
                if len(text) <= estimated_length:
                    return text

                # 简单按字符截断，避免在单词中间截断
                truncate_pos = estimated_length

                # 向后查找合适的截断点（空白或标点）
                for i in range(
                    min(estimated_length + 10, len(text) - 1),
                    max(estimated_length - 10, 0),
                    -1,
                ):
                    if text[i].isspace() or text[i] in '.,!?;:':
                        truncate_pos = i + 1
                        break

                result = text[:truncate_pos].rstrip()
                return result + suffix if result else text
            else:
                # 不启用fallback则抛出异常
                raise e

    def get_text_analysis(self, text: str) -> Dict[str, Any]:
        """获取文本分析结果

        Args:
            text: 要分析的文本

        Returns:
            Dict: 包含各种统计信息的字典
        """
        tokens = self.parse_tokens(text)

        # 统计各类型token数量
        type_counts = {token_type: 0 for token_type in TokenType}
        type_scores = {token_type: 0.0 for token_type in TokenType}

        for token in tokens:
            type_counts[token.type] += 1
            type_scores[token.type] += token.score

        return {
            "total_tokens": len(tokens),
            "total_score": self.calculate_total_score(tokens),
            "type_counts": {t.value: count for t, count in type_counts.items()},
            "type_scores": {t.value: score for t, score in type_scores.items()},
            "tokens": tokens,
        }


def smart_truncate_text(
    text: str,
    max_count: int,
    chinese_weight: float = 1.0,
    english_word_weight: float = 1.0,
    suffix: str = "...",
) -> str:
    """
    基于单词/字符计数的智能截取文本

    使用新的SmartTextParser进行更精确的token解析和分数计算。
    英文单词算一个单位，中文字符算一个单位，可以分配不同权重。

    Args:
        text: 要截取的文本
        max_count: 最大计数（权重累加后的总数）
        chinese_weight: 中文字符权重，默认1.0
        english_word_weight: 英文单词权重，默认1.0
        suffix: 截取时添加的后缀，默认为"..."

    Returns:
        str: 截取后的文本

    Examples:
        >>> smart_truncate_text("Hello World 你好世界", 4)
        'Hello World 你好...'  # 2个英文单词 + 2个中文字符 = 4
        >>> smart_truncate_text("Hello World 你好世界", 4, chinese_weight=0.5)
        'Hello World 你好世界'  # 2个英文单词 + 4*0.5个中文字符 = 4
    """
    if not text or max_count <= 0:
        return text or ""

    if not isinstance(text, str):
        text = str(text)

    # 使用新的智能解析器进行截断
    config = TokenConfig(
        cjk_char_score=chinese_weight,
        english_word_score=english_word_weight,
        continuous_number_score=english_word_weight,  # 数字使用英文单词权重
        punctuation_score=0.0,  # 标点不计分，保持向后兼容
        whitespace_score=0.0,  # 空白不计分，保持向后兼容
        other_score=0.0,  # 其他字符不计分，保持向后兼容
    )

    parser = SmartTextParser(config)
    return parser.smart_truncate_by_score(text, max_count, suffix)


def clean_whitespace(text: str) -> str:
    """
    清理文本中的多余空白字符

    使用SmartTextParser进行更精确的空白字符处理，
    保持其他token的完整性。

    Args:
        text: 要清理的文本

    Returns:
        str: 清理后的文本
    """
    if not text:
        return text

    if not isinstance(text, str):
        text = str(text)

    # 使用智能解析器处理空白字符
    parser = SmartTextParser()
    tokens = parser.parse_tokens(text)

    if not tokens:
        return text.strip()

    # 重建文本，合并连续空白为单个空格
    result_parts = []
    prev_was_whitespace = False

    for token in tokens:
        if token.type == TokenType.WHITESPACE:
            if not prev_was_whitespace:
                result_parts.append(' ')  # 统一使用单个空格
            prev_was_whitespace = True
        else:
            result_parts.append(token.content)
            prev_was_whitespace = False

    # 去除首尾空白
    return ''.join(result_parts).strip()
