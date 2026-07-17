"""
Base62编码工具
用于将数字ID转换为短字符串，支持0-9、a-z、A-Z共62个字符
"""

# Base62字符集：0-9 (10个) + a-z (26个) + A-Z (26个) = 62个字符
BASE62_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = len(BASE62_CHARSET)


def encode_base62(num: int) -> str:
    """
    将十进制数字编码为Base62字符串

    Args:
        num: 要编码的十进制数字（必须 >= 0）

    Returns:
        str: Base62编码后的字符串

    Raises:
        ValueError: 当输入数字小于0时

    Examples:
        >>> encode_base62(0)
        '0'
        >>> encode_base62(61)
        'Z'
        >>> encode_base62(62)
        '10'
        >>> encode_base62(1000000)
        '4C92'
    """
    if num < 0:
        raise ValueError("输入数字必须大于等于0")

    if num == 0:
        return BASE62_CHARSET[0]

    result = []
    while num > 0:
        result.append(BASE62_CHARSET[num % BASE])
        num //= BASE

    # 反转结果，因为我们是从低位到高位构建的
    return ''.join(reversed(result))


def decode_base62(encoded: str) -> int:
    """
    将Base62字符串解码为十进制数字

    Args:
        encoded: Base62编码的字符串

    Returns:
        int: 解码后的十进制数字

    Raises:
        ValueError: 当字符串包含非法字符时

    Examples:
        >>> decode_base62('0')
        0
        >>> decode_base62('Z')
        61
        >>> decode_base62('10')
        62
        >>> decode_base62('4C92')
        1000000
    """
    if not encoded:
        raise ValueError("编码字符串不能为空")

    result = 0
    for char in encoded:
        if char not in BASE62_CHARSET:
            raise ValueError(f"非法字符: {char}")
        result = result * BASE + BASE62_CHARSET.index(char)

    return result


def generate_short_code(id_value: int, min_length: int = 4) -> str:
    """
    基于ID生成短链接代码

    Args:
        id_value: 数据库ID值
        min_length: 最小长度，不足时前面补0（默认4位）

    Returns:
        str: 生成的短链接代码

    Examples:
        >>> generate_short_code(1)
        '0001'
        >>> generate_short_code(62)
        '0010'
        >>> generate_short_code(1000000)
        '4C92'
    """
    if id_value < 0:
        raise ValueError("ID值必须大于等于0")

    encoded = encode_base62(id_value)

    # 如果长度不足最小长度，前面补0
    if len(encoded) < min_length:
        encoded = BASE62_CHARSET[0] * (min_length - len(encoded)) + encoded

    return encoded


def is_valid_short_code(short_code: str) -> bool:
    """
    验证短链接代码是否有效

    Args:
        short_code: 要验证的短链接代码

    Returns:
        bool: 是否有效
    """
    if not short_code:
        return False

    # 检查是否只包含Base62字符集中的字符
    return all(char in BASE62_CHARSET for char in short_code)


def extract_id_from_short_code(short_code: str) -> int:
    """
    从短链接代码中提取原始ID

    Args:
        short_code: 短链接代码

    Returns:
        int: 原始ID值

    Raises:
        ValueError: 当短链接代码无效时
    """
    if not is_valid_short_code(short_code):
        raise ValueError(f"无效的短链接代码: {short_code}")

    return decode_base62(short_code)
