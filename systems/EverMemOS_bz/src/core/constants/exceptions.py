"""
异常处理模块

本模块定义了项目中使用的所有自定义异常类和错误代码。
遵循统一的异常处理规范，便于错误追踪和调试。
"""

from enum import Enum
from typing import Optional, Dict, Any
from core.constants.errors import ErrorCode


class BaseException(Exception):
    """基础异常类

    所有自定义异常的基类，提供统一的异常处理接口。
    包含错误代码、错误消息和可选的详细信息。
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        初始化基础异常

        Args:
            code: 错误代码
            message: 错误消息
            details: 可选的详细信息字典
            original_exception: 原始异常对象
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception

    def __str__(self) -> str:
        """返回异常的字符串表示"""
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """返回异常的详细表示"""
        details_str = f", details={self.details}" if self.details else ""
        original_str = (
            f", original={self.original_exception}" if self.original_exception else ""
        )
        return f"{self.__class__.__name__}(code='{self.code}', message='{self.message}'{details_str}{original_str})"

    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典格式，便于序列化"""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "exception_type": self.__class__.__name__,
        }


class AgentException(BaseException):
    """Agent相关异常基类

    所有与Agent执行相关的异常的基类。
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(code, message, details, original_exception)


class ValidationException(BaseException):
    """数据验证异常

    当输入数据验证失败时抛出此异常。
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if field:
            message = f"Field '{field}': {message}"

        super().__init__(
            code=ErrorCode.VALIDATION_ERROR.value, message=message, details=details
        )


class ResourceNotFoundException(BaseException):
    """资源未找到异常

    当请求的资源不存在时抛出此异常。
    """

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND.value, message=message, details=details
        )


class ConfigurationException(BaseException):
    """配置异常

    当系统配置错误或缺失时抛出此异常。
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if config_key:
            message = f"Configuration error for '{config_key}': {message}"

        super().__init__(
            code=ErrorCode.CONFIGURATION_ERROR.value, message=message, details=details
        )


class DatabaseException(BaseException):
    """数据库异常

    当数据库操作失败时抛出此异常。
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        if operation:
            message = f"Database {operation} failed: {message}"

        super().__init__(
            code=ErrorCode.DATABASE_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


class ExternalServiceException(BaseException):
    """外部服务异常

    当调用外部服务失败时抛出此异常。
    """

    def __init__(
        self,
        service_name: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        if status_code:
            message = f"{service_name} service error (HTTP {status_code}): {message}"
        else:
            message = f"{service_name} service error: {message}"

        super().__init__(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


class AuthenticationException(BaseException):
    """认证异常

    当用户认证失败时抛出此异常。
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            code=ErrorCode.AUTHENTICATION_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


class LLMOutputParsingException(AgentException):
    """LLM输出解析异常

    当LLM返回的内容无法正确解析时抛出此异常。
    """

    def __init__(
        self,
        message: str,
        llm_output: Optional[str] = None,
        expected_format: Optional[str] = None,
        attempt_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        if expected_format:
            message = f"LLM输出解析失败，期望格式: {expected_format}, 错误: {message}"
        if attempt_count:
            message = f"{message} [第{attempt_count}次尝试]"

        # 将LLM输出内容添加到详细信息中
        if details is None:
            details = {}
        if llm_output:
            details["llm_output"] = llm_output[:500]  # 限制长度避免过长

        super().__init__(
            code=ErrorCode.LLM_OUTPUT_PARSING_ERROR.value,
            message=message,
            details=details,
            original_exception=original_exception,
        )


def create_exception_from_error_code(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    original_exception: Optional[Exception] = None,
) -> BaseException:
    """
    根据错误代码创建对应的异常对象

    Args:
        error_code: 错误代码枚举
        message: 错误消息
        details: 可选的详细信息
        original_exception: 原始异常对象

    Returns:
        对应的异常对象
    """
    return BaseException(
        code=error_code.value,
        message=message,
        details=details,
        original_exception=original_exception,
    )


# Long Job System Errors - 长任务系统错误类
from core.longjob.longjob_error import (
    FatalError,
    BusinessLogicError,
    LongJobError,
    JobNotFoundError,
    JobAlreadyExistsError,
    JobStateError,
    ManagerShutdownError,
    MaxConcurrentJobsError,
)

# 导出长任务系统错误类
__all__ = [
    # 错误代码和基础异常
    'ErrorCode',
    'BaseException',
    'AgentException',
    'ValidationException',
    'ResourceNotFoundException',
    'ConfigurationException',
    'DatabaseException',
    'ExternalServiceException',
    'AuthenticationException',
    'LLMOutputParsingException',
    'create_exception_from_error_code',
    # 长任务系统错误类
    'FatalError',
    'BusinessLogicError',
    'LongJobError',
    'JobNotFoundError',
    'JobAlreadyExistsError',
    'JobStateError',
    'ManagerShutdownError',
    'MaxConcurrentJobsError',
]
