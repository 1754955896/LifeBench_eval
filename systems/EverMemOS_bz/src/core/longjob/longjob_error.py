"""
Long job system specific errors.
长任务系统专用错误定义。
"""


class FatalError(Exception):
    """
    致命错误，不应该重试
    用于标识那些重试也无法解决的错误

    Examples:
        - 内存不足
        - 系统级错误
        - 配置错误
        - 编程错误（TypeError, AttributeError等）
    """


class BusinessLogicError(Exception):
    """
    业务逻辑错误，可以重试
    用于标识那些可能通过重试解决的错误

    Examples:
        - 网络连接错误
        - 临时性数据库连接问题
        - 第三方服务暂时不可用
        - 资源锁定冲突
    """


class LongJobError(Exception):
    """
    长任务系统基础错误
    所有长任务相关错误的基类
    """


class JobNotFoundError(LongJobError):
    """任务未找到错误"""


class JobAlreadyExistsError(LongJobError):
    """任务已存在错误"""


class JobStateError(LongJobError):
    """任务状态错误"""


class ManagerShutdownError(LongJobError):
    """管理器已关闭错误"""


class MaxConcurrentJobsError(LongJobError):
    """超过最大并发任务数错误"""
