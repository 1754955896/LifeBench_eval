from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, TIMESTAMP, event
from common_utils.datetime_utils import get_now_with_timezone
from core.context.context import get_current_user_info
from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_auditable_model() -> SQLModel:
    """
    获取可审计的基础模型类

    该模型包含审计字段，通过事件监听器自动填充：
    - created_at, updated_at: 由事件监听器自动设置时间戳
    - created_by, updated_by: 由事件监听器自动设置操作用户
    - deleted_at, deleted_by: 在软删除时由事件监听器或业务逻辑设置

    Returns:
        SQLModel: 可审计的基础模型类
    """

    class AuditableModel(SQLModel):
        """可审计的基础模型，包含创建和更新信息"""

        # 时间审计字段 - 由事件监听器自动填充
        created_at: Optional[datetime] = Field(
            default=None,
            sa_column=Column(TIMESTAMP(timezone=True), nullable=True),
            description="创建时间（由事件监听器自动填充）",
        )

        updated_at: Optional[datetime] = Field(
            default=None,
            sa_column=Column(TIMESTAMP(timezone=True), nullable=True),
            description="更新时间（由事件监听器自动填充）",
        )

        deleted_at: Optional[datetime] = Field(
            default=None,
            sa_column=Column(TIMESTAMP(timezone=True), nullable=True),
            description="删除时间（软删除时设置）",
        )

        # 用户审计字段 - 由事件监听器自动填充
        created_by: Optional[str] = Field(
            default=None, description="创建者（由事件监听器自动填充）"
        )
        updated_by: Optional[str] = Field(
            default=None, description="更新者（由事件监听器自动填充）"
        )
        deleted_by: Optional[str] = Field(
            default=None, description="删除者（软删除时设置）"
        )

        def soft_delete(self, deleted_by: str):
            """软删除记录"""
            self.deleted_at = get_now_with_timezone()
            self.deleted_by = deleted_by

        def restore(self, restored_by: str = None):
            """恢复软删除的记录"""
            # restored_by 参数保留以保持接口兼容性，但实际由事件监听器设置
            _ = restored_by  # 避免未使用参数的警告
            self.deleted_at = None
            self.deleted_by = None

        @property
        def is_deleted(self) -> bool:
            """检查记录是否被软删除"""
            return self.deleted_at is not None

    # 注册事件监听器
    @event.listens_for(AuditableModel, 'before_insert', propagate=True)
    def before_insert_listener(
        mapper, connection, target
    ):  # pylint: disable=unused-argument
        """INSERT操作前的事件监听器"""
        # 忽略未使用的参数（SQLAlchemy事件监听器必须的签名）
        _ = mapper, connection

        current_time = get_now_with_timezone()
        current_user_id = _get_current_user_id()

        # 设置创建时间和创建者
        if hasattr(target, 'created_at') and target.created_at is None:
            target.created_at = current_time

        if hasattr(target, 'created_by') and target.created_by is None:
            target.created_by = current_user_id or "system"

        # 设置更新时间和更新者
        if hasattr(target, 'updated_at') and target.updated_at is None:
            target.updated_at = current_time

        if hasattr(target, 'updated_by') and target.updated_by is None:
            target.updated_by = current_user_id or "system"

    @event.listens_for(AuditableModel, 'before_update', propagate=True)
    def before_update_listener(
        mapper, connection, target
    ):  # pylint: disable=unused-argument
        """UPDATE操作前的事件监听器"""
        # 忽略未使用的参数（SQLAlchemy事件监听器必须的签名）
        _ = mapper, connection

        current_time = get_now_with_timezone()
        current_user_id = _get_current_user_id()

        # 设置更新时间和更新者
        if hasattr(target, 'updated_at'):
            target.updated_at = current_time

        # 只在updated_by为None时设置，不覆盖已有值（如"system"）
        if hasattr(target, 'updated_by') and target.updated_by is None:
            target.updated_by = current_user_id or "system"

        # 特殊处理软删除场景
        if hasattr(target, 'deleted_at') and target.deleted_at is not None:
            # 如果设置了deleted_at，说明是软删除操作
            if hasattr(target, 'deleted_by') and target.deleted_by is None:
                target.deleted_by = current_user_id or "system"

    return AuditableModel


def _get_current_user_id() -> Optional[str]:
    """
    获取当前用户ID

    Returns:
        Optional[str]: 当前用户ID，如果未设置则返回None
    """
    try:
        user_info = get_current_user_info()
        if user_info and 'user_id' in user_info:
            return str(user_info['user_id'])
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("获取当前用户信息失败: %s", e)
    return None
