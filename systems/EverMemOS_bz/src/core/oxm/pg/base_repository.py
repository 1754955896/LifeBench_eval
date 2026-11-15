from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic, Type
from core.oxm.pg.audit_base import get_auditable_model
from core.di.decorators import repository

# 定义泛型类型
T = TypeVar('T', bound=get_auditable_model())


class BaseSoftDeleteRepository(Generic[T], ABC):
    """支持软删除的基础仓储接口 - 纯业务接口，无技术实现细节"""

    @abstractmethod
    async def add(self, entity: T) -> T:
        """添加新实体"""
        raise NotImplementedError

    @abstractmethod
    async def get(self, entity_id: int, include_deleted: bool = False) -> Optional[T]:
        """根据ID获取实体（默认排除已删除）"""
        raise NotImplementedError

    @abstractmethod
    async def get_all(self, include_deleted: bool = False) -> List[T]:
        """获取所有实体（默认排除已删除）"""
        raise NotImplementedError

    @abstractmethod
    async def update(self, entity: T) -> T:
        """更新实体"""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, entity_id: int, deleted_by: str = "system") -> bool:
        """软删除实体"""
        raise NotImplementedError

    @abstractmethod
    async def restore(self, entity_id: int, restored_by: str = "system") -> bool:
        """恢复软删除的实体"""
        raise NotImplementedError

    @abstractmethod
    async def hard_delete(self, entity_id: int) -> bool:
        """硬删除实体（慎用！）"""
        raise NotImplementedError

    @abstractmethod
    async def count(self, include_deleted: bool = False) -> int:
        """统计实体数量（默认排除已删除）"""
        raise NotImplementedError
