"""
Mock Repository实现

仅保留ConversationDataRepository的Mock实现，其他Repository已使用真实实现
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from memory_layer.memcell_extractor.base_memcell_extractor import RawData


# ==================== 接口定义 ====================


class ConversationDataRepository(ABC):
    """对话数据访问接口"""

    @abstractmethod
    async def save_conversation_data(
        self, raw_data_list: List[RawData], group_id: str
    ) -> bool:
        """保存对话数据"""
        pass

    @abstractmethod
    async def get_conversation_data(
        self,
        group_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[RawData]:
        """获取对话数据"""
        pass

    @abstractmethod
    async def delete_conversation_data(self, group_id: str) -> bool:
        """
        删除指定群组的所有对话数据

        Args:
            group_id: 群组ID

        Returns:
            bool: 删除成功返回True，失败返回False
        """
        pass
