"""MongoDB-based cluster storage for ClusterManager."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from memory_layer.cluster_manager.storage import ClusterStorage
from core.observation.logger import get_logger
from core.di.decorators import component
from component.mongodb_client_factory import MongoDBClientFactory
from infra_layer.adapters.out.persistence.document.memory.cluster_state import ClusterState

logger = get_logger(__name__)


@component()
class MongoClusterStorage(ClusterStorage):
    """MongoDB-based cluster storage implementation.
    
    每个群组的聚类状态存储为一个独立的 MongoDB 文档。
    """
    
    def __init__(self, mongodb_factory: MongoDBClientFactory):
        """初始化 MongoDB 聚类存储
        
        Args:
            mongodb_factory: MongoDB 客户端工厂
        """
        self.mongodb_factory = mongodb_factory
        logger.info("MongoClusterStorage initialized")
    
    async def save_cluster_state(
        self,
        group_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """保存群组的聚类状态到 MongoDB
        
        Args:
            group_id: 群组ID
            state: 聚类状态字典
            
        Returns:
            是否保存成功
        """
        try:
            # 查找是否已存在
            existing = await ClusterState.find_one(ClusterState.group_id == group_id)
            
            if existing:
                # 更新现有文档
                for key, value in state.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                await existing.save()
                logger.debug(f"Updated cluster state for group {group_id}")
            else:
                # 创建新文档
                state["group_id"] = group_id
                cluster_state = ClusterState(**state)
                await cluster_state.insert()
                logger.info(f"Created cluster state for group {group_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save cluster state for group {group_id}: {e}", exc_info=True)
            return False
    
    async def load_cluster_state(
        self,
        group_id: str
    ) -> Optional[Dict[str, Any]]:
        """从 MongoDB 加载群组的聚类状态
        
        Args:
            group_id: 群组ID
            
        Returns:
            聚类状态字典，如果不存在返回 None
        """
        try:
            cluster_state = await ClusterState.find_one(ClusterState.group_id == group_id)
            
            if cluster_state is None:
                return None
            
            # 转换为字典（排除 MongoDB 内部字段）
            state_dict = cluster_state.model_dump(exclude={"id", "revision_id"})
            return state_dict
        
        except Exception as e:
            logger.error(f"Failed to load cluster state for group {group_id}: {e}", exc_info=True)
            return None
    
    async def get_cluster_assignments(
        self,
        group_id: str
    ) -> Dict[str, str]:
        """获取群组的 event_id -> cluster_id 映射
        
        Args:
            group_id: 群组ID
            
        Returns:
            事件到聚类的映射字典
        """
        try:
            cluster_state = await ClusterState.find_one(ClusterState.group_id == group_id)
            
            if cluster_state is None:
                return {}
            
            return cluster_state.eventid_to_cluster or {}
        
        except Exception as e:
            logger.error(f"Failed to get cluster assignments for group {group_id}: {e}", exc_info=True)
            return {}
    
    async def clear(self, group_id: Optional[str] = None) -> bool:
        """清除聚类状态
        
        Args:
            group_id: 群组ID（None 表示清除所有群组）
            
        Returns:
            是否清除成功
        """
        try:
            if group_id is None:
                # 清除所有聚类状态
                result = await ClusterState.delete_all()
                logger.info(f"Cleared all cluster states, deleted {result.deleted_count} documents")
            else:
                # 清除指定群组的聚类状态
                cluster_state = await ClusterState.find_one(ClusterState.group_id == group_id)
                if cluster_state:
                    await cluster_state.delete()
                    logger.info(f"Cleared cluster state for group {group_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear cluster state: {e}", exc_info=True)
            return False

