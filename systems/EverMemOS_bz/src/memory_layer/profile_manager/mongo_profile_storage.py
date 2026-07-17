"""MongoDB-based profile storage for ProfileManager."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from memory_layer.profile_manager.storage import ProfileStorage
from core.observation.logger import get_logger
from core.di.decorators import component
from component.mongodb_client_factory import MongoDBClientFactory
from infra_layer.adapters.out.persistence.document.memory.user_profile import UserProfile

logger = get_logger(__name__)


@component()
class MongoProfileStorage(ProfileStorage):
    """MongoDB-based profile storage implementation.
    
    每个用户在每个群组中有一个 Profile 文档。
    """
    
    def __init__(self, mongodb_factory: MongoDBClientFactory):
        """初始化 MongoDB Profile 存储
        
        Args:
            mongodb_factory: MongoDB 客户端工厂
        """
        self.mongodb_factory = mongodb_factory
        logger.info("MongoProfileStorage initialized")
    
    async def save_profile(
        self,
        user_id: str,
        profile: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """保存用户画像到 MongoDB
        
        Args:
            user_id: 用户ID
            profile: 用户画像数据（ProfileMemory 对象或字典）
            metadata: 元数据（包含 group_id, cluster_id 等）
            
        Returns:
            是否保存成功
        """
        try:
            metadata = metadata or {}
            group_id = metadata.get("group_id", "default")
            
            # 🔧 将 ProfileMemory 对象转换为字典
            if hasattr(profile, 'to_dict'):
                profile_data = profile.to_dict()
            elif isinstance(profile, dict):
                profile_data = profile
            else:
                profile_data = {"data": str(profile)}
            
            # 查找是否已存在
            existing = await UserProfile.find_one(
                UserProfile.user_id == user_id,
                UserProfile.group_id == group_id
            )
            
            if existing:
                # 更新现有文档
                existing.profile_data = profile_data
                existing.version += 1
                existing.confidence = metadata.get("confidence", existing.confidence)
                
                if "cluster_id" in metadata:
                    cluster_id = metadata["cluster_id"]
                    if cluster_id not in existing.cluster_ids:
                        existing.cluster_ids.append(cluster_id)
                    existing.last_updated_cluster = cluster_id
                
                if "memcell_count" in metadata:
                    existing.memcell_count = metadata["memcell_count"]
                
                await existing.save()
                logger.info(f"Updated profile for user {user_id} in group {group_id} (version {existing.version})")
            else:
                # 创建新文档
                user_profile = UserProfile(
                    user_id=user_id,
                    group_id=group_id,
                    profile_data=profile_data,
                    scenario=metadata.get("scenario", "group_chat"),
                    confidence=metadata.get("confidence", 0.0),
                    version=1,
                    cluster_ids=[metadata["cluster_id"]] if "cluster_id" in metadata else [],
                    memcell_count=metadata.get("memcell_count", 0),
                    last_updated_cluster=metadata.get("cluster_id")
                )
                
                await user_profile.insert()
                logger.info(f"Created profile for user {user_id} in group {group_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save profile for user {user_id}: {e}", exc_info=True)
            return False
    
    async def get_profile(self, user_id: str, group_id: str = "default") -> Optional[Any]:
        """从 MongoDB 获取用户画像
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            
        Returns:
            用户画像数据，如果不存在返回 None
        """
        try:
            user_profile = await UserProfile.find_one(
                UserProfile.user_id == user_id,
                UserProfile.group_id == group_id
            )
            
            if user_profile is None:
                return None
            
            return user_profile.profile_data
        
        except Exception as e:
            logger.error(f"Failed to get profile for user {user_id}: {e}", exc_info=True)
            return None
    
    async def get_all_profiles(self, group_id: str = "default") -> Dict[str, Any]:
        """获取群组内所有用户画像
        
        Args:
            group_id: 群组ID
            
        Returns:
            用户画像字典 {user_id: profile_data}
        """
        try:
            user_profiles = await UserProfile.find(
                UserProfile.group_id == group_id
            ).to_list()
            
            profiles = {}
            for up in user_profiles:
                profiles[up.user_id] = up.profile_data
            
            return profiles
        
        except Exception as e:
            logger.error(f"Failed to get all profiles for group {group_id}: {e}", exc_info=True)
            return {}
    
    async def get_profile_history(
        self,
        user_id: str,
        group_id: str = "default",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取用户画像历史版本
        
        注意：当前实现只返回最新版本，因为历史版本没有单独存储
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            limit: 返回数量限制
            
        Returns:
            历史版本列表
        """
        try:
            user_profile = await UserProfile.find_one(
                UserProfile.user_id == user_id,
                UserProfile.group_id == group_id
            )
            
            if user_profile is None:
                return []
            
            # 返回当前版本作为历史记录
            history = [{
                "version": user_profile.version,
                "profile": user_profile.profile_data,
                "confidence": user_profile.confidence,
                "updated_at": user_profile.updated_at,
                "cluster_id": user_profile.last_updated_cluster,
                "memcell_count": user_profile.memcell_count
            }]
            
            return history[:limit] if limit else history
        
        except Exception as e:
            logger.error(f"Failed to get profile history for user {user_id}: {e}", exc_info=True)
            return []
    
    async def clear(self, group_id: Optional[str] = None) -> bool:
        """清除用户画像
        
        Args:
            group_id: 群组ID（None 表示清除所有群组）
            
        Returns:
            是否清除成功
        """
        try:
            if group_id is None:
                # 清除所有用户画像
                result = await UserProfile.delete_all()
                logger.info(f"Cleared all user profiles, deleted {result.deleted_count} documents")
            else:
                # 清除指定群组的用户画像
                result = await UserProfile.find(UserProfile.group_id == group_id).delete()
                logger.info(f"Cleared user profiles for group {group_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear user profiles: {e}", exc_info=True)
            return False

