from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field
from core.oxm.mongo.audit_base import AuditBase


class UserProfile(DocumentBase, AuditBase):
    """
    用户画像文档模型
    
    存储从聚类对话中自动提取的用户画像信息
    """
    
    # 联合主键
    user_id: Indexed(str) = Field(..., description="用户ID")
    group_id: Indexed(str) = Field(..., description="群组ID")
    
    # Profile 内容（JSON 格式存储）
    profile_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="用户画像数据（包含角色、技能、偏好、性格等）"
    )
    
    # 元信息
    scenario: str = Field(default="group_chat", description="场景类型：group_chat 或 assistant")
    confidence: float = Field(default=0.0, description="画像置信度 (0-1)")
    version: int = Field(default=1, description="画像版本号")
    
    # 聚类关联
    cluster_ids: List[str] = Field(
        default_factory=list,
        description="关联的聚类ID列表"
    )
    memcell_count: int = Field(default=0, description="参与提取的 MemCell 数量")
    
    # 历史记录
    last_updated_cluster: Optional[str] = Field(
        default=None,
        description="最后一次更新时使用的聚类ID"
    )
    
    class Settings:
        name = "user_profiles"
        indexes = [
            [("user_id", 1), ("group_id", 1)],  # 联合索引
        ]

