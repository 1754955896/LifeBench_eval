from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict, BaseModel
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from core.oxm.mongo.audit_base import AuditBase
from common_utils.datetime_utils import to_iso_format


class TopicInfo(BaseModel):
    """
    Topic information aligned with design document.
    """

    name: str = Field(..., description="话题名 (短语化标签)")
    summary: str = Field(..., description="一句话概述")
    status: str = Field(..., description="exploring/disagreement/consensus/implemented")
    last_active_at: datetime = Field(..., description="最近活跃时间 (=updateTime)")
    id: Optional[str] = Field(
        default=None, description="话题唯一ID (系统生成，LLM不需要提供)"
    )
    update_type: Optional[str] = Field(
        default=None, description="'new' | 'update' (仅用于增量更新时)"
    )
    old_topic_id: Optional[str] = Field(
        default=None, description="更新时指向老topic (仅用于增量更新时)"
    )
    evidences: Optional[List[str]] = Field(
        default_factory=list, description="memcell_ids 作为证据"
    )
    confidence: Optional[str] = Field(
        default=None, description="'strong' | 'weak' - 置信度"
    )

    model_config = ConfigDict(json_encoders={datetime: to_iso_format})


class RoleUser(BaseModel):
    """
    角色用户模型
    """

    user_id: str = Field(..., description="用户ID")
    user_name: str = Field(..., description="用户名")

    model_config = ConfigDict(json_encoders={datetime: to_iso_format})


class RoleAssignment(BaseModel):
    """
    角色分配模型（包含证据和置信度）
    """

    user_id: str = Field(..., description="用户ID")
    user_name: str = Field(..., description="用户名")
    confidence: Optional[str] = Field(
        default=None, description="置信度: 'strong' | 'weak'"
    )
    evidences: Optional[List[str]] = Field(
        default_factory=list, description="支持该角色分配的 memcell_ids"
    )

    model_config = ConfigDict(json_encoders={datetime: to_iso_format})


class GroupProfile(DocumentBase, AuditBase):
    """
    群组记忆文档模型

    存储群组的基本信息、角色定义、用户标签和近期话题等信息。
    """

    group_id: Indexed(str) = Field(..., description="群组ID")

    # ==================== 版本控制字段 ====================
    version: Optional[str] = Field(default=None, description="版本号，用于支持版本管理")
    is_latest: Optional[bool] = Field(
        default=True, description="是否为最新版本，默认为True"
    )

    # 群组基本信息
    group_name: Optional[str] = Field(default=None, description="群组名（不一定有）")

    # 群组话题和知识领域
    topics: Optional[List[TopicInfo]] = Field(
        default_factory=list,  # 修改为 default_factory=list，避免 None 值
        description="群组最近话题列表，包含name、summary、status、last_active_at、id、update_type、old_topic_id等字段",
    )

    # 群组角色定义
    roles: Optional[Dict[str, List[RoleAssignment]]] = Field(
        default_factory=dict,  # 修改为 default_factory=dict，避免 None 值
        description="预定义的群组角色，每个 assignment 包含 user_id, user_name, confidence, evidences 字段",
    )

    # 时间戳
    timestamp: int = Field(..., description="发生时间戳")

    # 群组长期主题
    subject: Optional[str] = Field(default=None, description="群组长期主题")

    # 群组最近话题总结
    summary: Optional[str] = Field(default=None, description="群组最近话题总结")

    # 扩展字段
    extend: Optional[Dict[str, Any]] = Field(default=None, description="备用拓展字段")

    model_config = ConfigDict(
        collection="group_profiles",
        validate_assignment=True,
        json_encoders={datetime: to_iso_format},
        json_schema_extra={
            "example": {
                "group_id": "group_12345",
                "group_name": "技术讨论组",
                "topics": [
                    {
                        "name": "Python最佳实践",
                        "summary": "讨论Python编程的最佳实践方法",
                        "status": "exploring",
                        "last_active_at": "2025-09-22T10:00:00+08:00",
                        "id": "topic_001",
                        "update_type": "new",
                        "old_topic_id": None,
                        "confidence": "strong",
                        "evidences": ["memcell_001", "memcell_002"],
                    }
                ],
                "roles": {
                    "core_contributor": [
                        {
                            "user_id": "user_123",
                            "user_name": "张三",
                            "confidence": "strong",
                            "evidences": ["memcell_001", "memcell_002"],
                        }
                    ]
                },
                "timestamp": 1726992000000,
                "subject": "技术交流与学习",
                "summary": "本群组主要讨论各种技术话题，促进技术交流",
                "extend": {"priority": "high"},
            }
        },
    )

    class Settings:
        """Beanie 设置"""

        name = "group_profiles"
        indexes = [
            # 群组ID和版本联合唯一索引
            IndexModel(
                [("group_id", ASCENDING), ("version", ASCENDING)],
                unique=True,
                name="idx_group_id_version_unique",
            ),
            # is_latest字段索引（用于快速查询最新版本）
            IndexModel(
                [("group_id", ASCENDING), ("is_latest", ASCENDING)],
                name="idx_group_id_is_latest",
            ),
            # 群组名称索引（支持模糊查询）
            IndexModel([("group_name", TEXT)], name="idx_group_name_text"),
            # 审计字段索引
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
            # 复合索引：群组ID + 更新时间
            IndexModel(
                [("group_id", ASCENDING), ("updated_at", DESCENDING)],
                name="idx_group_id_updated_at",
            ),
        ]
        validate_on_save = True
        use_state_management = True
