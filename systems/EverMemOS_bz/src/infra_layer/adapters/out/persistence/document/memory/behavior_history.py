from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from core.oxm.mongo.audit_base import AuditBase


class BehaviorHistory(DocumentBase, AuditBase):
    """
    行为历史文档模型

    记录用户的各种行为历史，包括聊天、邮件、文件操作等。
    """

    # 联合主键
    user_id: Indexed(str) = Field(..., description="用户ID，联合主键")
    timestamp: Indexed(datetime) = Field(..., description="行为发生时间戳，联合主键")

    # 行为信息
    behavior_type: List[str] = Field(
        ...,
        description="行为类型列表（chat、follow-up、Smart-Reply、Vote、file、Email、link-doc等）",
    )
    event_id: Optional[str] = Field(
        default=None, description="关联记忆单元ID（若存在）"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="元信息：对话详情、Email原文等"
    )

    # 通用字段
    extend: Optional[Dict[str, Any]] = Field(default=None, description="备用拓展字段")

    model_config = ConfigDict(
        collection="behavior_histories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_001",
                "timestamp": datetime(2021, 1, 1, 0, 0, 0),
                "behavior_type": ["chat", "follow-up"],
                "event_id": "evt_001",
                "meta": {
                    "conversation_id": "conv_001",
                    "message_count": 5,
                    "duration_minutes": 15,
                    "topics": ["技术讨论", "项目规划"],
                },
                "extend": {"priority": "high", "location": "office"},
            }
        },
    )

    class Settings:
        """Beanie 设置"""

        name = "behavior_histories"
        indexes = [
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("behavior_type", ASCENDING),
                    ("timestamp", ASCENDING),
                ],
                name="idx_user_type_timestamp",
            ),
            IndexModel([("event_id", ASCENDING)], name="idx_event_id"),
        ]
        validate_on_save = True
        use_state_management = True
