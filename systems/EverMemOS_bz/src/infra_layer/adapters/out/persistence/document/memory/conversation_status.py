from datetime import datetime
from typing import Optional
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId
from core.oxm.mongo.audit_base import AuditBase


class ConversationStatus(DocumentBase, AuditBase):
    """
    对话状态文档模型

    存储对话的状态信息，包括群组ID、消息读取时间等。
    """

    # 基本信息
    group_id: str = Field(..., description="群组ID，为空表示私聊")
    old_msg_start_time: Optional[datetime] = Field(
        default=None, description="对话窗口读取起始时间"
    )
    new_msg_start_time: Optional[datetime] = Field(
        default=None, description="累积新对话读取起始时间"
    )
    last_memcell_time: Optional[datetime] = Field(
        default=None, description="累积memCell读取起始时间"
    )

    model_config = ConfigDict(
        collection="conversation_status",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "group_id": "group_001",
                "old_msg_start_time": datetime(2021, 1, 1, 0, 0, 0),
                "new_msg_start_time": datetime(2021, 1, 1, 0, 0, 0),
                "last_memcell_time": datetime(2021, 1, 1, 0, 0, 0),
            }
        },
    )

    @property
    def conversation_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie 设置"""

        name = "conversation_status"
        indexes = [
            # 注意：conversation_id 映射到 _id 字段，MongoDB 已自动为 _id 创建主键索引
            IndexModel(
                [("group_id", ASCENDING)], name="idx_group_id", unique=True
            ),  # group_id 必须唯一
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
