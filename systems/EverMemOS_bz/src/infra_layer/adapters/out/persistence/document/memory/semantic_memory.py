"""
SemanticMemory Beanie ODM 模型

基于 Beanie ODM 的语义记忆数据模型定义，存储用户对特定主题的语义理解和描述。
"""

from datetime import datetime
from typing import List, Optional
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import BaseModel, Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from core.oxm.mongo.audit_base import AuditBase


class SemanticMemory(DocumentBase, AuditBase):
    """
    语义记忆文档模型

    存储用户对特定主题的语义理解和描述，支持知识图谱构建。
    """

    # 核心字段（必填）
    user_id: Indexed(str, unique=True) = Field(..., description="用户ID，主键")
    subject: str = Field(..., min_length=1, description="想要描述的目标主题")
    description: str = Field(..., min_length=1, description="对subject的简单主体描述")

    # 可选字段
    link: Optional[List[str]] = Field(
        default=None, description="来源链接，如event_id、raw_data等"
    )

    model_config = ConfigDict(
        collection="semantic_memories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "subject": "Python编程",
                "description": "Python是一种高级编程语言，语法简洁，适合数据科学和Web开发",
                "link": ["evt_20241201_001", "raw_data_123"],
            }
        },
    )

    class Settings:
        """Beanie 设置"""

        name = "semantic_memories"

        indexes = [
            # 1. 用户ID唯一索引（主键）
            IndexModel(
                [("user_id", ASCENDING)], unique=True, name="idx_user_id_unique"
            ),
            # 4. 创建时间索引
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
        ]

        # 验证设置
        validate_on_save = True
        use_state_management = True


# 导出模型
__all__ = ["SemanticMemory"]
