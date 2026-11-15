from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase


class Relationship(DocumentBase, AuditBase):
    """
    关系库文档模型

    描述实体之间的关系，支持多种关系类型和详细信息。
    """

    # 联合主键
    source_entity_id: Indexed(str) = Field(..., description="主实体ID，联合主键")
    target_entity_id: Indexed(str) = Field(..., description="客实体ID，联合主键")

    # 关系信息
    relationship: List[Dict[str, str]] = Field(
        ..., description="关系列表，每个关系包含type、content、detail等字段"
    )

    # 通用字段
    extend: Optional[Dict[str, Any]] = Field(default=None, description="备用拓展字段")

    model_config = ConfigDict(
        collection="relationships",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "source_entity_id": "entity_001",
                "target_entity_id": "entity_002",
                "relationship": [
                    {
                        "type": "人际关系",
                        "content": "项目协作",
                        "detail": "在电商平台重构项目中有合作",
                    },
                    {
                        "type": "工作关系",
                        "content": "上下级",
                        "detail": "张三负责指导李四的技术工作",
                    },
                ],
                "extend": {"strength": "strong", "context": "工作环境"},
            }
        },
    )

    class Settings:
        """Beanie 设置"""

        name = "relationships"
        indexes = [
            IndexModel(
                [("source_entity_id", ASCENDING), ("target_entity_id", ASCENDING)],
                unique=True,
                name="idx_source_target_unique",
            ),
            IndexModel(
                [("target_entity_id", ASCENDING), ("source_entity_id", ASCENDING)],
                unique=True,
                name="idx_target_source_unique",
            ),
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
