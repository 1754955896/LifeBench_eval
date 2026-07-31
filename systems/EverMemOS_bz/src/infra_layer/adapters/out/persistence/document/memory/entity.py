from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from beanie import PydanticObjectId
from core.oxm.mongo.audit_base import AuditBase


class Entity(DocumentBase, AuditBase):
    """
    实体库文档模型

    存储从情景记忆中提取出的实体信息，包括人物、项目、组织等。
    """

    # 基本信息
    name: str = Field(..., description="实体名字")
    type: str = Field(..., description="实体类型（Project、Person、组织名等）")
    aliases: Optional[List[str]] = Field(default=None, description="关联的别名")

    # 通用字段
    extend: Optional[Dict[str, Any]] = Field(default=None, description="备用拓展字段")

    model_config = ConfigDict(
        collection="entities",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "name": "张三",
                "type": "Person",
                "aliases": ["小张", "张工", "zhangsan"],
                "extend": {"department": "技术部", "level": "高级工程师"},
            }
        },
    )

    @property
    def entity_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie 设置"""

        name = "entities"
        indexes = [
            # 注意：entity_id 映射到 _id 字段，MongoDB 已自动为 _id 创建主键索引
            IndexModel([("aliases", ASCENDING)], name="idx_aliases", sparse=True),
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
