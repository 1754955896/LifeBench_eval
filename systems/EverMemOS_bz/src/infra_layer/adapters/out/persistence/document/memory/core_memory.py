from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase


class CoreMemory(DocumentBase, AuditBase):
    """
    核心记忆文档模型

    统一存储用户的基础信息、个人档案和偏好设置。
    单个文档包含所有三种记忆类型的数据。
    """

    user_id: Indexed(str) = Field(..., description="用户ID")

    # ==================== 版本控制字段 ====================
    version: Optional[str] = Field(default=None, description="版本号，用于支持版本管理")
    is_latest: Optional[bool] = Field(
        default=True, description="是否为最新版本，默认为True"
    )

    # ==================== BaseMemory 字段 ====================
    # 基础信息字段
    user_name: Optional[str] = Field(default=None, description="用户姓名")
    gender: Optional[str] = Field(default=None, description="性别")
    position: Optional[str] = Field(default=None, description="职位")
    supervisor_user_id: Optional[str] = Field(default=None, description="上级UserID")
    team_members: Optional[List[str]] = Field(
        default=None, description="团队成员UserID"
    )
    okr: Optional[List[Dict[str, str]]] = Field(default=None, description="OKR指标")
    base_location: Optional[str] = Field(default=None, description="base地点")
    hiredate: Optional[str] = Field(default=None, description="入职时间")
    age: Optional[int] = Field(default=None, description="年纪")
    department: Optional[str] = Field(default=None, description="部门")

    # ==================== Profile 字段 ====================
    # 个人档案字段 - 所有字段现在使用嵌入 evidences 的格式
    # 技能格式: [{"value": "Python", "level": "高级", "evidences": ["2024-01-01|conv_123"]}]
    # Legacy格式: [{"skill": "Python", "level": "高级", "evidences": ["..."]}] (自动转换)
    hard_skills: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="硬技能，SQL、Python、产品设计等，及其熟练程度，包含 evidences",
    )
    soft_skills: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="软技能，沟通能力、团队合作、情绪智力等，包含 evidences",
    )
    output_reasoning: Optional[str] = Field(
        default=None, description="当前输出结果的推理说明"
    )
    motivation_system: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="激励系统，包含 value/level/evidences"
    )
    fear_system: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="恐惧系统，包含 value/level/evidences"
    )
    value_system: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="价值体系，包含 value/level/evidences"
    )
    humor_use: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="幽默使用方式，包含 value/level/evidences"
    )
    colloquialism: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="俗语或口头禅偏好，包含 value/level/evidences"
    )

    # 其他属性格式: [{"value": "xxx", "evidences": ["2024-01-01|conv_123"]}]
    personality: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="用户性格，包含 evidences"
    )
    projects_participated: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="参与项目信息"
    )
    user_goal: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="用户目标，包含 evidences"
    )
    work_responsibility: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="工作职责，包含 evidences"
    )
    working_habit_preference: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="工作习惯偏好，包含 evidences"
    )
    interests: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="兴趣爱好，包含 evidences"
    )
    tendency: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="用户选择偏好，包含 evidences"
    )
    way_of_decision_making: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="决策方式，包含 evidences"
    )

    group_importance_evidence: Optional[Dict[str, Any]] = Field(
        default=None, description="群组重要性证据"
    )

    # ==================== 通用字段 ====================
    extend: Optional[Dict[str, Any]] = Field(default=None, description="备用拓展字段")

    model_config = ConfigDict(
        collection="core_memories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "user_name": "张三",
                "gender": "男",
                "position": "高级工程师",
                "department": "技术部",
                "age": 30,
                "personality": "内向但善于沟通，喜欢深度思考",
                "hard_skills": [{"Python": "高级"}],
                "working_habit_preference": ["远程工作", "弹性时间"],
                "user_goal": ["成为技术专家", "提升领导力"],
                "extend": {"priority": "high"},
            }
        },
    )

    class Settings:
        """Beanie 设置"""

        name = "core_memories"
        indexes = [
            # 用户ID和版本联合唯一索引
            IndexModel(
                [("user_id", ASCENDING), ("version", ASCENDING)],
                unique=True,
                name="idx_user_id_version_unique",
            ),
            # is_latest字段索引（用于快速查询最新版本）
            IndexModel(
                [("user_id", ASCENDING), ("is_latest", ASCENDING)],
                name="idx_user_id_is_latest",
            ),
            # 审计字段索引
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
