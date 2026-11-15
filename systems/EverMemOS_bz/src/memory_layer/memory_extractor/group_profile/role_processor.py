"""Role management utilities for group profile extraction."""

from typing import Dict, List, Set, Optional
from enum import Enum

from core.observation.logger import get_logger

logger = get_logger(__name__)


class RoleProcessor:
    """角色处理器 - 负责角色分配和证据管理"""

    def __init__(self, data_processor):
        """
        初始化角色处理器

        Args:
            data_processor: GroupProfileDataProcessor 实例，用于验证和合并 memcell_ids
        """
        self.data_processor = data_processor

    def process_roles_with_evidences(
        self,
        role_data: Dict,
        speaker_mapping: Dict[str, Dict[str, str]],
        existing_roles: Dict,
        valid_memcell_ids: Set[str],
        memcell_list: List,
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        处理所有 roles（包括 weak 的），合并历史 evidences，并按 confidence 排序（strong 在前）

        Args:
            role_data: LLM 输出的 role 数据（包含 evidences 和 confidence）
            speaker_mapping: speaker_id 到 user_name 的映射
            existing_roles: 历史 roles 数据（包含 evidences 和 confidence）
            valid_memcell_ids: 有效的 memcell_ids 集合
            memcell_list: 当前的 memcell 列表（用于获取时间戳进行排序）

        Returns:
            处理后的 roles，格式为 role -> [{"user_id": "xxx", "user_name": "xxx", "confidence": "strong|weak", "evidences": [...]}]
        """
        from ..group_profile_memory_extractor import GroupRole

        # 定义有效的角色列表（基于 GroupRole 枚举）
        VALID_ROLES = {role.value for role in GroupRole}

        def validate_and_filter_roles(roles_dict: Dict, source: str) -> Dict:
            """验证并过滤非法角色"""
            if not roles_dict:
                return {}

            filtered = {}
            invalid_roles = []

            for role_name, assignments in roles_dict.items():
                if role_name in VALID_ROLES:
                    filtered[role_name] = assignments
                else:
                    invalid_roles.append(role_name)

            if invalid_roles:
                logger.warning(
                    f"[process_roles_with_evidences] Filtered out {len(invalid_roles)} invalid roles from {source}: {invalid_roles}"
                )

            return filtered

        # 过滤 LLM 输出和历史数据中的非法角色
        role_data = validate_and_filter_roles(role_data, "LLM output")
        existing_roles = validate_and_filter_roles(existing_roles, "historical data")

        processed_roles = {}

        # 1. 构建历史 roles 的映射 (role_name, user_id) -> evidences
        historical_role_map = {}
        for role_name, assignments in existing_roles.items():
            for assignment in assignments:
                user_id = assignment.get("user_id", "")
                if user_id:
                    key = (role_name, user_id)
                    historical_role_map[key] = {
                        "evidences": assignment.get("evidences", []),
                        "confidence": assignment.get("confidence", "weak"),
                    }

        # 2. 处理 LLM 输出的 roles
        for role_name, assignments in role_data.items():
            if not assignments:
                continue

            processed_assignments = []
            for assignment in assignments:
                # Handle both old format (string) and new format (dict)
                if isinstance(assignment, str):
                    # Old format: assignment is just speaker_id, treat as weak
                    speaker_id = assignment
                    confidence = "weak"
                    llm_evidences = []
                elif isinstance(assignment, dict):
                    # New format: assignment has speaker, confidence, evidences
                    speaker_id = assignment.get("speaker", "")
                    confidence = assignment.get("confidence", "weak")
                    llm_evidences = assignment.get("evidences", [])
                else:
                    continue

                if not speaker_id:
                    continue

                user_name = self.data_processor.get_user_name(
                    speaker_id, speaker_mapping
                )

                # 3. 合并历史 evidences
                key = (role_name, speaker_id)
                if key in historical_role_map:
                    historical_evidences = historical_role_map[key]["evidences"]
                    historical_confidence = historical_role_map[key]["confidence"]

                    # 合并 evidences（会验证 user 是否在 participants 中）
                    merged_evidences = self.data_processor.merge_memcell_ids(
                        historical=historical_evidences,
                        new=llm_evidences,
                        valid_ids=valid_memcell_ids,
                        memcell_list=memcell_list,
                        user_id=speaker_id,
                        max_count=50,
                    )

                    # 更新 confidence（如果新的更强）
                    if confidence == "strong" or historical_confidence == "strong":
                        final_confidence = "strong"
                    else:
                        final_confidence = confidence
                else:
                    # 新 role，验证 evidences（包括 participants 检查）
                    merged_evidences = (
                        self.data_processor.validate_and_filter_memcell_ids(
                            llm_evidences,
                            valid_memcell_ids,
                            user_id=speaker_id,
                            memcell_list=memcell_list,
                        )
                    )
                    final_confidence = confidence

                processed_assignments.append(
                    {
                        "user_id": speaker_id,
                        "user_name": user_name,
                        "confidence": final_confidence,
                        "evidences": merged_evidences,
                    }
                )

            # Sort assignments: strong first, then weak
            processed_assignments.sort(
                key=lambda x: (x["confidence"] != "strong", x["user_name"])
            )

            if processed_assignments:
                processed_roles[role_name] = processed_assignments

        return processed_roles
