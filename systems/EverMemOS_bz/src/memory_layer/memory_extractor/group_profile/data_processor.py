"""Data processing utilities for group profile extraction."""

from typing import Any, Dict, List, Optional, Set
import re

from core.observation.logger import get_logger

logger = get_logger(__name__)


class GroupProfileDataProcessor:
    """数据处理器 - 封装数据验证、转换、映射逻辑"""

    def __init__(self, conversation_source: str = "original"):
        """
        初始化数据处理器

        Args:
            conversation_source: 对话来源，"original" 或 "episode"
        """
        self.conversation_source = conversation_source

    def validate_and_filter_memcell_ids(
        self,
        memcell_ids: List[str],
        valid_ids: Set[str],
        user_id: Optional[str] = None,
        memcell_list: Optional[List] = None,
    ) -> List[str]:
        """
        验证并过滤 memcell_ids（用于验证 LLM 新输出的 evidences）

        验证规则：
        1. memcell_id 必须在 valid_ids 中（存在性检查）
        2. 如果指定了 user_id，还要检查 user 是否在 memcell.participants 中

        Args:
            memcell_ids: 需要验证的 memcell_ids（LLM 新输出的）
            valid_ids: 有效的 memcell_ids 集合（从当前 memcell_list 构建）
            user_id: 可选，如果提供则验证用户是否在 participants 中（用于 roles）
            memcell_list: 可选，user_id 提供时必须传入

        Returns:
            过滤后的有效 memcell_ids
        """
        if not memcell_ids:
            return []

        # 第一步：验证存在性
        valid_memcell_ids = [mid for mid in memcell_ids if mid in valid_ids]
        invalid_memcell_ids = [mid for mid in memcell_ids if mid not in valid_ids]

        if invalid_memcell_ids:
            # 显示前5个无效ID作为示例
            sample_size = min(5, len(invalid_memcell_ids))
            sample_ids = invalid_memcell_ids[:sample_size]
            if len(invalid_memcell_ids) > sample_size:
                logger.warning(
                    f"[validate_and_filter_memcell_ids] Filtered {len(invalid_memcell_ids)} non-existent memcell_ids. "
                    f"Examples: {sample_ids} (and {len(invalid_memcell_ids) - sample_size} more...)"
                )
            else:
                logger.warning(
                    f"[validate_and_filter_memcell_ids] Filtered {len(invalid_memcell_ids)} non-existent memcell_ids: {invalid_memcell_ids}"
                )

        # 第二步：如果需要，验证 participants
        if user_id is not None:
            if memcell_list is None:
                logger.error(
                    "[validate_and_filter_memcell_ids] user_id provided but memcell_list is None"
                )
                return valid_memcell_ids

            # 构建 memcell 参与者映射
            memcell_participants = {}
            for memcell in memcell_list:
                if hasattr(memcell, 'event_id'):
                    memcell_id = str(memcell.event_id)
                    participants = (
                        set(memcell.participants)
                        if hasattr(memcell, 'participants') and memcell.participants
                        else set()
                    )
                    memcell_participants[memcell_id] = participants

            # 过滤：只保留用户参与的 memcell
            participant_valid = []
            participant_invalid = []

            for memcell_id in valid_memcell_ids:
                # 理论上 memcell_id 一定在 memcell_participants 中，使用 get 兜底
                participants = memcell_participants.get(memcell_id, set())
                if user_id in participants:
                    participant_valid.append(memcell_id)
                else:
                    participant_invalid.append(memcell_id)

            if participant_invalid:
                sample_size = min(3, len(participant_invalid))
                sample_ids = participant_invalid[:sample_size]
                logger.warning(
                    f"[validate_and_filter_memcell_ids] User {user_id} not in participants of {len(participant_invalid)} memcells. "
                    f"Examples: {sample_ids}{'...' if len(participant_invalid) > sample_size else ''}"
                )

            return participant_valid

        return valid_memcell_ids

    def merge_memcell_ids(
        self,
        historical: Optional[List[str]],
        new: List[str],
        valid_ids: Set[str],
        memcell_list: List,
        user_id: Optional[str] = None,
        max_count: int = 50,
    ) -> List[str]:
        """
        合并历史和新的 memcell_ids。保持历史顺序不变，只对新的 memcell_ids 按时间戳排序。

        Args:
            historical: 历史 memcell_ids（不做验证，保持原有顺序）
            new: 新的 memcell_ids（需要验证，会按时间戳排序）
            valid_ids: 当前有效的 memcell_ids 集合（只用于验证新的 memcell_ids）
            memcell_list: 当前的 memcell 列表（用于获取时间戳进行排序）
            user_id: 可选，如果提供则验证用户是否在 participants 中（用于 roles）
            max_count: 最大保留数量

        Returns:
            合并、去重后的 memcell_ids（历史顺序不变，新的按时间排序追加，最多 max_count 个）
        """
        from common_utils.datetime_utils import get_now_with_timezone
        from ..group_profile_memory_extractor import convert_to_datetime

        historical = historical or []

        # 历史的 memcell_ids 直接保留，不做验证（因为对应的 memcell 不在当前输入中）
        # 只验证新的 memcell_ids（包括存在性和可选的 participants 检查）
        valid_new = self.validate_and_filter_memcell_ids(
            new, valid_ids, user_id=user_id, memcell_list=memcell_list
        )

        # 构建 memcell_id 到 timestamp 的映射（用于对新的 memcell_ids 排序）
        memcell_id_to_timestamp = {}
        for memcell in memcell_list:
            if hasattr(memcell, 'event_id') and hasattr(memcell, 'timestamp'):
                # 转换为字符串以匹配 LLM 输出的格式
                memcell_id = str(memcell.event_id)
                timestamp = convert_to_datetime(memcell.timestamp)
                memcell_id_to_timestamp[memcell_id] = timestamp

        # 对新的 memcell_ids 按时间戳排序（旧的在前，新的在后）
        valid_new_sorted = sorted(
            valid_new,
            key=lambda mid: memcell_id_to_timestamp.get(
                mid, get_now_with_timezone().replace(year=1900)
            ),
        )

        # 合并：保持历史顺序，追加新的（去重）
        seen = set(historical)  # 历史中已存在的 ID
        merged = list(historical)  # 保持历史顺序

        for mid in valid_new_sorted:
            if mid not in seen:
                merged.append(mid)
                seen.add(mid)

        # 限制数量（保留最新的，即列表后面的）
        if len(merged) > max_count:
            logger.debug(
                f"[merge_memcell_ids] Limiting from {len(merged)} to {max_count} memcell_ids "
                f"(historical: {len(historical)}, new: {len(valid_new)})"
            )
            merged = merged[-max_count:]

        return merged

    def get_comprehensive_speaker_mapping(
        self,
        memcell_list: List,
        existing_roles: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        获取综合的speaker映射，结合当前memcell和历史roles信息

        Args:
            memcell_list: 当前的memcell列表
            existing_roles: 历史roles信息，格式为 role -> [{"user_id": "xxx", "user_name": "xxx"}]

        Returns:
            speaker_id -> {"user_id": speaker_id, "user_name": speaker_name} 的映射
        """
        # 1. 从当前memcell构建映射
        current_mapping = {}
        for memcell in memcell_list:
            if hasattr(memcell, 'original_data') and memcell.original_data:
                for data in memcell.original_data:
                    speaker_id = data.get('speaker_id', '')
                    speaker_name = data.get('speaker_name', '')
                    if speaker_id and speaker_name:
                        current_mapping[speaker_id] = {
                            "user_id": speaker_id,
                            "user_name": speaker_name,
                        }

        # 2. 从历史roles中提取speaker映射信息
        historical_mapping = {}
        if existing_roles:
            for role, users in existing_roles.items():
                for user_info in users:
                    user_id = user_info.get("user_id", "")
                    user_name = user_info.get("user_name", "")
                    if (
                        user_id
                        and user_name
                        and user_id not in ["not_found", "unknown"]
                    ):
                        historical_mapping[user_id] = {
                            "user_id": user_id,
                            "user_name": user_name,
                        }

        # 3. 合并映射：当前优先，历史补充
        comprehensive_mapping = current_mapping.copy()
        for speaker_id, info in historical_mapping.items():
            if speaker_id not in comprehensive_mapping:
                comprehensive_mapping[speaker_id] = info

        return comprehensive_mapping

    def get_conversation_text(self, data_list: List[Any]) -> str:
        """Convert raw data to conversation text format."""
        lines = []
        for data in data_list:
            if hasattr(data, 'content'):
                speaker_name = data.content.get('speaker_name', '')
                speaker_id = data.content.get('speaker_id', '')
                speaker = (
                    f"{speaker_name}(user_id:{speaker_id})"
                    if speaker_id
                    else speaker_name
                )
                content = data.content.get('content')
            else:
                speaker_name = data.get('speaker_name', '')
                speaker_id = data.get('speaker_id', '')
                speaker = (
                    f"{speaker_name}(user_id:{speaker_id})"
                    if speaker_id
                    else speaker_name
                )
                content = data.get('content')

            if not content:
                continue
            # 不再包含时间戳，避免 LLM 混淆
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def get_episode_text(self, memcell) -> str:
        """Extract episode text from memcell."""
        if hasattr(memcell, 'episode') and memcell.episode:
            return memcell.episode
        return ""

    def combine_conversation_text_with_ids(self, memcell_list: List) -> str:
        """Combine conversation text with memcell IDs for evidence extraction."""
        all_conversation_text = []

        for memcell in memcell_list:
            # 确保 memcell_id 是字符串（处理 MongoDB ObjectId）
            raw_id = getattr(memcell, 'event_id', f'unknown_{id(memcell)}')
            memcell_id = str(raw_id)

            if self.conversation_source == "original":
                # 方式1：只用original_data（当前方式）
                conversation_text = self.get_conversation_text(memcell.original_data)
                # 使用更明显的分隔符，避免与时间戳的方括号混淆
                annotated_text = (
                    f"=== MEMCELL_ID: {memcell_id} ===\n{conversation_text}"
                )
                all_conversation_text.append(annotated_text)

            elif self.conversation_source == "episode":
                # 方式2：只用episode字段
                episode_text = self.get_episode_text(memcell)
                if episode_text:
                    annotated_text = f"=== MEMCELL_ID: {memcell_id} ===\n{episode_text}"
                    all_conversation_text.append(annotated_text)
                else:
                    # 如果没有episode，回退到original_data
                    logger.warning(
                        f"No episode found for memcell {memcell_id}, using original_data as fallback"
                    )
                    conversation_text = self.get_conversation_text(
                        memcell.original_data
                    )
                    annotated_text = f"=== MEMCELL_ID: {memcell_id} ===\n[FALLBACK] {conversation_text}"
                    all_conversation_text.append(annotated_text)

            else:
                raise ValueError(
                    f"Unsupported conversation_source: {self.conversation_source}"
                )

        return "\n\n".join(all_conversation_text)

    def extract_existing_group_profile(
        self, old_memory_list: Optional[List]
    ) -> Optional[Dict]:
        """
        Extract existing group profile from old memories.

        Extracts all topics/roles with their evidences and confidence.
        Returns separate fields for easier processing.
        """
        from datetime import datetime
        from api_specs.memory_types import MemoryType

        if not old_memory_list:
            return None

        for memory in old_memory_list:
            if memory.memory_type == MemoryType.GROUP_PROFILE:
                existing_topics = getattr(memory, "topics", [])
                # 确保不为 None
                if existing_topics is None:
                    existing_topics = []

                # Convert TopicInfo objects to dict, preserving evidences and confidence
                topics_list = []
                if existing_topics:
                    for topic in existing_topics:
                        if hasattr(topic, '__dict__'):
                            topic_dict = topic.__dict__.copy()
                            # Convert datetime to ISO string
                            if isinstance(topic_dict.get('last_active_at'), datetime):
                                topic_dict['last_active_at'] = topic_dict[
                                    'last_active_at'
                                ].isoformat()
                            topics_list.append(topic_dict)
                        elif isinstance(topic, dict):
                            topics_list.append(topic)

                # Roles already include evidences and confidence in new format
                existing_roles = getattr(memory, "roles", {})
                # 确保不为 None
                if existing_roles is None:
                    existing_roles = {}

                return {
                    "topics": topics_list,  # 包含 evidences 和 confidence
                    "summary": getattr(memory, "summary", ""),
                    "subject": getattr(memory, "subject", ""),
                    "roles": existing_roles,  # 包含 evidences 和 confidence
                }
        return None

    def get_user_name(
        self, user_id: str, speaker_mapping: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Get user name from comprehensive_speaker_mapping, fallback to user_id if not found"""
        if speaker_mapping and user_id in speaker_mapping:
            return speaker_mapping[user_id]["user_name"]
        return user_id
