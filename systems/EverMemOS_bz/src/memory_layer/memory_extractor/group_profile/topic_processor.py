"""Topic management utilities for group profile extraction."""

from typing import List, Dict, Set, Optional
from datetime import datetime
import uuid

from common_utils.datetime_utils import get_now_with_timezone, from_iso_format
from core.observation.logger import get_logger

logger = get_logger(__name__)


class TopicProcessor:
    """话题处理器 - 负责话题的增量更新和管理"""

    def __init__(self, data_processor):
        """
        初始化话题处理器

        Args:
            data_processor: GroupProfileDataProcessor 实例，用于验证和合并 memcell_ids
        """
        self.data_processor = data_processor

    def find_topic_to_replace(
        self, topics: List, reference_time: Optional[datetime] = None
    ) -> object:
        """
        找到要被替换的topic
        策略：
        1. 优先替换 30 天前已 implemented 的 topic
        2. 否则替换最老的 topic（无论状态）

        这样既能保护近期完成的重要项目，又能合理淘汰长期 implemented 的老旧 topics。

        Args:
            topics: TopicInfo 对象列表
            reference_time: 参考时间点（通常是 memcell 列表中的最晚时间）。
                           如果不提供，使用当前系统时间。

        Returns:
            要被替换的 TopicInfo 对象
        """
        from datetime import timedelta

        # 使用提供的参考时间，如果没有则使用当前时间
        now = reference_time if reference_time else get_now_with_timezone()
        threshold = now - timedelta(days=30)  # 30天阈值

        return sorted(
            topics,
            key=lambda t: (
                # 第一优先级：不是"老旧的implemented"（最新的和非implemented的排后面）
                not (
                    t.status == "implemented" and (t.last_active_at or now) < threshold
                ),
                # 第二优先级：按时间从旧到新
                t.last_active_at or get_now_with_timezone().replace(year=1900),
            ),
        )[0]

    def get_latest_memcell_timestamp(
        self, memcell_list: List, memcell_ids: Optional[List[str]] = None
    ) -> datetime:
        """
        Get the latest timestamp from memcell list.

        Args:
            memcell_list: List of all memcells
            memcell_ids: Optional list of memcell IDs to filter by.
                        If provided and not empty, only consider memcells with these IDs.
                        If not provided or empty, consider all memcells.

        Returns:
            Latest timestamp from (filtered) memcells, or current time if no valid timestamp found.
        """
        from ..group_profile_memory_extractor import convert_to_datetime

        # If memcell_ids provided and not empty, create a set for fast lookup
        filter_ids = set(memcell_ids) if memcell_ids else None

        latest_time = None
        matched_count = 0

        for memcell in memcell_list:
            # If filter_ids provided, only consider memcells in the filter
            # 转换为字符串以匹配 filter_ids 中的格式
            if filter_ids and hasattr(memcell, 'event_id'):
                memcell_id_str = str(memcell.event_id)
                if memcell_id_str not in filter_ids:
                    continue

            if hasattr(memcell, 'timestamp') and memcell.timestamp:
                matched_count += 1
                memcell_time = convert_to_datetime(memcell.timestamp)
                if latest_time is None or memcell_time > latest_time:
                    latest_time = memcell_time

        if filter_ids and matched_count > 0:
            logger.debug(
                f"[get_latest_memcell_timestamp] Found {matched_count} memcells matching {len(filter_ids)} IDs"
            )

        return latest_time if latest_time else get_now_with_timezone()

    def apply_topic_incremental_updates(
        self,
        llm_topics: List[Dict],
        existing_topics_with_evidences: List,  # 包含 evidences 的历史 topics
        memcell_list: List,
        valid_memcell_ids: Set[str],  # 有效的 memcell_ids
        max_topics: int = 5,
    ) -> List:
        """
        Apply incremental topic updates based on LLM output.

        Now handles evidences merging internally.

        Args:
            llm_topics: LLM 输出的话题列表（包含 evidences 和 confidence）
            existing_topics_with_evidences: 历史话题列表（包含 evidences）
            memcell_list: 当前的 memcell 列表
            valid_memcell_ids: 有效的 memcell_ids 集合
            max_topics: 最大话题数量

        Returns:
            处理后的 TopicInfo 对象列表（按 last_active_at 排序）
        """
        from ..group_profile_memory_extractor import TopicInfo

        # 计算 memcell 列表中的最晚时间作为参考时间点
        # 用于 topic 替换策略的时间判断（离线批处理场景）
        reference_time = self.get_latest_memcell_timestamp(memcell_list)

        # Parse existing topics (preserving evidences)
        existing_topics = []
        existing_topics_map = {}  # id -> topic_data_with_evidences

        for topic_data in existing_topics_with_evidences:
            if isinstance(topic_data, dict):
                last_active_str = topic_data.get("last_active_at", "")
                try:
                    if last_active_str:
                        last_active_at = from_iso_format(last_active_str)
                    else:
                        last_active_at = get_now_with_timezone()
                except Exception as e:
                    logger.warning(
                        f"Failed to parse last_active_at: {last_active_str}, error: {e}"
                    )
                    last_active_at = get_now_with_timezone()

                topic_id = topic_data.get(
                    "id", f"topic_{str(uuid.uuid4()).replace('-', '')[:8]}"
                )
                topic_info = TopicInfo(
                    id=topic_id,
                    name=topic_data.get("name", ""),
                    summary=topic_data.get("summary", ""),
                    status=topic_data.get("status", "exploring"),
                    last_active_at=last_active_at,
                    evidences=topic_data.get("evidences", []),
                    confidence=topic_data.get("confidence", "strong"),
                    update_type=topic_data.get("update_type") or "new",
                )
                existing_topics.append(topic_info)
                existing_topics_map[topic_id] = topic_data

        # Create topic dict for fast lookup
        topic_dict = {topic.id: topic for topic in existing_topics}

        # Process LLM topics
        for llm_topic in llm_topics:
            update_type = llm_topic.get("update_type") or "new"
            old_topic_id = llm_topic.get("old_topic_id")
            topic_name = llm_topic.get("name", "")

            # 获取 LLM 输出的 evidences 和 confidence
            llm_evidences = llm_topic.get("evidences", [])
            llm_confidence = llm_topic.get("confidence", "weak")

            if update_type == "update" and old_topic_id and old_topic_id in topic_dict:
                # Update existing topic - 合并历史和新的 evidences
                old_topic = topic_dict[old_topic_id]
                historical_evidences = old_topic.evidences or []

                # 合并 evidences（内部会验证）
                merged_evidences = self.data_processor.merge_memcell_ids(
                    historical=historical_evidences,
                    new=llm_evidences,
                    valid_ids=valid_memcell_ids,
                    memcell_list=memcell_list,
                    max_count=10,
                )

                # Calculate last_active_at based on merged evidences
                last_active_at = self.get_latest_memcell_timestamp(
                    memcell_list, merged_evidences
                )
                logger.debug(
                    f"[TopicIncremental] Topic '{topic_name}' has {len(merged_evidences)} valid evidences, "
                    f"confidence: {llm_confidence}, last_active_at: {last_active_at}"
                )

                updated_topic = TopicInfo(
                    id=old_topic.id,
                    name=llm_topic.get("name", old_topic.name),
                    summary=llm_topic.get("summary", old_topic.summary),
                    status=llm_topic.get("status", old_topic.status),
                    last_active_at=last_active_at,
                    evidences=merged_evidences,
                    confidence=llm_confidence,
                    update_type=update_type,
                )
                topic_dict[old_topic.id] = updated_topic
                logger.debug(
                    f"[TopicIncremental] Updated topic: {old_topic.id} -> {updated_topic.name}, "
                    f"evidences: {len(merged_evidences)}"
                )

            elif update_type == "new":
                # Add new topic - 验证 evidences
                valid_llm_evidences = (
                    self.data_processor.validate_and_filter_memcell_ids(
                        llm_evidences, valid_memcell_ids
                    )
                )

                # Calculate last_active_at based on evidences
                last_active_at = self.get_latest_memcell_timestamp(
                    memcell_list, valid_llm_evidences
                )
                logger.debug(
                    f"[TopicIncremental] New topic '{topic_name}' has {len(valid_llm_evidences)} valid evidences, "
                    f"confidence: {llm_confidence}, last_active_at: {last_active_at}"
                )

                new_id = f"topic_{str(uuid.uuid4()).replace('-', '')[:8]}"
                new_topic = TopicInfo(
                    id=new_id,
                    name=llm_topic.get("name", ""),
                    summary=llm_topic.get("summary", ""),
                    status=llm_topic.get("status", "exploring"),
                    last_active_at=last_active_at,
                    evidences=valid_llm_evidences,
                    confidence=llm_confidence,
                    update_type=update_type,
                )

                # If at max capacity, replace oldest/implemented topic
                if len(topic_dict) >= max_topics:
                    topic_to_replace = self.find_topic_to_replace(
                        list(topic_dict.values()), reference_time=reference_time
                    )
                    logger.debug(
                        f"[TopicIncremental] Replacing topic: {topic_to_replace.id} ({topic_to_replace.name}) "
                        f"with new topic: {new_topic.name}"
                    )
                    del topic_dict[topic_to_replace.id]

                topic_dict[new_id] = new_topic
                logger.debug(
                    f"[TopicIncremental] Added new topic: {new_id} -> {new_topic.name}"
                )

        # 按照 last_active_at 时间排序（最新的在前）
        final_topics = sorted(
            topic_dict.values(),
            key=lambda t: t.last_active_at or datetime.min,
            reverse=True,
        )
        return final_topics
