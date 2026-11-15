"""
Simple Boundary Detection Base Class for EverMemOS

This module provides a simple and extensible base class for detecting
boundaries in various types of content (conversations, emails, notes, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from memory_layer.llm.llm_provider import LLMProvider
from api_specs.memory_types import RawDataType, Memory, MemCell
from api_specs.dtos.memory_command import RawData


@dataclass
class MemCellExtractRequest:
    history_raw_data_list: List[RawData]
    new_raw_data_list: List[RawData]
    # 整个群的user id
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None

    old_memory_list: Optional[List[Memory]] = None
    smart_mask_flag: Optional[bool] = False


@dataclass
class StatusResult:
    """Status control result."""

    # 表示下次触发时，这次的对话会累积一起作为new message输入
    should_wait: bool


class MemCellExtractor(ABC):
    def __init__(self, raw_data_type: RawDataType, llm_provider=LLMProvider):
        self.raw_data_type = raw_data_type
        self._llm_provider = llm_provider

    @abstractmethod
    async def extract_memcell(
        self, request: MemCellExtractRequest
    ) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        pass
