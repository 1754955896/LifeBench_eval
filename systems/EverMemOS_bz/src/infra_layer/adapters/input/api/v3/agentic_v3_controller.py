"""
Agentic Layer V3 控制器

提供专门用于处理群聊记忆的 RESTful API 路由
直接接收简单直接的消息格式，逐条处理并存储
"""

import logging
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import BaseController, post
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.memory_manager import MemoryManager
from api_specs.request_converter import handle_conversation_format
from api_specs.dtos.memory_query import ConversationMetaRequest, UserDetail
from infra_layer.adapters.input.api.mapper.group_chat_converter import (
    convert_simple_message_to_memorize_input,
)
from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
    ConversationMeta,
    UserDetailModel,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from component.redis_provider import RedisProvider

logger = logging.getLogger(__name__)


@controller("agentic_v3_controller", primary=True)
class AgenticV3Controller(BaseController):
    """
    Agentic Layer V3 API 控制器

    提供专门用于群聊记忆的接口：
    - memorize: 存储单条消息记忆
    - retrieve_lightweight: 轻量级检索（Embedding + BM25 + RRF）
    """

    def __init__(self, conversation_meta_repository: ConversationMetaRawRepository):
        """初始化控制器"""
        super().__init__(
            prefix="/api/v3/agentic",
            tags=["Agentic Layer V3"],
            default_auth="none",  # 根据实际需求调整认证策略
        )
        self.memory_manager = MemoryManager()
        self.conversation_meta_repository = conversation_meta_repository
        # 获取 RedisProvider
        self.redis_provider = get_bean_by_type(RedisProvider)
        logger.info(
            "AgenticV3Controller initialized with MemoryManager and ConversationMetaRepository"
        )

    @post(
        "/memorize",
        response_model=Dict[str, Any],
        summary="存储单条群聊消息记忆",
        description="""
        接收简单直接的单条消息格式并存储为记忆
        
        ## 功能说明：
        - 接收简单直接的单条消息数据（无需预转换）
        - 将单条消息提取为记忆单元（memcells）
        - 适用于实时消息处理场景
        - 返回已保存的记忆列表
        
        ## 输入格式（简单直接）：
        ```json
        {
          "group_id": "group_123",
          "group_name": "项目讨论组",
          "message_id": "msg_001",
          "create_time": "2025-01-15T10:00:00+08:00",
          "sender": "user_001",
          "sender_name": "张三",
          "content": "今天讨论下新功能的技术方案",
          "refer_list": ["msg_000"]
        }
        ```
        
        ## 字段说明：
        - **group_id** (可选): 群组ID
        - **group_name** (可选): 群组名称
        - **message_id** (必需): 消息ID
        - **create_time** (必需): 消息创建时间（ISO 8601格式）
        - **sender** (必需): 发送者用户ID
        - **sender_name** (可选): 发送者名称
        - **content** (必需): 消息内容
        - **refer_list** (可选): 引用的消息ID列表
        
        ## 与其他接口的区别：
        - **V3 /memorize**: 简单直接的单条消息格式（本接口，推荐）
        - **V2 /memorize**: 接收内部格式，需要外部转换
        
        ## 使用场景：
        - 实时消息流处理
        - 聊天机器人集成
        - 消息队列消费
        - 单条消息导入
        """,
        responses={
            200: {
                "description": "成功存储记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "记忆存储成功，共保存 1 条记忆",
                            "result": {
                                "saved_memories": [
                                    {
                                        "memory_type": "episode_summary",
                                        "user_id": "user_001",
                                        "group_id": "group_123",
                                        "timestamp": "2025-01-15T10:00:00",
                                        "content": "用户讨论了新功能的技术方案",
                                    }
                                ],
                                "count": 1,
                            },
                        }
                    }
                },
            },
            400: {
                "description": "请求参数错误",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "数据格式错误：缺少必需字段 message_id",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/memorize",
                        }
                    }
                },
            },
            500: {
                "description": "服务器内部错误",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "存储记忆失败，请稍后重试",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/memorize",
                        }
                    }
                },
            },
        },
    )
    async def memorize_single_message(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        存储单条消息记忆数据

        接收简单直接的单条消息格式，通过 group_chat_converter 转换并存储

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 记忆存储响应，包含已保存的记忆列表

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 1. 从请求中获取 JSON body（简单直接的格式）
            message_data = await fastapi_request.json()
            logger.info("收到 V3 memorize 请求（单条消息）")

            # 3. 使用 group_chat_converter 转换为内部格式
            logger.info("开始转换简单消息格式到内部格式")
            memorize_input = convert_simple_message_to_memorize_input(message_data)

            # 提取元信息用于日志
            group_name = memorize_input.get("group_name")
            group_id = memorize_input.get("group_id")

            logger.info("转换完成: group_id=%s, group_name=%s", group_id, group_name)

            # 4. 转换为 MemorizeRequest 对象并调用 memory_manager
            logger.info("开始处理记忆请求")
            memorize_request = await handle_conversation_format(memorize_input)
            memories = await self.memory_manager.memorize(memorize_request)

            # 5. 返回统一格式的响应
            memory_count = len(memories) if memories else 0
            logger.info("处理记忆请求完成，保存了 %s 条记忆", memory_count)

            # 优化返回信息，帮助用户理解运行状态
            if memory_count > 0:
                message = f"Extracted {memory_count} memories"
            else:
                message = "Message queued, awaiting boundary detection"

            return {
                "status": ErrorStatus.OK.value,
                "message": message,
                "result": {
                    "saved_memories": memories,
                    "count": memory_count,
                    "status_info": "accumulated" if memory_count == 0 else "extracted",
                },
            }

        except ValueError as e:
            logger.error("V3 memorize 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 memorize 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="存储记忆失败，请稍后重试"
            ) from e

    @post(
        "/retrieve_lightweight",
        response_model=Dict[str, Any],
        summary="轻量级记忆检索（Embedding + BM25 + RRF）",
        description="""
        轻量级记忆检索接口，使用 Embedding + BM25 + RRF 融合策略
        
        ## 功能说明：
        - 并行执行向量检索和关键词检索
        - 使用 RRF（Reciprocal Rank Fusion）融合结果
        - 速度快，适合实时场景
        
        ## 输入格式：
        ```json
        {
          "query": "北京旅游美食",
          "user_id": "default",
          "group_id": "assistant",
          "time_range_days": 365,
          "top_k": 20,
          "retrieval_mode": "rrf",
          "data_source": "episode",
        }
        ```
        
        ## 字段说明：
        - **query** (必需): 用户查询
        - **user_id** (可选): 用户ID（用于过滤）
        - **group_id** (可选): 群组ID（用于过滤）
        - **time_range_days** (可选): 时间范围天数（默认365天）
        - **top_k** (可选): 返回结果数量（默认20）
        - **retrieval_mode** (可选): 检索模式
          * "rrf": RRF 融合（默认）
          * "embedding": 纯向量检索
          * "bm25": 纯关键词检索
        - **data_source** (可选): 数据源
          * "episode": 从 MemCell.episode 检索（默认）
          * "event_log": 从 event_log.atomic_fact 检索
          * "semantic_memory": 从语义记忆检索
          * "profile": 仅需 user_id + group_id 的档案检索（query 可空）
        - **current_time** (可选): 当前时间，YYYY-MM-DD格式，用于过滤有效期内的语义记忆（仅 data_source=semantic_memory 时有效）
        - **radius** (可选): COSINE 相似度阈值，范围 [-1, 1]，默认 0.6
          * 只返回相似度 >= radius 的结果
          * 影响向量检索部分（embedding/rrf 模式）的结果质量
          * 对语义记忆和情景记忆有效（semantic_memory/episode），事件日志使用 L2 距离暂不支持
        
        ## 返回格式：
        ```json
        {
          "status": "ok",
          "message": "检索成功，找到 10 条记忆",
          "result": {
            "memories": [...],
            "count": 10,
            "metadata": {
              "retrieval_mode": "lightweight",
              "emb_count": 15,
              "bm25_count": 12,
              "final_count": 10,
              "total_latency_ms": 123.45
            }
          }
        }
        ```
        """,
    )
    async def retrieve_lightweight(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        轻量级记忆检索（Embedding + BM25 + RRF 融合）

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 检索结果响应
        """
        try:
            # 1. 解析请求参数
            request_data = await fastapi_request.json()
            query = request_data.get("query")
            user_id = request_data.get("user_id")
            group_id = request_data.get("group_id")
            time_range_days = request_data.get("time_range_days", 365)
            top_k = request_data.get("top_k", 20)
            retrieval_mode = request_data.get("retrieval_mode", "rrf")
            data_source = request_data.get("data_source", "episode")
            current_time_str = request_data.get("current_time")  # YYYY-MM-DD格式
            radius = request_data.get("radius")  # COSINE 相似度阈值（可选）

            if not query and data_source != "profile":
                raise ValueError("缺少必需参数：query")
            if data_source == "memcell":
                data_source = "episode"
            if data_source == "profile":
                if not user_id or not group_id:
                    raise ValueError(
                        "data_source=profile 时必须同时提供 user_id 和 group_id"
                    )

            # 解析 current_time
            from datetime import datetime

            current_time = None
            if current_time_str:
                try:
                    current_time = datetime.strptime(current_time_str, "%Y-%m-%d")
                except ValueError as e:
                    raise ValueError(
                        f"current_time 格式错误，应为 YYYY-MM-DD: {e}"
                    ) from e

            logger.info(
                f"收到 lightweight 检索请求: query={query}, group_id={group_id}, "
                f"mode={retrieval_mode}, source={data_source}"
                f"current_time={current_time_str}, top_k={top_k}"
            )

            # 2. 调用 memory_manager 的 lightweight 检索
            result = await self.memory_manager.retrieve_lightweight(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                data_source=data_source,
                current_time=current_time,
                radius=radius,
            )

            # 3. 返回统一格式
            return {
                "status": ErrorStatus.OK.value,
                "message": f"检索成功，找到 {result['count']} 条记忆",
                "result": result,
            }

        except ValueError as e:
            logger.error("V3 retrieve_lightweight 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("V3 retrieve_lightweight 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="检索失败，请稍后重试") from e

    @post(
        "/retrieve_agentic",
        response_model=Dict[str, Any],
        summary="Agentic 记忆检索（LLM 引导的多轮检索）",
        description="""
        Agentic 记忆检索接口，使用 LLM 引导的多轮智能检索
        
        ## 功能说明：
        - 使用 LLM 判断检索充分性
        - 自动进行多轮检索和查询改进
        - 使用 Rerank 提升结果质量
        - 适合需要深度理解的复杂查询
        
        ## 检索流程：
        1. Round 1: RRF 混合检索（Embedding + BM25）
        2. Rerank 优化结果
        3. LLM 判断是否充分
        4. 如果不充分：生成多个改进查询
        5. Round 2: 多查询并行检索
        6. 融合并 Rerank 返回最终结果
        
        ## 输入格式：
        ```json
        {
          "query": "用户喜欢吃什么？",
          "user_id": "default",
          "group_id": "assistant",
          "time_range_days": 365,
          "top_k": 20,
          "llm_config": {
            "api_key": "your_api_key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
          }
        }
        ```
        
        ## 字段说明：
        - **query** (必需): 用户查询
        - **user_id** (可选): 用户ID（用于过滤）
        - **group_id** (可选): 群组ID（用于过滤）
        - **time_range_days** (可选): 时间范围天数（默认365天）
        - **top_k** (可选): 返回结果数量（默认20）
        - **llm_config** (可选): LLM 配置
          * api_key: LLM API Key（可选，默认使用环境变量）
          * base_url: LLM API 地址（可选，默认 OpenRouter）
          * model: LLM 模型（可选，默认 gpt-4o-mini）
        
        ## 返回格式：
        ```json
        {
          "status": "ok",
          "message": "Agentic 检索成功，找到 15 条记忆",
          "result": {
            "memories": [...],
            "count": 15,
            "metadata": {
              "retrieval_mode": "agentic",
              "is_multi_round": true,
              "round1_count": 20,
              "is_sufficient": false,
              "reasoning": "需要更多关于饮食偏好的具体信息",
              "refined_queries": ["用户最喜欢的菜系？", "用户不喜欢吃什么？"],
              "round2_count": 40,
              "final_count": 15,
              "total_latency_ms": 2345.67
            }
          }
        }
        ```
        
        ## 使用场景：
        - 复杂问题回答
        - 深度信息挖掘
        - 多维度记忆检索
        - 智能对话系统
        
        ## 注意事项：
        - 需要配置 LLM API Key
        - 检索耗时较长（通常 2-5 秒）
        - 会产生 LLM API 调用费用
        """,
    )
    async def retrieve_agentic(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        Agentic 记忆检索（LLM 引导的多轮智能检索）

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 检索结果响应
        """
        try:
            # 1. 解析请求参数
            request_data = await fastapi_request.json()
            query = request_data.get("query")
            user_id = request_data.get("user_id")
            group_id = request_data.get("group_id")
            time_range_days = request_data.get("time_range_days", 365)
            top_k = request_data.get("top_k", 20)
            llm_config = request_data.get("llm_config", {})

            if not query:
                raise ValueError("缺少必需参数：query")

            logger.info(
                f"收到 agentic 检索请求: query={query}, group_id={group_id}, top_k={top_k}"
            )

            # 2. 创建 LLM Provider
            from memory_layer.llm.llm_provider import LLMProvider
            import os

            # 从请求或环境变量获取配置
            api_key = llm_config.get("api_key") or os.getenv("LLM_API_KEY")
            base_url = llm_config.get("base_url") or os.getenv("LLM_BASE_URL")
            model = llm_config.get("model") or os.getenv("LLM_MODEL")

            if not api_key:
                raise ValueError(
                    "缺少 LLM API Key，请在 llm_config.api_key 中提供或设置环境变量 OPENROUTER_API_KEY/OPENAI_API_KEY"
                )

            # 创建 LLM Provider（使用 OpenAI 兼容接口）
            llm_provider = LLMProvider(
                provider_type="openai",
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.3,
                max_tokens=2048,
            )

            logger.info(f"使用 LLM: {model} @ {base_url}")

            # 3. 调用 memory_manager 的 agentic 检索
            result = await self.memory_manager.retrieve_agentic(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=top_k,
                llm_provider=llm_provider,
                agentic_config=None,  # 使用默认配置
            )

            # 4. 返回统一格式
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Agentic 检索成功，找到 {result['count']} 条记忆",
                "result": result,
            }

        except ValueError as e:
            logger.error("V3 retrieve_agentic 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("V3 retrieve_agentic 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Agentic 检索失败，请稍后重试"
            ) from e

    @post(
        "/conversation-meta",
        response_model=Dict[str, Any],
        summary="保存对话元数据",
        description="""
        保存对话的元数据信息，包括场景、参与者、标签等
        """,
    )
    async def save_conversation_meta(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        保存对话元数据

        接收 ConversationMetaRequest 格式的数据，转换为 ConversationMeta ODM 模型并保存到 MongoDB

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 保存响应，包含已保存的元数据信息

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 1. 从请求中获取 JSON body
            request_data = await fastapi_request.json()
            logger.info(
                "收到 V3 conversation-meta 保存请求: group_id=%s",
                request_data.get("group_id"),
            )

            # 2. 解析为 ConversationMetaRequest
            # 处理 user_details 的转换
            user_details_data = request_data.get("user_details", {})
            user_details = {}
            for user_id, detail_data in user_details_data.items():
                user_details[user_id] = UserDetail(
                    full_name=detail_data["full_name"],
                    role=detail_data["role"],
                    extra=detail_data.get("extra", {}),
                )

            conversation_meta_request = ConversationMetaRequest(
                version=request_data["version"],
                scene=request_data["scene"],
                scene_desc=request_data["scene_desc"],
                name=request_data["name"],
                description=request_data["description"],
                group_id=request_data["group_id"],
                created_at=request_data["created_at"],
                default_timezone=request_data["default_timezone"],
                user_details=user_details,
                tags=request_data.get("tags", []),
            )

            logger.info(
                "解析 ConversationMetaRequest 成功: group_id=%s",
                conversation_meta_request.group_id,
            )

            # 3. 转换为 ConversationMeta ODM 模型
            user_details_model = {}
            for user_id, detail in conversation_meta_request.user_details.items():
                user_details_model[user_id] = UserDetailModel(
                    full_name=detail.full_name, role=detail.role, extra=detail.extra
                )

            conversation_meta = ConversationMeta(
                version=conversation_meta_request.version,
                scene=conversation_meta_request.scene,
                scene_desc=conversation_meta_request.scene_desc,
                name=conversation_meta_request.name,
                description=conversation_meta_request.description,
                group_id=conversation_meta_request.group_id,
                conversation_created_at=conversation_meta_request.created_at,
                default_timezone=conversation_meta_request.default_timezone,
                user_details=user_details_model,
                tags=conversation_meta_request.tags,
            )

            # 4. 使用 upsert 方式保存（如果 group_id 已存在则更新）
            logger.info("开始保存对话元数据到 MongoDB")
            saved_meta = await self.conversation_meta_repository.upsert_by_group_id(
                group_id=conversation_meta.group_id,
                conversation_data={
                    "version": conversation_meta.version,
                    "scene": conversation_meta.scene,
                    "scene_desc": conversation_meta.scene_desc,
                    "name": conversation_meta.name,
                    "description": conversation_meta.description,
                    "conversation_created_at": conversation_meta.conversation_created_at,
                    "default_timezone": conversation_meta.default_timezone,
                    "user_details": conversation_meta.user_details,
                    "tags": conversation_meta.tags,
                },
            )

            if not saved_meta:
                raise HTTPException(status_code=500, detail="保存对话元数据失败")

            logger.info(
                "保存对话元数据成功: id=%s, group_id=%s",
                saved_meta.id,
                saved_meta.group_id,
            )

            # 5. 返回成功响应
            return {
                "status": ErrorStatus.OK.value,
                "message": "对话元数据保存成功",
                "result": {
                    "id": str(saved_meta.id),
                    "group_id": saved_meta.group_id,
                    "scene": saved_meta.scene,
                    "name": saved_meta.name,
                    "version": saved_meta.version,
                    "created_at": (
                        saved_meta.created_at.isoformat()
                        if saved_meta.created_at
                        else None
                    ),
                    "updated_at": (
                        saved_meta.updated_at.isoformat()
                        if saved_meta.updated_at
                        else None
                    ),
                },
            }

        except KeyError as e:
            logger.error("V3 conversation-meta 请求缺少必需字段: %s", e)
            raise HTTPException(
                status_code=400, detail=f"缺少必需字段: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("V3 conversation-meta 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 conversation-meta 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="保存对话元数据失败，请稍后重试"
            ) from e
