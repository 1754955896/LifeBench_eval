"""
Zep Cloud Adapter for LifeBench_eval.

API Reference: https://docs.getzep.com/

Zep Cloud uses sync SDK - all calls wrapped with asyncio.to_thread().
Blocking serial execution with per-request latency tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)

from zep_cloud.client import Zep
from zep_cloud.types import Message as ZepMessage


@dataclass
class RequestResult:
    """Result of a single request with timing info."""
    success: bool
    latency_ms: float
    error: Optional[str] = None


class AddMessagesResult:
    """Result of add_messages with optional task_id for polling."""
    message_uuids: List[str]
    task_id: Optional[str] = None


async def _poll_task_completion(
    client: Zep,
    task_id: str,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
) -> bool:
    """
    Poll task status until completed or failed.

    Args:
        client: Zep client
        task_id: Task ID returned from add_messages or other async operations
        poll_interval: Seconds between polls (default 2.0)
        timeout: Max seconds to wait (default 300.0 = 5 minutes)

    Returns:
        True if task succeeded, False if failed or timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            task_result = await asyncio.to_thread(
                client.task.get, task_id=task_id
            )
            status = task_result.status if task_result else None

            if status == "succeeded":
                return True
            elif status == "failed":
                logger.warning(f"Task {task_id} failed: {task_result}")
                return False

            # Still pending/processing, wait before next poll
            await asyncio.sleep(poll_interval)

        except Exception as e:
            logger.warning(f"Error polling task {task_id}: {e}")
            await asyncio.sleep(poll_interval)

    logger.warning(f"Task {task_id} timed out after {timeout}s")
    return False


@register_adapter("zep_cloud")
class ZepCloudAdapter(BaseAdapter):
    """
    Zep Cloud API adapter with blocking serial execution and per-request latency.

    Key operations:
    - user.add() - Create/upsert user
    - thread.create() - Create session thread
    - thread.add_messages() - Add messages with timestamps
    - thread.get_user_context() - Retrieve memories

    Configuration:
        api_key: Zep Cloud API key (required)
        max_retries: Maximum retry attempts (default 3)
        retry_delay: Base delay in seconds between retries (default 2.0)
        timeout: HTTP request timeout in seconds (default 60.0)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.api_key = config.get("api_key", "")
        if not self.api_key:
            raise ValueError("Zep Cloud API key is required. Set 'api_key' in config.")

        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.timeout = config.get("timeout", 60.0)

        self._client: Optional[Zep] = None
        self._user_id_prefix = config.get("user_id_prefix", "lifebench")
        self._users_created: set = set()

    def _get_client(self) -> Zep:
        """Get or create Zep client (sync, called from thread pool)."""
        if self._client is None:
            self._client = Zep(api_key=self.api_key)
        return self._client

    async def _sync_call(self, func, *args, **kwargs):
        """Run a sync function in a thread pool to avoid blocking the event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def prepare(self, conversations: List, **kwargs) -> None:
        """
        Pre-create users and threads for all conversations.

        Args:
            conversations: Standard format conversation list
        """
        for conv in conversations:
            conversation_id = conv.conversation_id
            user_id = f"{self._user_id_prefix}_{conversation_id}"

            if user_id not in self._users_created:
                try:
                    await self._sync_call(
                        self._get_client().user.add,
                        user_id=user_id,
                    )
                    self._users_created.add(user_id)
                    logger.info(f"Created/verified user: {user_id}")
                except Exception as exc:
                    logger.warning(f"User creation warning for {user_id}: {exc}")

            try:
                await self._sync_call(
                    self._get_client().thread.create,
                    thread_id=conversation_id,
                    user_id=user_id,
                )
                logger.info(f"Created/verified thread: {conversation_id}")
            except Exception as exc:
                logger.warning(f"Thread creation warning for {conversation_id}: {exc}")

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """
        Add message chunks to Zep Cloud threads with blocking per-request latency.

        Each chunk is sent after the previous one completes, ensuring clean latency.
        If the Zep API returns a task_id, this method polls until processing
        completes to measure true end-to-end latency.
        """
        latencies = []
        total_added = 0
        total_failed = 0
        errors = []
        poll_timeout = kwargs.get("poll_timeout", 300.0)  # 5 min default

        for i, chunk in enumerate(chunks):
            if not chunk.messages:
                continue

            conversation_id = chunk.conversation_id
            user_id = f"{self._user_id_prefix}_{conversation_id}"

            zep_messages = []
            for msg in chunk.messages:
                role = "user" if "assistant" not in msg.speaker_name.lower() else "assistant"

                if msg.timestamp:
                    if isinstance(msg.timestamp, datetime):
                        created_at = msg.timestamp.isoformat()
                    else:
                        created_at = datetime.fromtimestamp(
                            msg.timestamp, tz=timezone.utc
                        ).isoformat()
                else:
                    created_at = datetime.now(timezone.utc).isoformat()

                zep_messages.append(ZepMessage(
                    created_at=created_at,
                    name=msg.speaker_name,
                    role=role,
                    content=msg.content,
                ))

            t0 = time.perf_counter()
            task_id = None
            try:
                # Capture response to get message_uuids and potentially task_id
                response = await self._sync_call(
                    self._get_client().thread.add_messages,
                    thread_id=conversation_id,
                    messages=zep_messages,
                )

                # Check if response has task_id for polling
                if hasattr(response, "task_id") and response.task_id:
                    task_id = response.task_id
                    logger.info(
                        f"ADD: thread={conversation_id}, msg_count={len(zep_messages)}, "
                        f"task_id={task_id} - polling for completion"
                    )
                elif hasattr(response, "message_uuids") and response.message_uuids:
                    logger.info(
                        f"ADD: thread={conversation_id}, msg_count={len(zep_messages)}, "
                        f"message_uuids={response.message_uuids[:3]}... - no task_id available"
                    )

                # If we have a task_id, poll until completion for true latency
                if task_id:
                    success = await _poll_task_completion(
                        self._get_client(),
                        task_id,
                        poll_interval=2.0,
                        timeout=poll_timeout,
                    )
                    if not success:
                        errors.append({
                            "chunk_index": i,
                            "error": f"Task {task_id} failed or timed out"
                        })
                        total_failed += 1
                        latencies.append((time.perf_counter() - t0) * 1000)
                        continue

                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)
                total_added += 1
                logger.info(
                    f"ADD: thread={conversation_id}, msg_count={len(zep_messages)}, "
                    f"latency={latency_ms:.2f}ms"
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)
                total_failed += 1
                errors.append({"chunk_index": i, "error": str(exc)[:200]})
                logger.warning(
                    f"ADD failed for thread={conversation_id}: {exc}"
                )

        return {
            "type": "zep_cloud",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
            "latencies_ms": latencies,
            "errors": errors,
            "latency_stats": {
                "count": len(latencies),
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
                "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
            },
        }

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """
        Search memories via Zep Cloud thread.get_user_context() with latency tracking.
        """
        top_k = kwargs.get("top_k", 10)

        t0 = time.perf_counter()
        try:
            user_context = await self._sync_call(
                self._get_client().thread.get_user_context,
                thread_id=conversation_id,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            context_block = user_context.context
            results = []

            if context_block:
                results.append(RetrievedMemory(
                    content=context_block,
                    score=1.0,
                    metadata={
                        "source": "zep_user_context",
                        "conversation_id": conversation_id,
                    },
                ))

            results = results[:top_k]

            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=results,
                retrieval_metadata={
                    "adapter": "zep_cloud",
                    "total_results": len(results),
                    "latency_ms": latency_ms,
                }
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"SEARCH failed for conversation={conversation_id}: {exc}")
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "adapter": "zep_cloud",
                    "error": str(exc)[:200],
                    "latency_ms": latency_ms,
                }
            )

    async def close(self) -> None:
        """Cleanup resources."""
        if self._client:
            self._client = None
