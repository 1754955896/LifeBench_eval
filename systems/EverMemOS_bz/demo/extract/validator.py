"""结果验证器

验证提取结果是否正确存储到 MongoDB、Milvus、ES。
"""

from collections import Counter

from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from agentic_layer.vectorize_service import get_vectorize_service


class ResultValidator:
    """结果验证器"""

    def __init__(self, group_id: str):
        """初始化验证器

        Args:
            group_id: 群组 ID
        """
        self.group_id = group_id

    async def validate(self) -> None:
        """验证存储结果"""
        print("\n" + "=" * 80)
        print("验证存储结果")
        print("=" * 80)

        # 验证 MongoDB
        print("\n[MongoDB] 检查 MemCell")
        memcell_repo = get_bean_by_type(MemCellRawRepository)
        memcells = await memcell_repo.find_by_group_id(self.group_id, limit=1000)
        print(f"  - 找到 {len(memcells)} 个 MemCell")

        if memcells:
            total_semantic = sum(
                len(m.semantic_memories)
                for m in memcells
                if hasattr(m, 'semantic_memories') and m.semantic_memories
            )
            total_eventlog = sum(
                len(
                    m.event_log.get('atomic_fact', [])
                    if isinstance(m.event_log, dict)
                    else []
                )
                for m in memcells
                if hasattr(m, 'event_log') and m.event_log
            )

            print(f"  - episode: {len(memcells)} 个")
            print(f"  - semantic_memories: {total_semantic} 个")
            print(f"  - event_log atomic_facts: {total_eventlog} 个")

        # 验证 Milvus
        print("\n[Milvus] 检查记录")
        milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
        vectorize_service = get_vectorize_service()
        query_vector = await vectorize_service.get_embedding("测试")

        # 注意：limit 不能太大，Milvus HNSW 索引要求 ef >= k
        # 默认 ef=64，所以 limit 最多设置为 64
        milvus_results = await milvus_repo.vector_search(
            query_vector=query_vector,
            user_id="default",
            limit=50,  # 减小 limit，避免超过 ef 参数（默认 64）
        )
        print(f"  - 找到 {len(milvus_results)} 条记录")

        if milvus_results:
            types = [r.get('memory_types', 'unknown') for r in milvus_results]
            print(f"  - 类型分布: {dict(Counter(types))}")

        # 验证 ES
        print("\n[ES] 检查记录")
        es_repo = get_bean_by_type(EpisodicMemoryEsRepository)
        es_results = await es_repo.multi_search(query=[], user_id="default", size=1000)
        print(f"  - 找到 {len(es_results)} 条记录")

        if es_results:
            types = [r.get('_source', {}).get('type', 'unknown') for r in es_results]
            print(f"  - 类型分布: {dict(Counter(types))}")

        # 结果汇总
        print("\n" + "=" * 80)
        print("验证结果汇总")
        print("=" * 80)
        print(f"\n✅ MongoDB: {len(memcells)} 个")
        print(f"✅ Milvus: {len(milvus_results)} 条")
        print(f"✅ ES: {len(es_results)} 条")

        if len(memcells) > 0:
            print("\n🎉 提取和存储成功！")
        else:
            print("\n⚠️ 未找到记忆数据，请检查日志")
