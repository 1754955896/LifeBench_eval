"""
⚠️ 危险操作：彻底重置所有数据库结构 ⚠️

这个脚本会执行毁灭性操作：
1. MongoDB: 直接删除整个 memsys 数据库
2. Milvus: Drop 所有集合
3. Elasticsearch: 删除所有名称包含 'memsys' 的索引（包括旧的、新的、别名）
4. Redis: 清空数据库

使用场景：
- 环境数据脏乱，需要彻底重来
- 修复了 Schema，需要重建索引结构

使用后必须重启服务以重新初始化结构。
"""
import asyncio
import sys
import os

# 确保能导入项目模块
sys.path.append(os.getcwd())

from pymilvus import utility, connections
from core.di import get_bean_by_type
from component.redis_provider import RedisProvider
from component.mongodb_client_factory import MongoDBClientFactory
from component.elasticsearch_client_factory import ElasticsearchClientFactory
from bootstrap import setup_project_context


async def reset_mongodb():
    print("🔥 [MongoDB] 正在删除数据库...")
    try:
        factory = get_bean_by_type(MongoDBClientFactory)
        client_wrapper = await factory.get_default_client()
        # Beanie 的 client 属性就是 MotorClient
        motor_client = client_wrapper.client
        db_name = client_wrapper.database.name
        # 直接删除整个数据库
        await motor_client.drop_database(db_name)
        print(f"   ✅ 已删除数据库: {db_name}")
    except Exception as e:
        print(f"   ❌ MongoDB 重置失败: {e}")


def reset_milvus():
    print("🔥 [Milvus] 正在 Drop 所有集合...")
    try:
        # 连接配置
        milvus_host = os.getenv('MILVUS_HOST', 'localhost')
        milvus_port = int(os.getenv('MILVUS_PORT', '19530'))
        connections.connect(host=milvus_host, port=milvus_port)
        
        collections = utility.list_collections()
        if not collections:
            print("   ⚪ 没有发现集合")
            return

        for name in collections:
            utility.drop_collection(name)
            print(f"   ✅ Dropped collection: {name}")
            
    except Exception as e:
        print(f"   ❌ Milvus 重置失败: {e}")


async def reset_elasticsearch():
    print("🔥 [Elasticsearch] 正在删除所有相关索引...")
    try:
        factory = get_bean_by_type(ElasticsearchClientFactory)
        client_wrapper = await factory.get_default_client()
        es = client_wrapper.async_client
        
        # 删除所有包含 memsys 的索引
        target_pattern = "*memsys*"
        
        # 1. 获取具体索引列表
        indices_resp = await es.cat.indices(index=target_pattern, format="json")
        
        if not indices_resp:
            print(f"   ⚪ 没有发现匹配 '{target_pattern}' 的索引")
            return

        # 2. 提取索引名列表
        index_names = [item['index'] for item in indices_resp]
        count = len(index_names)
        
        # 3. 显式删除这些索引
        # 使用逗号分隔的字符串或列表
        await es.indices.delete(index=list(index_names), ignore=[404])
        print(f"   ✅ 已删除 {count} 个索引: {', '.join(index_names[:3])}...")
        
    except Exception as e:
        print(f"   ❌ Elasticsearch 重置失败: {e}")


async def reset_redis():
    print("🔥 [Redis] 正在 FlushDB...")
    try:
        provider = get_bean_by_type(RedisProvider)
        client = await provider.get_client()
        await client.flushdb()
        print("   ✅ Redis 已清空")
    except Exception as e:
        print(f"   ❌ Redis 重置失败: {e}")


async def main():
    print("\n" + "="*60)
    print("🧨 数据库彻底重置工具 🧨")
    print("="*60 + "\n")
    
    await setup_project_context()
    
    await reset_mongodb()
    reset_milvus()
    await reset_elasticsearch()
    await reset_redis()
    
    print("\n" + "="*60)
    print("✨ 重置完成！请立即重启服务以重建索引结构 ✨")
    print("="*60 + "\n")


if __name__ == "__main__":
    # 加载 .env
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main())

