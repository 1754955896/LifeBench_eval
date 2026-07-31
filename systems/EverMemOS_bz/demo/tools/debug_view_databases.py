#!/usr/bin/env python3
"""
数据库数据查看工具

用法：
    python debug_view_databases.py                    # 查看所有数据库概览
    python debug_view_databases.py --mongo            # 只查看 MongoDB
    python debug_view_databases.py --milvus           # 只查看 Milvus
    python debug_view_databases.py --es               # 只查看 Elasticsearch
    python debug_view_databases.py --detail           # 显示详细数据（包括样例）
"""

import asyncio
import os
import sys
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymilvus import connections, Collection, utility
from elasticsearch import AsyncElasticsearch
import json


class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}--- {title} ---{Colors.END}")


def print_success(text: str):
    """打印成功信息"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


async def check_mongodb(detail: bool = False):
    """检查 MongoDB 数据"""
    print_section("MongoDB 数据")
    
    try:
        # 连接配置（无认证）
        mongo_host = os.getenv('MONGO_HOST', 'localhost')
        mongo_port = int(os.getenv('MONGO_PORT', '27017'))
        
        client = AsyncIOMotorClient(f'mongodb://{mongo_host}:{mongo_port}')
        db = client['memsys']
        
        print_success(f"已连接到 MongoDB: {mongo_host}:{mongo_port}")
        
        # 动态获取所有集合
        collection_names = await db.list_collection_names()
        collection_names.sort()
        
        if not collection_names:
            print_warning("数据库中没有集合")
            return

        for collection_name in collection_names:
            # 跳过系统集合
            if collection_name.startswith("system."):
                continue

            print_subsection(f"集合: {collection_name}")
            
            collection = db[collection_name]
            total = await collection.count_documents({})
            
            if total == 0:
                print_warning(f"无数据")
                continue
            
            print(f"总数: {Colors.BOLD}{total}{Colors.END} 条")
            
            # 按 user_id 统计 (如果存在)
            try:
                pipeline = [
                    {'$group': {'_id': '$user_id', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}}
                ]
                # 检查是否有 user_id 字段（简单采样检查）
                sample = await collection.find_one()
                if sample and 'user_id' in sample:
                    result = await collection.aggregate(pipeline).to_list(None)
                    
                    if result:
                        print("\n按 user_id 分组:")
                        for item in result[:10]:  # 只显示前 10 个
                            user_id = item['_id'] if item['_id'] else '(空/群组)'
                            print(f"  - {user_id}: {item['count']} 条")
            except Exception:
                pass # 忽略聚合错误
            
            # 显示样例
            if detail:
                print("\n样例数据:")
                cursor = collection.find().limit(2)
                async for doc in cursor:
                    # 移除过长的字段
                    doc.pop('_id', None)
                    doc.pop('vector', None)
                    doc.pop('embedding', None)
                    doc.pop('original_data', None)
                    
                    # 限制字段长度
                    for key, value in doc.items():
                        if isinstance(value, str) and len(value) > 100:
                            doc[key] = value[:100] + '...'
                    
                    print(f"  {json.dumps(doc, ensure_ascii=False, indent=2, default=str)}")
        
        client.close()
        
    except Exception as e:
        print_error(f"MongoDB 连接失败: {e}")


def _get_milvus_row_count(collection_name: str, collection: Collection) -> int:
    """
    获取 Milvus 实时实体数。

    优先使用 utility.get_collection_stats (如果该 API 可用)，
    其次尝试 utility.get_query_segment_info，最后回退到 num_entities。
    """
    get_stats = getattr(utility, "get_collection_stats", None)
    if callable(get_stats):
        stats_info = get_stats(collection_name)
        if isinstance(stats_info, dict):
            return int(stats_info.get("row_count", 0))

    # 部分老版本没有 get_collection_stats，退而求其次汇总 segment 行数
    segment_infos = utility.get_query_segment_info(collection_name)
    if segment_infos:
        total_rows = 0
        for seg in segment_infos:
            num_rows = getattr(seg, "num_rows", None)
            if num_rows is None:
                num_rows = getattr(seg, "row_count", 0)
            total_rows += int(num_rows or 0)
        return total_rows

    # 最终兜底：返回 num_entities（可能包含已删除数据）
    return collection.num_entities


def check_milvus(detail: bool = False):
    """检查 Milvus 数据"""
    print_section("Milvus 数据")
    
    try:
        # 连接配置
        milvus_host = os.getenv('MILVUS_HOST', 'localhost')
        milvus_port = int(os.getenv('MILVUS_PORT', '19530'))
        
        connections.connect(host=milvus_host, port=milvus_port)
        print_success(f"已连接到 Milvus: {milvus_host}:{milvus_port}")
        
        # 动态获取所有集合
        all_collections = utility.list_collections()
        all_collections.sort()
        
        if not all_collections:
            print_warning("Milvus 中没有集合")
            return
        
        for collection_name in all_collections:
            print_subsection(f"Collection: {collection_name}")
            
            collection = Collection(collection_name)
            collection.load()
            
            stats = _get_milvus_row_count(collection_name, collection)
            print(f"当前实体数: {Colors.BOLD}{stats}{Colors.END} 条")
            
            # 查询样例
            if detail and stats > 0:
                print("\n样例数据:")
                
                # 原有的查询逻辑依赖 'id' 字段，如果集合主键不是 'id' 会报错
                # 我们暂时仅保留计数功能，或者仅对已知集合查样例
                known_pk_map = {
                    'episodic_memory_memsys': 'id',
                    'semantic_memory_memsys': 'id',
                    'event_log_memsys': 'id'
                }
                
                pk_field = known_pk_map.get(collection_name)
                if pk_field:
                     try:
                        results = collection.query(
                            expr=f"{pk_field} >= 0",
                            output_fields=["user_id", "timestamp", "episode", "atomic_fact", "content", "semantic"],
                            limit=2
                        )
                        for result in results:
                            for key, value in result.items():
                                if isinstance(value, str) and len(value) > 100:
                                    result[key] = value[:100] + '...'
                            print(f"  {json.dumps(result, ensure_ascii=False, indent=2, default=str)}")
                     except Exception as e:
                         print(f"  查询样例失败: {e}")
                else:
                    print("  (未配置主键字段，跳过样例查询)")

        
        connections.disconnect("default")
        
    except Exception as e:
        print_error(f"Milvus 连接失败: {e}")


async def check_elasticsearch(detail: bool = False):
    """检查 Elasticsearch 数据"""
    print_section("Elasticsearch 数据")
    
    try:
        # 连接配置
        es_host = os.getenv('ES_HOSTS', 'http://localhost:19200')
        
        # 使用 HTTP 直接查询，避免版本兼容问题
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            print_success(f"已连接到 Elasticsearch: {es_host}")
            
            # 获取所有索引
            async with session.get(f"{es_host}/_cat/indices?format=json") as resp:
                if resp.status != 200:
                    print_error(f"获取索引列表失败: {resp.status}")
                    return
                indices = await resp.json()
            
            # 显示所有索引（不过滤）
            relevant_indices = indices
            
            if not relevant_indices:
                print_warning("没有找到相关索引")
                return
            
            # 按索引名排序
            relevant_indices.sort(key=lambda x: x['index'])
            
            for idx_info in relevant_indices:
                idx_name = idx_info['index']
                # 跳过系统索引
                if idx_name.startswith('.'):
                    continue
                    
                doc_count = int(idx_info['docs.count'])
                
                print_subsection(f"{idx_name} ({doc_count} 条)")
                
                if doc_count == 0:
                    print_warning("无数据")
                    continue
                
                # 使用聚合查询统计所有 type
                try:
                    aggs_body = {
                        "size": 0,
                        "aggs": {
                            "types": {
                                "terms": {
                                    "field": "type",
                                    "size": 50
                                }
                            }
                        }
                    }
                    async with session.post(
                        f"{es_host}/{idx_name}/_search",
                        json=aggs_body,
                        headers={"Content-Type": "application/json"}
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            buckets = result.get('aggregations', {}).get('types', {}).get('buckets', [])
                            if buckets:
                                for bucket in buckets:
                                    print(f"  - type={bucket['key']}: {bucket['doc_count']} 条")
                            else:
                                print(f"  - (无 type 字段分类)")
                except Exception as e:
                    print_warning(f"聚合统计失败: {e}")
                
                # 显示样例
                if detail and doc_count > 0:
                    print("\n样例数据:")
                    try:
                        query_body = {
                            "size": 2
                        }
                        async with session.post(
                            f"{es_host}/{idx_name}/_search",
                            json=query_body,
                            headers={"Content-Type": "application/json"}
                        ) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                
                                for hit in result['hits']['hits']:
                                    src = hit['_source']
                                    
                                    # 限制字段长度
                                    for key, value in src.items():
                                        if isinstance(value, str) and len(value) > 100:
                                            src[key] = value[:100] + '...'
                                    
                                    print(f"  {json.dumps(src, ensure_ascii=False, indent=2, default=str)}")
                    except Exception as e:
                        print_warning(f"查询样例失败: {e}")
        
    except Exception as e:
        print_error(f"Elasticsearch 连接失败: {e}")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='查看数据库数据')
    parser.add_argument('--mongo', action='store_true', help='只查看 MongoDB')
    parser.add_argument('--milvus', action='store_true', help='只查看 Milvus')
    parser.add_argument('--es', action='store_true', help='只查看 Elasticsearch')
    parser.add_argument('--detail', action='store_true', help='显示详细数据（包括样例）')
    
    args = parser.parse_args()
    
    # 加载 .env 文件（如果存在）
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}数据库数据查看工具{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"\n{Colors.CYAN}时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # 如果没有指定任何数据库，则查看所有
    if not (args.mongo or args.milvus or args.es):
        args.mongo = args.milvus = args.es = True
    
    if args.mongo:
        await check_mongodb(args.detail)
    
    if args.milvus:
        check_milvus(args.detail)
    
    if args.es:
        await check_elasticsearch(args.detail)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}✅ 完成{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.END}\n")


if __name__ == '__main__':
    asyncio.run(main())
