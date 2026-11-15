"""Agentic 检索完整测试脚本

完全按照 test_retrieval_comprehensive.py 的思路：
- 多场景测试（个人、群组、多维度）
- 统计汇总（成功率、multi-round 比例、平均耗时）
- 结果导出到 JSON

使用方法：
    # 确保 API 服务器已启动
    uv run python src/bootstrap.py src/run.py --port 8001
    
    # 在另一个终端运行测试
    uv run python src/bootstrap.py demo/tools/test_agentic_retrieve.py
    
    # 或使用 Mock LLM（不需要真实 API Key）
    uv run python src/bootstrap.py demo/tools/test_agentic_retrieve.py --mock-llm
"""

import asyncio
import httpx
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from aiohttp import web
except ImportError:
    web = None


class AgenticRetrievalTester:
    """Agentic 检索测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """初始化测试器
        
        Args:
            base_url: API 服务器地址
        """
        self.base_url = base_url
        self.retrieve_url = f"{base_url}/api/v3/agentic/retrieve_agentic"
        
        # 测试结果统计
        self.total_tests = 0
        self.successful_tests = 0
        self.failed_tests = 0
        self.test_results = []
        
        # Agentic 特有统计
        self.multi_round_count = 0
        self.sufficient_count = 0
        
    async def test_agentic(
        self,
        query: str,
        user_id: str,
        group_id: str = None,
        top_k: int = 10,
        time_range_days: int = 365,
        llm_config: Dict[str, Any] = None,
        test_name: str = None,
    ) -> Dict[str, Any]:
        """执行单次 Agentic 检索测试
        
        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID
            top_k: 返回结果数量
            time_range_days: 时间范围（天）
            llm_config: LLM 配置
            test_name: 测试名称
            
        Returns:
            测试结果字典
        """
        self.total_tests += 1
        
        if test_name is None:
            test_name = f"agentic_{self.total_tests}"
        
        # 构建请求参数
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "time_range_days": time_range_days,
            "llm_config": llm_config or {},
        }
        
        if group_id:
            payload["group_id"] = group_id
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.retrieve_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "ok":
                    memories = result.get("result", {}).get("memories", [])
                    metadata = result.get("result", {}).get("metadata", {})
                    
                    # 提取 Agentic 特有信息
                    is_multi_round = metadata.get("is_multi_round", False)
                    is_sufficient = metadata.get("is_sufficient", None)
                    reasoning = metadata.get("reasoning", "")
                    missing_info = metadata.get("missing_info", [])
                    refined_queries = metadata.get("refined_queries", [])
                    round1_count = metadata.get("round1_count", 0)
                    round2_count = metadata.get("round2_count", 0)
                    final_count = metadata.get("final_count", len(memories))
                    latency = metadata.get("total_latency_ms", 0)
                    
                    # 更新统计
                    if is_multi_round:
                        self.multi_round_count += 1
                    if is_sufficient:
                        self.sufficient_count += 1
                    
                    self.successful_tests += 1
                    test_result = {
                        "test_name": test_name,
                        "status": "✅ 成功",
                        "query": query,
                        "user_id": user_id,
                        "group_id": group_id,
                        "count": len(memories),
                        "latency_ms": latency,
                        "is_multi_round": is_multi_round,
                        "is_sufficient": is_sufficient,
                        "reasoning": reasoning,
                        "missing_info": missing_info,
                        "refined_queries": refined_queries,
                        "round1_count": round1_count,
                        "round2_count": round2_count,
                        "final_count": final_count,
                        "metadata": metadata,
                        "memories": memories[:3],  # 只保存前3条
                    }
                    
                    # 打印详细信息
                    print(f"\n  ✅ {test_name}: 找到 {len(memories)} 条记忆")
                    print(f"     查询: {query[:50]}{'...' if len(query) > 50 else ''}")
                    print(f"     耗时: {latency:.2f}ms")
                    print(f"     多轮: {'是' if is_multi_round else '否'}")
                    print(f"     充分: {'是' if is_sufficient else '否'}")
                    if reasoning:
                        print(f"     LLM 判断: {reasoning[:60]}{'...' if len(reasoning) > 60 else ''}")
                    if refined_queries:
                        print(f"     改进查询: {refined_queries}")
                    print(f"     Round1: {round1_count} 条 → Round2: {round2_count} 条 → 最终: {final_count} 条")
                    
                    # 打印前3条记忆的摘要
                    if memories:
                        print(f"\n     🎯 Top 3 记忆摘要:")
                        for i, mem in enumerate(memories[:3], 1):
                            score = mem.get('score', 0)
                            subject = mem.get('subject', '')
                            summary = mem.get('summary', '')
                            episode = mem.get('episode', '')
                            content = subject or summary or episode[:60]
                            print(f"       [{i}] 分数: {score:.4f} | {content[:50]}{'...' if len(content) > 50 else ''}")
                    
                    return test_result
                else:
                    self.failed_tests += 1
                    error_msg = result.get('message', '未知错误')
                    print(f"  ❌ {test_name}: 检索失败 - {error_msg}")
                    return {
                        "test_name": test_name,
                        "status": "❌ 失败",
                        "error": error_msg,
                    }
                    
        except httpx.ConnectError:
            self.failed_tests += 1
            print(f"  ❌ {test_name}: 无法连接到 API 服务器")
            return {
                "test_name": test_name,
                "status": "❌ 连接失败",
                "error": "无法连接到 API 服务器",
            }
        except Exception as e:
            self.failed_tests += 1
            print(f"  ❌ {test_name}: 异常 - {e}")
            return {
                "test_name": test_name,
                "status": "❌ 异常",
                "error": str(e),
            }
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*80)
        print("📊 Agentic 检索测试总结")
        print("="*80)
        print(f"总测试数: {self.total_tests}")
        print(f"成功: {self.successful_tests} ✅")
        print(f"失败: {self.failed_tests} ❌")
        if self.total_tests > 0:
            print(f"成功率: {(self.successful_tests/self.total_tests*100):.1f}%")
        
        # Agentic 特有统计
        print(f"\n🔍 Agentic 特性统计:")
        if self.successful_tests > 0:
            print(f"  多轮检索: {self.multi_round_count}/{self.successful_tests} ({(self.multi_round_count/self.successful_tests*100):.1f}%)")
            print(f"  LLM 判定充分: {self.sufficient_count}/{self.successful_tests} ({(self.sufficient_count/self.successful_tests*100):.1f}%)")
        
        # 平均耗时
        successful_results = [r for r in self.test_results if r.get("status") == "✅ 成功"]
        if successful_results:
            avg_latency = sum(r.get("latency_ms", 0) for r in successful_results) / len(successful_results)
            avg_count = sum(r.get("count", 0) for r in successful_results) / len(successful_results)
            print(f"\n📈 平均指标:")
            print(f"  平均耗时: {avg_latency:.2f}ms")
            print(f"  平均召回: {avg_count:.1f} 条")
        
        # 失败的测试详情
        failed_results = [r for r in self.test_results if r.get("status") != "✅ 成功"]
        if failed_results:
            print("\n❌ 失败的测试:")
            for r in failed_results:
                print(f"  - {r.get('test_name')}: {r.get('error', '未知错误')}")
    
    def export_results(self, output_file: str = "demo/results/agentic_test_results.json"):
        """导出测试结果到 JSON 文件
        
        Args:
            output_file: 输出文件路径
        """
        import json
        from pathlib import Path
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建导出数据
        export_data = {
            "test_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "successful_tests": self.successful_tests,
                "failed_tests": self.failed_tests,
                "success_rate": f"{(self.successful_tests/self.total_tests*100):.1f}%" if self.total_tests > 0 else "0%",
                "multi_round_count": self.multi_round_count,
                "sufficient_count": self.sufficient_count,
            },
            "test_results": self.test_results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 测试结果已保存到: {output_file}")


async def start_mock_llm(port: int = 9001) -> Optional["web.AppRunner"]:
    """启动 Mock LLM 服务器"""
    if web is None:
        raise RuntimeError("需要 aiohttp 才能启用 Mock LLM，请先: uv add aiohttp")
    
    import json
    
    async def handle(request: web.Request) -> web.Response:
        data = await request.json()
        prompt = data.get("messages", [{}])[0].get("content", "")
        
        # 根据 prompt 内容返回不同响应
        if "改进查询" in prompt:
            # Multi-query generation
            content = json.dumps(
                {
                    "queries": [
                        "用户喜欢的餐厅有哪些？",
                        "robot_001 的口味偏好？",
                        "用户的饮食习惯和禁忌？"
                    ],
                    "reasoning": "mock refinement: 生成多个互补查询"
                },
                ensure_ascii=False,
            )
        else:
            # Sufficiency check
            content = json.dumps(
                {
                    "is_sufficient": True,
                    "reasoning": "mock sufficient: 检索结果充分",
                    "missing_information": []
                },
                ensure_ascii=False,
            )
        
        return web.json_response(
            {
                "choices": [
                    {"message": {"content": content}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 15,
                    "total_tokens": 65,
                },
            }
        )
    
    app = web.Application()
    app.router.add_post("/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    print(f"✅ Mock LLM 已启动: http://127.0.0.1:{port}/chat/completions")
    return runner


async def main():
    """主测试函数"""
    
    print("="*80)
    print("🧪 Agentic 检索完整测试")
    print("="*80)
    print("\n本测试将系统地测试 Agentic 检索的多个场景：")
    print("  1. 个人记忆查询（无 group_id）")
    print("  2. 群组记忆查询（有 group_id）")
    print("  3. 多维度深挖查询（复杂问题）")
    print("\n⚠️  请确保 API 服务器已启动: uv run python src/bootstrap.py src/run.py --port 8001")
    
    # 检查是否有 LLM API Key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://openrouter.ai/api/v1"
    model = os.getenv("LLM_MODEL") or "qwen/qwen3-235b-a22b-2507"
    
    use_mock = False
    mock_runner = None
    
    if not api_key:
        print("\n⚠️  未检测到 LLM API Key (OPENROUTER_API_KEY/OPENAI_API_KEY/LLM_API_KEY)")
        print("   将使用 Mock LLM 进行测试（模拟 LLM 响应）")
        use_mock = True
    else:
        print(f"\n✅ 检测到 LLM 配置:")
        print(f"   Base URL: {base_url}")
        print(f"   Model: {model}")
        print(f"   API Key: {api_key[:10]}...")
    
    print("\n⏳ 开始测试...")
    
    # 启动 Mock LLM（如果需要）
    if use_mock:
        try:
            mock_runner = await start_mock_llm(port=9001)
            api_key = "mock-key"
            base_url = "http://127.0.0.1:9001"
            model = "mock-llm"
            await asyncio.sleep(0.2)  # 等待服务启动
        except Exception as e:
            print(f"❌ 启动 Mock LLM 失败: {e}")
            print("   提示: 如需使用 Mock LLM，请先安装: uv add aiohttp")
            return
    
    # LLM 配置
    llm_config = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }
    
    try:
        # 创建测试器
        tester = AgenticRetrievalTester()
        
        # ========== 测试场景 1: 个人记忆查询 ==========
        print("\n" + "🔬"*40)
        print("测试场景 1: 个人记忆查询（无 group_id）")
        print("🔬"*40)
        
        result1 = await tester.test_agentic(
            query="北京旅游美食推荐",
            user_id="robot_001",
            group_id=None,
            top_k=10,
            time_range_days=365,
            llm_config=llm_config,
            test_name="场景1_个人记忆",
        )
        tester.test_results.append(result1)
        await asyncio.sleep(1)
        
        # ========== 测试场景 2: 群组记忆查询 ==========
        print("\n" + "🔬"*40)
        print("测试场景 2: 群组记忆查询（有 group_id）")
        print("🔬"*40)
        
        result2 = await tester.test_agentic(
            query="北京美食和旅游推荐",
            user_id="robot_001",
            group_id="chat_user_001_assistant",
            top_k=15,
            time_range_days=365,
            llm_config=llm_config,
            test_name="场景2_群组记忆",
        )
        tester.test_results.append(result2)
        await asyncio.sleep(1)
        
        # ========== 测试场景 3: 多维度深挖 ==========
        print("\n" + "🔬"*40)
        print("测试场景 3: 多维度深挖（复杂问题）")
        print("🔬"*40)
        
        result3 = await tester.test_agentic(
            query="用户的性格特点、兴趣爱好、饮食偏好和旅行习惯是什么？",
            user_id="robot_001",
            group_id="chat_user_001_assistant",
            top_k=20,
            time_range_days=365,
            llm_config=llm_config,
            test_name="场景3_多维度",
        )
        tester.test_results.append(result3)
        await asyncio.sleep(1)
        
        # ========== 测试场景 4: 短周期查询 ==========
        print("\n" + "🔬"*40)
        print("测试场景 4: 短周期查询（30天内）")
        print("🔬"*40)
        
        result4 = await tester.test_agentic(
            query="最近的北京旅游计划",
            user_id="robot_001",
            group_id="chat_user_001_assistant",
            top_k=10,
            time_range_days=30,
            llm_config=llm_config,
            test_name="场景4_短周期",
        )
        tester.test_results.append(result4)
        
        # ========== 打印总结 ==========
        tester.print_summary()
        
        # ========== 导出结果 ==========
        tester.export_results()
        
        print("\n" + "="*80)
        print("✅ Agentic 检索测试完成！")
        print("="*80)
        
    finally:
        # 清理 Mock LLM
        if mock_runner:
            await mock_runner.cleanup()
            print("\n✅ Mock LLM 已关闭")


if __name__ == "__main__":
    asyncio.run(main())
