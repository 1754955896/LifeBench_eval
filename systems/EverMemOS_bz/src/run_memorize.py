#!/usr/bin/env python3
"""
群聊记忆存储脚本

读取符合 GroupChatFormat 格式的 JSON 文件，转换后调用 memorize 接口存储记忆

使用方法:
    # 调用 memorize 接口：简单直接的单条消息格式，逐条处理
    python src/bootstrap.py src/run_memorize.py --input data/group_chat.json --api-url http://localhost:1995/api/v1/memories

    # 仅验证格式
    python src/bootstrap.py src/run_memorize.py --input data/example.json --validate-only
"""

import json
import argparse
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional

ALLOWED_SCENES: tuple[str, str] = ("assistant", "companion")
# todo xinzegao 这里原来是 group_chat

from infra_layer.adapters.input.api.mapper.group_chat_converter import (
    validate_group_chat_format_input,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


class GroupChatMemorizer:
    """群聊记忆存储处理类"""

    def __init__(self, api_url: str, scene: str = "assistant"):
        """
        初始化

        Args:
            api_url: memorize API地址（必需）
            scene: 记忆提取场景（默认 "assistant"）
        """
        self.api_url = api_url
        self.scene = scene

    def validate_input_file(self, file_path: str) -> bool:
        """
        验证输入文件格式

        Args:
            file_path: 输入文件路径

        Returns:
            是否验证通过
        """
        logger.info("=" * 70)
        logger.info("验证输入文件格式")
        logger.info("=" * 70)

        try:
            # 读取文件
            logger.info(f"正在读取文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 验证格式
            logger.info("正在验证 GroupChatFormat 格式...")
            is_valid = validate_group_chat_format_input(data)

            if is_valid:
                logger.info("✓ 格式验证通过！")

                # 输出统计信息
                meta = data.get("conversation_meta", {})
                messages = data.get("conversation_list", [])

                logger.info("\n=== 数据统计 ===")
                logger.info(f"格式版本: {data.get('version', 'N/A')}")
                logger.info(f"群聊名称: {meta.get('name', 'N/A')}")
                logger.info(f"群聊ID: {meta.get('group_id', 'N/A')}")
                logger.info(f"用户数量: {len(meta.get('user_details', {}))}")
                logger.info(f"消息数量: {len(messages)}")

                if messages:
                    first_time = messages[0].get('create_time', 'N/A')
                    last_time = messages[-1].get('create_time', 'N/A')
                    logger.info(f"时间范围: {first_time} ~ {last_time}")

                return True
            else:
                logger.error("✗ 格式验证失败！")
                logger.error("请确保输入文件符合 GroupChatFormat 规范")
                return False

        except json.JSONDecodeError as e:
            logger.error(f"✗ JSON 解析失败: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ 验证失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def process_with_api(self, group_chat_data: Dict[str, Any]) -> bool:
        """
        通过 API 逐条处理（使用简单直接的单条消息格式）

        Args:
            group_chat_data: GroupChatFormat 格式的数据

        Returns:
            是否成功
        """
        logger.info("\n" + "=" * 70)
        logger.info("开始逐条调用 memorize API")
        logger.info("=" * 70)

        try:
            import httpx

            meta = group_chat_data.get("conversation_meta", {})
            messages = group_chat_data.get("conversation_list", [])

            group_id = meta.get("group_id")
            group_name = meta.get("name")

            logger.info(f"群组名称: {group_name or 'N/A'}")
            logger.info(f"群组ID: {group_id or 'N/A'}")
            logger.info(f"消息数量: {len(messages)}")
            logger.info(f"API地址: {self.api_url}")

            # ========== 第一步：先调用 conversation-meta 接口保存 scene ==========
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info("\n--- 保存对话元数据 (conversation-meta) ---")

                # 构建 conversation-meta 请求数据
                conversation_meta_request = {
                    "version": group_chat_data.get("version", "1.0.0"),
                    "scene": self.scene,  # 使用命令行传入的 scene
                    "scene_desc": meta.get("scene_desc", {}),
                    "name": meta.get("name", "未命名对话"),
                    "description": meta.get("description", ""),
                    "group_id": group_id,
                    "created_at": meta.get("created_at", ""),
                    "default_timezone": meta.get("default_timezone", "Asia/Shanghai"),
                    "user_details": meta.get("user_details", {}),
                    "tags": meta.get("tags", []),
                }

                # 获取 conversation-meta API 地址（在 memories API 的基础上构建）
                # 假设 memories API 是 http://host:port/api/v1/memories
                # 则 conversation-meta API 是 http://host:port/api/v1/memories/conversation-meta
                conversation_meta_url = f"{self.api_url}/conversation-meta"

                logger.info(f"正在保存对话元数据到: {conversation_meta_url}")
                logger.info(f"Scene: {self.scene}, Group ID: {group_id}")

                try:
                    response = await client.post(
                        conversation_meta_url,
                        json=conversation_meta_request,
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"  ✓ 对话元数据保存成功")
                        logger.info(f"  Scene: {self.scene}")
                    else:
                        logger.warning(
                            f"  ⚠ 对话元数据保存失败: {response.status_code}"
                        )
                        logger.warning(f"  响应内容: {response.text}")
                        logger.warning(f"  继续处理消息...")

                except Exception as e:
                    logger.warning(f"  ⚠ 保存对话元数据时出错: {e}")
                    logger.warning(f"  继续处理消息...")

            # ========== 第二步：逐条处理消息 ==========

            total_memories = 0
            success_count = 0

            async with httpx.AsyncClient(timeout=300.0) as client:
                for i, message in enumerate(messages):
                    logger.info(f"\n--- 处理第 {i+1}/{len(messages)} 条消息 ---")

                    # 构建简单直接的单条消息格式
                    request_data = {
                        "message_id": message.get("message_id"),
                        "create_time": message.get("create_time"),
                        "sender": message.get("sender"),
                        "sender_name": message.get("sender_name"),
                        "content": message.get("content"),
                        "refer_list": message.get("refer_list", []),
                    }

                    # 添加可选的 group 信息
                    if group_id:
                        request_data["group_id"] = group_id
                    if group_name:
                        request_data["group_name"] = group_name

                    # 发送请求
                    try:
                        response = await client.post(
                            self.api_url,
                            json=request_data,
                            headers={"Content-Type": "application/json"},
                        )

                        if response.status_code == 200:
                            result = response.json()
                            result_data = result.get('result', {})
                            memory_count = result_data.get('count', 0)

                            total_memories += memory_count
                            success_count += 1
                            if memory_count > 0:
                                logger.info(f"  ✓ 成功保存 {memory_count} 条记忆")
                            else:
                                logger.info(f"  ⏳ 等待情景边界")
                            # 添加延迟避免过快处理
                            time.sleep(0.1)

                        else:
                            logger.error(f"  ✗ API调用失败: {response.status_code}")
                            logger.error(f"  响应内容: {response.text}")

                    except Exception as e:
                        logger.error(f"  ✗ 处理失败: {e}")

            # 输出总结
            logger.info("\n" + "=" * 70)
            logger.info("处理完成")
            logger.info("=" * 70)
            logger.info(f"✓ 成功处理: {success_count}/{len(messages)} 条消息")
            logger.info(f"✓ 共保存: {total_memories} 条记忆")

            return success_count == len(messages)

        except ImportError:
            logger.error("✗ 需要安装 httpx 库: pip install httpx")
            return False
        except Exception as e:
            logger.error(f"✗ 处理失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def process_file(self, file_path: str) -> bool:
        """
        处理群聊文件

        Args:
            file_path: 输入文件路径

        Returns:
            是否成功
        """
        # 先验证格式
        if not self.validate_input_file(file_path):
            return False

        # 检查 API 地址
        if not self.api_url:
            logger.error("✗ 未提供 API 地址，请使用 --api-url 参数指定")
            return False

        try:
            # 读取文件
            logger.info("\n" + "=" * 70)
            logger.info("读取群聊数据")
            logger.info("=" * 70)
            logger.info(f"正在读取文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                group_chat_data = json.load(f)

            # 逐条接口：直接发送 GroupChatFormat 格式，逐条处理
            logger.info("使用简单直接的单条消息格式，逐条处理")
            return await self.process_with_api(group_chat_data)

        except Exception as e:
            logger.error(f"✗ 读取或处理失败: {e}")
            import traceback

            traceback.print_exc()
            return False


async def async_main():
    """异步主函数"""
    parser = argparse.ArgumentParser(
        description='群聊记忆存储脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 调用 memorize 接口：简单直接的单条消息格式，逐条处理
  python src/bootstrap.py src/run_memorize.py --input data/group_chat.json --api-url http://localhost:1995/api/v1/memories
  
  # 仅验证格式（不需要 API 地址）
  python src/bootstrap.py src/run_memorize.py --input data/group_chat.json --validate-only
  
输入文件格式:
  输入文件必须符合 GroupChatFormat 规范，参考 data_format/group_chat/group_chat_format.py
        """,
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的群聊JSON文件路径（GroupChatFormat格式）',
    )
    parser.add_argument(
        '--api-url', type=str, help='memorize API地址（必需，除非使用 --validate-only）'
    )
    parser.add_argument(
        '--scene',
        type=str,
        choices=ALLOWED_SCENES,
        required=True,
        help='记忆提取场景（必填，仅支持 assistant 或 companion）',
    )
    parser.add_argument(
        '--validate-only', action='store_true', help='仅验证输入文件格式，不调用 API'
    )

    args = parser.parse_args()

    # 处理输入文件路径
    input_file = Path(args.input)
    if not input_file.is_absolute():
        # 相对路径，相对于当前工作目录
        input_file = Path.cwd() / input_file

    if not input_file.exists():
        logger.error(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)

    logger.info("🚀 群聊记忆存储脚本")
    logger.info("=" * 70)
    logger.info(f"📄 输入文件: {input_file}")
    logger.info(f"🔍 验证模式: {'是' if args.validate_only else '否'}")
    if args.api_url:
        logger.info(f"🌐 API地址: {args.api_url}")
    logger.info("=" * 70)

    # 如果只是验证模式，验证后退出
    if args.validate_only:
        # 验证模式不需要 API 地址
        memorizer = GroupChatMemorizer(api_url="", scene=args.scene)  # 传入空字符串占位
        success = memorizer.validate_input_file(str(input_file))
        if success:
            logger.info("\n✓ 验证完成，文件格式正确！")
            sys.exit(0)
        else:
            logger.error("\n✗ 验证失败，文件格式不正确！")
            sys.exit(1)

    # 非验证模式，必须提供 API 地址
    if not args.api_url:
        logger.error("✗ 错误: 必须提供 --api-url 参数")
        logger.error("   使用方法:")
        logger.error(
            "     python src/bootstrap.py src/run_memorize.py --input <file> --api-url http://localhost:1995/api/v1/memories"
        )
        logger.error("   或使用 --validate-only 仅验证格式")
        sys.exit(1)

    # 创建处理器并处理文件
    memorizer = GroupChatMemorizer(api_url=args.api_url, scene=args.scene)
    success = await memorizer.process_file(str(input_file))

    if success:
        logger.info("\n" + "=" * 70)
        logger.info("✓ 处理完成！")
        logger.info("=" * 70)
    else:
        logger.error("\n" + "=" * 70)
        logger.error("✗ 处理失败！")
        logger.error("=" * 70)


def main():
    """同步主函数入口"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 执行失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
