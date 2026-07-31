#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memsys Bootstrap Script - 通用的上下文加载器和脚本运行器

这个脚本让算法同事能够无认知负担地运行任何测试脚本，自动处理：
- Python 路径设置
- 环境变量加载
- 依赖注入容器初始化
- Mock 模式支持

用法:
    python src/bootstrap.py [你的脚本路径] [你的脚本的参数...]

示例:
    python src/bootstrap.py tests/algorithms/debug_my_model.py
    python src/bootstrap.py unit_test/memory_manager_single_test.py --verbose
    python src/bootstrap.py evaluation/dynamic_memory_evaluation/locomo_eval.py --dataset small
"""

import sys
import runpy
import argparse
import os
import nest_asyncio

nest_asyncio.apply()
import asyncio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def file_path_to_module_name(target_path: Path, src_path: Path) -> str:
    """
    将文件路径转换为模块名称

    Args:
        target_path: 目标脚本的路径
        src_path: src 目录的路径

    Returns:
        模块名称，如 "api_layer.get_data.run_consumer"
    """
    # 确保路径是绝对路径
    target_path = target_path.resolve()
    src_path = src_path.resolve()

    try:
        # 首先检查是否在 src 目录下
        if target_path.is_relative_to(src_path):
            # 如果在 src 目录下，相对于 src 目录计算
            relative_path = target_path.relative_to(src_path)
            module_name = (
                str(relative_path.with_suffix('')).replace('/', '.').replace('\\', '.')
            )
            return module_name
        else:
            # 如果不在 src 目录下，相对于项目根目录计算
            project_root = src_path.parent
            relative_path = target_path.relative_to(project_root)
            module_name = (
                str(relative_path.with_suffix('')).replace('/', '.').replace('\\', '.')
            )
            return module_name
    except ValueError:
        # 如果无法计算相对路径，尝试相对于当前目录
        try:
            relative_path = target_path.relative_to(Path.cwd())
            module_name = (
                str(relative_path.with_suffix('')).replace('/', '.').replace('\\', '.')
            )
            return module_name
        except ValueError:
            # 最后的fallback：使用文件名作为模块名
            return target_path.stem


async def setup_project_context(env_file=".env", mock_mode=False):
    """
    设置项目上下文环境 - 完全照抄 run.py 的加载逻辑
    """
    # 照抄 run.py 的环境加载逻辑
    from import_parent_dir import add_parent_path

    add_parent_path(0)

    from common_utils.load_env import setup_environment

    # 设置环境（Python路径和.env文件）
    setup_environment(load_env_file_name=env_file, check_env_var="MONGODB_HOST")

    # 照抄 run.py 的 Mock 模式检查逻辑
    from core.di.utils import enable_mock_mode

    # 检查是否启用Mock模式：优先检查命令行参数，其次检查环境变量
    if mock_mode or (
        os.getenv("MOCK_MODE") and os.getenv("MOCK_MODE").lower() == "true"
    ):
        enable_mock_mode()
        logger.info("🚀 启用Mock模式")
    else:
        logger.info("🚀 禁用Mock模式")

    # 照抄 run.py 的依赖注入设置
    from application_startup import setup_all

    # 在模块加载时就执行依赖注入和异步任务设置
    setup_all()

    # 异步启动应用生命周期
    try:
        from app import app

        if hasattr(app, "start_lifespan"):
            await app.start_lifespan()
            logger.info("✅ 应用lifespan启动完成")
        else:
            logger.warning("⚠️ app实例没有start_lifespan方法")
    except Exception as e:
        logger.warning(f"⚠️ 启动应用lifespan时出错: {e}")
        # 不抛出异常，继续执行


async def async_main():
    """异步主函数：解析参数并运行目标脚本"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="在完整的应用上下文中运行 Python 脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python src/bootstrap.py tests/algorithms/debug_my_model.py
  python src/bootstrap.py unit_test/memory_manager_single_test.py --verbose
  python src/bootstrap.py evaluation/dynamic_memory_evaluation/locomo_eval.py --dataset small
  
环境变量:
  MOCK_MODE=true    启用 Mock 模式（用于测试）
        """,
    )

    parser.add_argument("script_path", help="要运行的 Python 脚本路径")
    parser.add_argument(
        'script_args', nargs=argparse.REMAINDER, help="传递给目标脚本的参数"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="指定要加载的环境变量文件 (默认: .env)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="启用Mock模式 (用于测试和开发)"
    )

    args = parser.parse_args()

    print("🚀 Memsys Bootstrap Script")
    print("=" * 50)
    print(f"📄 目标脚本: {args.script_path}")
    print(f"📝 脚本参数: {args.script_args}")
    print(f"📄 Env File: {args.env_file}")
    print(f"🎭 Mock 模式: {'启用' if args.mock else '禁用'}")
    print("=" * 50)

    # 设置项目上下文（完全照抄 run.py 的逻辑）
    await setup_project_context(env_file=args.env_file, mock_mode=args.mock)

    # 验证目标脚本是否存在
    script_path = Path(args.script_path)
    if not script_path.exists():
        print(f"❌ 错误: 脚本文件不存在: {args.script_path}", file=sys.stderr)
        sys.exit(1)

    # 准备执行目标脚本
    # 关键：修改 sys.argv，让目标脚本认为它是被直接调用的
    # 这样它才能正确接收自己的参数
    original_argv = sys.argv.copy()  # 备份原始参数
    sys.argv = [str(script_path)] + args.script_args

    print(f"\n🎬 开始执行脚本: {args.script_path}")
    print("-" * 50)

    try:
        # 使用 runpy 执行目标脚本
        # run_path 会像 'python script_path' 一样执行脚本
        # run_name="__main__" 确保脚本中的 if __name__ == "__main__": 块能正常执行
        runpy.run_path(str(script_path), run_name="__main__")

    except ImportError as e:
        # 如果遇到相对导入错误，尝试使用模块模式运行
        if "attempted relative import with no known parent package" in str(e):
            print(f"\n⚠️  检测到相对导入错误，尝试使用模块模式运行...")
            try:
                # 获取 src 目录路径
                src_path = Path(__file__).parent  # bootstrap.py 在 src 目录中
                module_name = file_path_to_module_name(script_path, src_path)
                print(
                    f"📦 将路径 '{script_path}' 解释为模块 '{module_name}'，重试中..."
                )

                # 让脚本的 sys.argv[0] 依然是文件路径
                sys.argv[0] = str(script_path)
                runpy.run_module(module_name, run_name="__main__")

            except Exception as module_error:
                print(f"\n❌ 模块模式运行也失败: {module_error}", file=sys.stderr)
                print(f"原始错误: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                sys.exit(1)
        else:
            # 其他导入错误，直接抛出
            raise

    except SystemExit as e:
        # 目标脚本可能会调用 sys.exit()，这是正常的
        print(f"\n📋 脚本执行完成，退出码: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        print(f"\n❌ 脚本执行出错: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
    finally:
        # 恢复原始的 sys.argv
        sys.argv = original_argv
        print(f"\n🏁 脚本执行结束: {args.script_path}")


def main():
    """同步主函数入口点"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
