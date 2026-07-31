"""记忆增强对话脚本

使用方法：
    uv run python src/bootstrap.py demo/chat_with_memory.py
    
备选方式：
    cd demo
    python chat_with_memory.py
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from demo.chat import ChatOrchestrator

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def main():
    """主入口 - 启动聊天应用"""
    orchestrator = ChatOrchestrator(PROJECT_ROOT)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
