"""
Hello World 任务

提供一个简单的 Hello World 任务
"""

from core.asynctasks.task_manager import task
from typing import Any


@task()
async def hello_world(data: Any) -> Any:
    return f"hello world: {data}"
