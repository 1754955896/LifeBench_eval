import os

PROJECT_NAME = "EverMem"
PROJECT_VERSION = "1.0.0"


def get_env_project_name():
    """
    获取环境变量中的项目名称
    """
    project_name = os.getenv("project_name") or os.getenv("PROJECT_NAME")
    if project_name:
        return project_name
    else:
        return PROJECT_NAME
