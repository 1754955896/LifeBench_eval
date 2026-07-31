# -*- coding: utf-8 -*-
"""
依赖注入系统异常类定义
"""

from typing import Type, Any, List


class DIException(Exception):
    """依赖注入系统基础异常"""

    pass


class CircularDependencyError(DIException):
    """循环依赖异常"""

    def __init__(self, dependency_chain: List[Type]):
        self.dependency_chain = dependency_chain
        chain_str = " -> ".join([cls.__name__ for cls in dependency_chain])
        super().__init__(f"检测到循环依赖: {chain_str}")


class BeanNotFoundError(DIException):
    """Bean未找到异常"""

    def __init__(self, bean_type: Type = None, bean_name: str = None):
        self.bean_type = bean_type
        self.bean_name = bean_name

        if bean_name:
            super().__init__(f"未找到名为 '{bean_name}' 的Bean")
        elif bean_type:
            # 处理字符串类型的bean_type
            if isinstance(bean_type, str):
                super().__init__(f"未找到类型为 '{bean_type}' 的Bean")
            else:
                super().__init__(f"未找到类型为 '{bean_type.__name__}' 的Bean")
        else:
            super().__init__("未找到指定的Bean")


class DuplicateBeanError(DIException):
    """重复Bean异常"""

    def __init__(self, bean_type: Type = None, bean_name: str = None):
        self.bean_type = bean_type
        self.bean_name = bean_name

        if bean_name:
            super().__init__(f"名为 '{bean_name}' 的Bean已存在")
        elif bean_type:
            super().__init__(f"类型为 '{bean_type.__name__}' 的Bean已存在")
        else:
            super().__init__("Bean已存在")


class FactoryError(DIException):
    """Factory异常"""

    def __init__(self, factory_type: Type, message: str = None):
        self.factory_type = factory_type
        default_msg = f"Factory '{factory_type.__name__}' 创建实例失败"
        super().__init__(message or default_msg)


class DependencyResolutionError(DIException):
    """依赖解析异常"""

    def __init__(self, target_type: Type, missing_dependency: Type):
        self.target_type = target_type
        self.missing_dependency = missing_dependency
        super().__init__(
            f"无法解析 '{target_type.__name__}' 的依赖 '{missing_dependency.__name__}'"
        )


class MockNotEnabledError(DIException):
    """Mock模式未启用异常"""

    def __init__(self):
        super().__init__("Mock模式未启用，无法注册Mock实现")


class PrimaryBeanConflictError(DIException):
    """Primary Bean冲突异常"""

    def __init__(self, bean_type: Type, existing_primary: Type, new_primary: Type):
        self.bean_type = bean_type
        self.existing_primary = existing_primary
        self.new_primary = new_primary
        super().__init__(
            f"类型 '{bean_type.__name__}' 存在多个Primary实现: "
            f"'{existing_primary.__name__}' 和 '{new_primary.__name__}'"
        )
