# -*- coding: utf-8 -*-
"""
调试控制器

提供DI容器中Bean的调试接口，支持调用特定service的特定方法
仅在开发环境下启用，生产环境自动禁用
"""

import asyncio
import inspect
import os
import json
import traceback
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from fastapi import HTTPException

from core.interface.controller.base_controller import BaseController, post, get
from core.di import get_container, get_bean, get_bean_by_type
from core.di.decorators import controller
from core.observation.logger import get_logger

from core.constants.errors import ErrorMessage

logger = get_logger(__name__)


class BeanCallRequest(BaseModel):
    """Bean方法调用请求模型（兼容代码执行）"""

    # Bean标识（二选一）
    bean_name: Optional[str] = Field(None, description="Bean名称")
    bean_type: Optional[str] = Field(None, description="Bean类型名称")

    # 方法调用
    method: str = Field(..., description="要调用的方法名称")

    # 传统参数方式
    args: List[Any] = Field(default_factory=list, description="位置参数列表")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="关键字参数字典")

    # 代码执行方式（可选）
    code: Optional[str] = Field(
        None, description="Python代码，用于生成args和kwargs参数（可选）"
    )

    @classmethod
    def model_validate(cls, obj, *, strict=None, from_attributes=None, context=None):
        """自定义验证逻辑"""
        if (
            isinstance(obj, dict)
            and not obj.get("bean_name")
            and not obj.get("bean_type")
        ):
            logger.error("Bean调用请求验证失败：缺少bean_name或bean_type参数")
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )


class BeanCallWithCodeRequest(BaseModel):
    """通过代码生成参数的Bean方法调用请求模型"""

    # Bean标识（二选一）
    bean_name: Optional[str] = Field(None, description="Bean名称")
    bean_type: Optional[str] = Field(None, description="Bean类型名称")

    # 方法调用
    method: str = Field(..., description="要调用的方法名称")

    # Python代码生成参数
    code: str = Field(..., description="Python代码，用于生成args和kwargs参数")

    @classmethod
    def model_validate(cls, obj, *, strict=None, from_attributes=None, context=None):
        """自定义验证逻辑"""
        if (
            isinstance(obj, dict)
            and not obj.get("bean_name")
            and not obj.get("bean_type")
        ):
            logger.error("Bean代码调用请求验证失败：缺少bean_name或bean_type参数")
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )


class BeanCallResponse(BaseModel):
    """Bean方法调用响应模型"""

    success: bool = Field(..., description="调用是否成功")
    result: Optional[Any] = Field(None, description="方法返回值（JSON可序列化）")
    result_str: Optional[str] = Field(
        None, description="方法返回值的字符串表示（当无法JSON序列化时）"
    )
    error: Optional[str] = Field(None, description="错误信息")
    traceback: Optional[str] = Field(None, description="详细的错误堆栈信息")
    bean_info: Optional[Dict[str, Any]] = Field(None, description="调用的Bean信息")
    code_execution: Optional[Dict[str, Any]] = Field(
        None, description="代码执行信息（仅在使用代码生成参数时）"
    )


class BeanInfoResponse(BaseModel):
    """Bean信息响应模型"""

    name: str = Field(..., description="Bean名称")
    type_name: str = Field(..., description="Bean类型名称")
    scope: str = Field(..., description="Bean作用域")
    is_primary: bool = Field(..., description="是否为Primary Bean")
    is_mock: bool = Field(..., description="是否为Mock Bean")
    methods: List[str] = Field(default_factory=list, description="可调用的方法列表")


@controller(name="debug_controller")
class DebugController(BaseController):
    """
    DI容器调试 API 控制器

    提供依赖注入容器的调试和测试功能，支持在开发环境下：
    - 查看所有已注册的Bean信息
    - 调用任意Bean的任意方法进行测试
    - 获取Bean的详细配置和方法列表
    - 监控DI容器的运行状态

    **安全机制**：
    - 只在开发环境启用（ENV=DEV）
    - 生产环境自动禁用所有调试接口
    - 不需要用户认证，但受环境变量控制

    **主要功能**：
    1. **Bean查询**: 支持按名称或类型查找Bean
    2. **方法调用**: 支持传递参数调用Bean方法
    3. **状态监控**: 查看容器Mock模式、Bean数量等
    4. **错误诊断**: 提供详细的调用错误信息和堆栈跟踪
    """

    def __init__(self):
        super().__init__(
            prefix="/asdf/debug/di",
            tags=["Debug"],
            default_auth="none",  # 调试接口不需要认证，但会通过环境变量控制访问
        )
        self.container = get_container()

    def _check_debug_enabled(self) -> bool:
        """检查调试功能是否启用"""
        return os.environ.get('ENV', 'prod').upper() == 'DEV'

    def _ensure_debug_enabled(self):
        """确保调试功能已启用，否则抛出404错误"""
        if not self._check_debug_enabled():
            logger.error("调试功能未启用，拒绝访问调试接口")
            raise HTTPException(
                status_code=404, detail=ErrorMessage.PERMISSION_DENIED.value
            )

    def _get_bean_methods(self, bean_instance: Any) -> List[str]:
        """获取Bean实例的可调用方法列表"""
        methods = []
        for attr_name in dir(bean_instance):
            if not attr_name.startswith('_'):  # 排除私有方法
                attr = getattr(bean_instance, attr_name)
                if callable(attr):
                    methods.append(attr_name)
        return sorted(methods)

    def _get_bean_by_identifier(
        self, bean_name: Optional[str], bean_type: Optional[str]
    ) -> tuple[Any, Dict[str, Any]]:
        """
        根据标识符获取Bean实例和信息

        Args:
            bean_name: Bean名称
            bean_type: Bean类型名称

        Returns:
            tuple: (bean_instance, bean_info)

        Raises:
            HTTPException: 当找不到Bean或参数无效时
        """
        if bean_name and bean_type:
            logger.error("Bean标识符参数错误：不能同时提供bean_name和bean_type")
            raise HTTPException(
                status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
            )

        if not bean_name and not bean_type:
            logger.error("Bean标识符参数错误：必须提供bean_name或bean_type之一")
            raise HTTPException(
                status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
            )

        try:
            if bean_name:
                # 通过名称获取Bean
                bean_instance = get_bean(bean_name)
                bean_info = {
                    "name": bean_name,
                    "type_name": type(bean_instance).__name__,
                    "lookup_method": "by_name",
                }
            else:
                # 通过类型名称获取Bean
                # 首先需要找到对应的类型
                bean_class = self._find_bean_type_by_name(bean_type)
                if not bean_class:
                    logger.error(f"未找到类型为'{bean_type}'的Bean类")
                    raise HTTPException(
                        status_code=404, detail=ErrorMessage.BEAN_NOT_FOUND.value
                    )

                bean_instance = get_bean_by_type(bean_class)
                bean_info = {
                    "name": getattr(bean_instance, '_di_name', bean_type.lower()),
                    "type_name": bean_type,
                    "lookup_method": "by_type",
                }

            return bean_instance, bean_info

        except Exception as e:
            if "not found" in str(e).lower():
                identifier = bean_name or bean_type
                method = "名称" if bean_name else "类型"
                logger.error(f"通过{method}'{identifier}'未找到Bean：{str(e)}")
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_NOT_FOUND.value
                ) from e
            else:
                logger.error(f"获取Bean时发生错误：{str(e)}")
                raise HTTPException(
                    status_code=500, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
                ) from e

    def _find_bean_type_by_name(self, type_name: str) -> Optional[Type]:
        """
        根据类型名称查找对应的Bean类型

        使用更可靠的方式：通过获取所有Bean并检查其类型，
        避免依赖可能不准确的list_all_beans_info

        Args:
            type_name: 类型名称

        Returns:
            对应的类型，如果未找到则返回None
        """
        try:
            # 方法1: 通过get_beans()获取所有Bean实例，检查类型名称
            all_beans_dict = self.container.get_beans()

            for _, bean_instance in all_beans_dict.items():
                if bean_instance is not None:
                    bean_type = type(bean_instance)
                    if bean_type.__name__ == type_name:
                        return bean_type

            return None

        except Exception:
            # 如果get_beans()失败，使用备用方法
            # 尝试使用一些常见的类型名称模式来推测
            try:
                # 先尝试获取所有Bean信息作为备用
                all_beans = self.container.list_all_beans_info()

                for bean_info in all_beans:
                    if bean_info['type_name'] == type_name:
                        # 通过名称获取Bean实例，然后获取其类型
                        try:
                            bean_instance = get_bean(bean_info['name'])
                            return type(bean_instance)
                        except Exception:
                            continue

                return None

            except Exception:
                # 如果所有方法都失败，返回None
                return None

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """
        序列化方法调用结果

        Args:
            result: 方法返回值

        Returns:
            包含序列化结果的字典
        """
        try:
            # 尝试JSON序列化
            json.dumps(result)
            return {"result": result}
        except (TypeError, ValueError):
            # 如果无法JSON序列化，返回字符串表示
            return {"result_str": repr(result)}

    def _execute_parameter_code(self, code: str) -> Dict[str, Any]:
        """
        安全执行Python代码生成参数

        Args:
            code: Python代码字符串

        Returns:
            包含args和kwargs的字典

        Raises:
            ValueError: 当代码执行失败或返回格式不正确时
        """
        try:
            # 创建安全的执行环境，允许自由导入
            safe_globals = {
                '__builtins__': {
                    # 基础类型和函数
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'print': print,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                    'type': type,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'sorted': sorted,
                    'reversed': reversed,
                    'any': any,
                    'all': all,
                    'map': map,
                    'filter': filter,
                    # 允许导入
                    '__import__': __import__,
                },
                # 预导入常用模块和类型
                'datetime': None,
                'json': None,
                'uuid': None,
                'typing': None,
            }

            # 预导入常用模块
            try:
                import datetime
                import json
                import uuid
                import typing

                safe_globals['datetime'] = datetime
                safe_globals['json'] = json
                safe_globals['uuid'] = uuid
                safe_globals['typing'] = typing
            except ImportError as e:
                logger.warning(f"预导入模块失败: {e}")

            # 不再预导入项目内部模块，支持完全自由导入

            local_vars = {}

            # 执行代码
            exec(code, safe_globals, local_vars)

            # 检查是否定义了args和kwargs
            if 'args' not in local_vars and 'kwargs' not in local_vars:
                logger.error("代码执行结果无效：未定义args或kwargs变量")
                raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

            args = local_vars.get('args', [])
            kwargs = local_vars.get('kwargs', {})

            # 验证类型
            if not isinstance(args, (list, tuple)):
                logger.error(
                    f"args参数类型错误：期望list或tuple，实际为{type(args).__name__}"
                )
                raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

            if not isinstance(kwargs, dict):
                logger.error(
                    f"kwargs参数类型错误：期望dict，实际为{type(kwargs).__name__}"
                )
                raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

            return {'args': list(args), 'kwargs': kwargs}

        except Exception as e:
            logger.error(f"执行参数生成代码失败: {e}")
            logger.error(f"代码执行异常详情：{str(e)}")
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

    @get(
        "/status",
        response_model=Dict[str, Any],
        summary="获取调试功能状态",
        responses={
            200: {
                "description": "调试状态信息获取成功",
                "content": {
                    "application/json": {
                        "example": {
                            "debug_enabled": True,
                            "container_info": {"mock_mode": False, "total_beans": 15},
                        }
                    }
                },
            }
        },
    )
    def get_debug_status(self) -> Dict[str, Any]:
        """
        获取调试功能状态信息

        返回调试功能是否启用以及DI容器的基本信息：
        - debug_enabled: 调试功能是否启用（基于ENV环境变量）
        - container_info: DI容器信息，包括Mock模式状态和Bean总数

        **注意**：
        - 当ENV != 'DEV'时，debug_enabled为false
        - 此接口不受调试开关控制，总是可以访问

        Returns:
            Dict[str, Any]: 包含调试状态和容器信息的字典
        """
        return {
            "debug_enabled": self._check_debug_enabled(),
            "container_info": {
                "mock_mode": self.container.is_mock_mode(),
                "total_beans": len(self.container.list_all_beans_info()),
            },
        }

    @get(
        "/beans",
        extra_models=[BeanInfoResponse],
        response_model=List[BeanInfoResponse],
        summary="列出所有已注册的Bean信息",
        responses={
            200: {
                "description": "Bean列表获取成功",
                "content": {
                    "application/json": {
                        "example": [
                            {
                                "name": "user_service",
                                "type_name": "UserService",
                                "scope": "singleton",
                                "is_primary": True,
                                "is_mock": False,
                                "methods": [
                                    "get_user",
                                    "create_user",
                                    "update_user",
                                    "delete_user",
                                ],
                            },
                            {
                                "name": "email_service",
                                "type_name": "EmailService",
                                "scope": "singleton",
                                "is_primary": False,
                                "is_mock": False,
                                "methods": ["send_email", "validate_email"],
                            },
                        ]
                    }
                },
            },
            404: {"description": "调试功能未启用或未找到Bean"},
            500: {"description": "获取Bean列表时发生内部错误"},
        },
    )
    def list_all_beans(self) -> List[BeanInfoResponse]:
        """
        列出所有已注册的Bean信息

        返回DI容器中所有已注册Bean的详细信息列表，包括：
        - name: Bean名称
        - type_name: Bean类型名称
        - scope: Bean作用域（singleton/prototype/factory）
        - is_primary: 是否为Primary Bean
        - is_mock: 是否为Mock Bean
        - methods: 可调用的公共方法列表

        **注意**：
        - 只在调试模式启用时可用（ENV=DEV）
        - 方法列表只包含不以下划线开头的公共方法
        - 如果获取某个Bean的方法列表失败，该Bean仍会返回但methods为空

        Returns:
            List[BeanInfoResponse]: Bean信息列表

        Raises:
            HTTPException: 当调试功能未启用或获取Bean列表失败时
        """
        self._ensure_debug_enabled()

        try:
            beans_info = []
            all_beans = self.container.list_all_beans_info()

            for bean_info in all_beans:
                try:
                    # 获取Bean实例以便获取方法列表
                    bean_instance = get_bean(bean_info['name'])
                    methods = self._get_bean_methods(bean_instance)

                    beans_info.append(
                        BeanInfoResponse(
                            name=bean_info['name'],
                            type_name=bean_info['type_name'],
                            scope=bean_info['scope'],
                            is_primary=bean_info['is_primary'],
                            is_mock=bean_info['is_mock'],
                            methods=methods,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "获取Bean '%s' 的方法列表失败: %s", bean_info['name'], str(e)
                    )
                    # 即使获取方法列表失败，也返回基本信息
                    beans_info.append(
                        BeanInfoResponse(
                            name=bean_info['name'],
                            type_name=bean_info['type_name'],
                            scope=bean_info['scope'],
                            is_primary=bean_info['is_primary'],
                            is_mock=bean_info['is_mock'],
                            methods=[],
                        )
                    )

            return beans_info

        except Exception as e:
            logger.error("列出所有Bean时发生错误: %s", str(e))
            logger.error(f"获取Bean列表异常详情：{str(e)}")
            raise HTTPException(
                status_code=500, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
            ) from e

    @post(
        "/call",
        extra_models=[BeanCallRequest, BeanCallResponse],
        response_model=BeanCallResponse,
        summary="调用指定Bean的指定方法",
        responses={
            200: {
                "description": "Bean方法调用成功",
                "content": {
                    "application/json": {
                        "examples": {
                            "traditional_way": {
                                "summary": "传统方式调用成功",
                                "value": {
                                    "success": True,
                                    "result": ["uuid1", "uuid2", "uuid3"],
                                    "bean_info": {
                                        "name": "resource_repository",
                                        "type_name": "SQLModelResourceRepositoryImpl",
                                        "lookup_method": "by_name",
                                    },
                                },
                            },
                            "code_execution_way": {
                                "summary": "代码执行方式调用成功",
                                "value": {
                                    "success": True,
                                    "result": [
                                        "d7a8782f-d35f-48fb-81fb-ce2fa3c01cdf",
                                        "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                                    ],
                                    "bean_info": {
                                        "name": "resource_repository",
                                        "type_name": "SQLModelResourceRepositoryImpl",
                                        "lookup_method": "by_name",
                                    },
                                    "code_execution": {
                                        "generated_args": [],
                                        "generated_kwargs": {
                                            "resource_ids": [274, 281, 282],
                                            "resource_type": "LITERATURE",
                                            "user_id": 1,
                                        },
                                    },
                                },
                            },
                            "failure": {
                                "summary": "调用失败，包含错误信息",
                                "value": {
                                    "success": False,
                                    "error": "Bean not found",
                                    "traceback": "Traceback (most recent call last):\n  File ...",
                                    "bean_info": None,
                                },
                            },
                        }
                    }
                },
            },
            400: {"description": "请求参数错误，如缺少必要参数、Bean标识符无效等"},
            404: {"description": "调试功能未启用、Bean不存在或方法不存在"},
            500: {"description": "方法调用过程中发生内部错误"},
        },
    )
    async def call_bean_method(self, request: BeanCallRequest) -> BeanCallResponse:
        """
        调用指定Bean的指定方法（兼容代码执行）

        支持两种参数传递方式：
        1. **传统方式**: 直接使用 `args` 和 `kwargs` 参数
        2. **代码执行**: 使用 `code` 参数动态生成参数（支持枚举类型和复杂对象）

        **Bean标识方式**：
        - bean_name/bean_type: 二选一，用于标识要调用的Bean

        **参数传递方式**：
        - 传统方式: 使用 `args` 和 `kwargs` 字段
        - 代码执行: 使用 `code` 字段编写Python代码生成参数

        **代码执行示例**：
        ```json
        {
            "bean_name": "resource_repository",
            "method": "get_uuids_by_ids_and_type",
            "code": "from domain.models.enums import ResourceType\\n\\nargs = []\\nkwargs = {\\n    'resource_ids': [274, 281, 282],\\n    'resource_type': ResourceType.LITERATURE,\\n    'user_id': 1\\n}"
        }
        ```

        **传统方式示例**：
        ```json
        {
            "bean_name": "resource_repository",
            "method": "get_by_ids",
            "args": [],
            "kwargs": {
                "resource_ids": [274, 281, 282]
            }
        }
        ```

        Args:
            request: Bean方法调用请求

        Returns:
            BeanCallResponse: 方法调用结果

        Raises:
            HTTPException: 当调试功能未启用、参数错误、Bean不存在或方法调用失败时
        """
        self._ensure_debug_enabled()

        try:
            # 确定使用哪种参数方式
            if request.code:
                # 使用代码执行方式
                logger.info("使用代码执行方式生成参数")
                code_result = self._execute_parameter_code(request.code)
                args = code_result['args']
                kwargs = code_result['kwargs']
                code_execution_info = {
                    'generated_args': args,
                    'generated_kwargs': kwargs,
                }
            else:
                # 使用传统方式
                logger.info("使用传统参数方式")
                args = request.args
                kwargs = request.kwargs
                code_execution_info = None

            # 获取Bean实例和信息
            bean_instance, bean_info = self._get_bean_by_identifier(
                request.bean_name, request.bean_type
            )

            # 检查方法是否存在
            if not hasattr(bean_instance, request.method):
                logger.error(
                    f"Bean '{bean_info['name']}' 不存在方法 '{request.method}'"
                )
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
                )

            method_to_call = getattr(bean_instance, request.method)

            # 检查是否为可调用对象
            if not callable(method_to_call):
                logger.error(
                    f"Bean '{bean_info['name']}' 的属性 '{request.method}' 不是可调用对象"
                )
                raise HTTPException(
                    status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
                )

            # 调用方法
            logger.info(
                "调用Bean方法: %s.%s(args=%s, kwargs=%s)",
                bean_info['name'],
                request.method,
                args,
                kwargs,
            )

            # 检查是否为协程函数，兼容异步和同步方法
            if asyncio.iscoroutinefunction(
                method_to_call
            ) or inspect.iscoroutinefunction(method_to_call):
                result = await method_to_call(*args, **kwargs)
            else:
                result = method_to_call(*args, **kwargs)

            # 序列化结果
            serialized_result = self._serialize_result(result)

            # 构造响应
            response_data = {
                'success': True,
                'bean_info': bean_info,
                **serialized_result,
            }

            # 如果使用了代码执行，添加执行信息
            if code_execution_info:
                response_data['code_execution'] = code_execution_info

            return BeanCallResponse(**response_data)

        except HTTPException:
            # 重新抛出HTTP异常
            raise
        except Exception as e:
            # 捕获并处理其他异常
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            logger.error("调用Bean方法时发生错误: %s", error_msg)
            logger.debug("错误堆栈: %s", error_traceback)

            return BeanCallResponse(
                success=False,
                error=error_msg,
                traceback=error_traceback,
                bean_info=getattr(locals(), 'bean_info', None),
                code_execution=getattr(locals(), 'code_execution_info', None),
            )

    @get(
        "/beans/{bean_name}",
        extra_models=[BeanInfoResponse],
        response_model=BeanInfoResponse,
        summary="根据Bean名称获取详细信息",
        responses={
            200: {
                "description": "Bean信息获取成功",
                "content": {
                    "application/json": {
                        "example": {
                            "name": "user_service",
                            "type_name": "UserService",
                            "scope": "singleton",
                            "is_primary": True,
                            "is_mock": False,
                            "methods": [
                                "get_user",
                                "create_user",
                                "update_user",
                                "delete_user",
                                "list_users",
                                "validate_user",
                            ],
                        }
                    }
                },
            },
            404: {"description": "调试功能未启用或指定的Bean不存在"},
            500: {"description": "获取Bean信息时发生内部错误"},
        },
    )
    def get_bean_info(self, bean_name: str) -> BeanInfoResponse:
        """
        根据Bean名称获取详细信息

        通过Bean名称查询指定Bean的完整信息，包括类型、作用域、
        是否为Primary Bean、是否为Mock Bean以及所有可调用的公共方法列表。

        **返回信息包含**：
        - name: Bean名称
        - type_name: Bean的类型名称
        - scope: Bean作用域（singleton/prototype/factory）
        - is_primary: 是否为Primary Bean（当有多个同类型Bean时的首选Bean）
        - is_mock: 是否为Mock Bean（用于测试环境的模拟实现）
        - methods: 可调用的公共方法列表（不包含以下划线开头的私有方法）

        **使用场景**：
        - 查看特定Bean的详细配置信息
        - 了解Bean提供的所有可调用方法
        - 调试DI容器中Bean的注册状态

        **注意事项**：
        - 只在调试模式启用时可用（ENV=DEV）
        - Bean名称必须完全匹配，区分大小写
        - 方法列表按字母顺序排序

        Args:
            bean_name: Bean的注册名称，必须与DI容器中的名称完全匹配

        Returns:
            BeanInfoResponse: Bean的详细信息，包含元数据和方法列表

        Raises:
            HTTPException: 当调试功能未启用、Bean不存在或获取信息失败时
        """
        self._ensure_debug_enabled()

        try:
            bean_instance, _ = self._get_bean_by_identifier(bean_name, None)
            methods = self._get_bean_methods(bean_instance)

            # 从容器获取Bean的元信息
            all_beans = self.container.list_all_beans_info()
            bean_meta = next((b for b in all_beans if b['name'] == bean_name), None)

            if not bean_meta:
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_NOT_FOUND.value
                )

            return BeanInfoResponse(
                name=bean_meta['name'],
                type_name=bean_meta['type_name'],
                scope=bean_meta['scope'],
                is_primary=bean_meta['is_primary'],
                is_mock=bean_meta['is_mock'],
                methods=methods,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("获取Bean信息时发生错误: %s", str(e))
            raise HTTPException(
                status_code=500, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
            ) from e

    @post(
        "/call-with-code",
        extra_models=[BeanCallWithCodeRequest, BeanCallResponse],
        response_model=BeanCallResponse,
        summary="通过Python代码生成参数调用Bean方法",
        responses={
            200: {
                "description": "Bean方法调用成功",
                "content": {
                    "application/json": {
                        "examples": {
                            "success_with_enum": {
                                "summary": "使用枚举参数调用成功",
                                "value": {
                                    "success": True,
                                    "result": ["uuid1", "uuid2", "uuid3"],
                                    "bean_info": {
                                        "name": "resource_repository",
                                        "type_name": "SQLModelResourceRepositoryImpl",
                                        "lookup_method": "by_name",
                                    },
                                    "code_execution": {
                                        "generated_args": [],
                                        "generated_kwargs": {
                                            "resource_ids": [274, 281, 282],
                                            "resource_type": "LITERATURE",
                                            "user_id": 1,
                                        },
                                    },
                                },
                            }
                        }
                    }
                },
            },
            400: {"description": "请求参数错误或代码执行失败"},
            404: {"description": "调试功能未启用、Bean不存在或方法不存在"},
            500: {"description": "方法调用过程中发生内部错误"},
        },
    )
    async def call_bean_method_with_code(
        self, request: BeanCallWithCodeRequest
    ) -> BeanCallResponse:
        """
        通过Python代码生成参数调用Bean方法

        这个接口允许你编写Python代码来动态生成方法参数，特别适用于：
        1. **枚举类型参数**: 如ResourceType.LITERATURE
        2. **复杂对象构造**: 如AIInputValueObject实例
        3. **动态参数计算**: 根据逻辑生成参数值
        4. **类型转换**: 处理JSON无法直接表示的Python类型

        **代码执行环境**:
        - 提供安全的执行环境，限制可用的内置函数
        - 自动导入常用枚举类型：ResourceType, ResourceScope, ResourceProcessingStatus
        - 自动导入常用值对象：AIInputValueObject
        - 代码必须定义 `args` 和/或 `kwargs` 变量

        **代码示例**:
        ```python
        # 示例1: 使用枚举类型
        args = []
        kwargs = {
            'resource_ids': [274, 281, 282],
            'resource_type': ResourceType.LITERATURE,
            'user_id': 1
        }

        # 示例2: 构造复杂对象
        ai_input = AIInputValueObject({
            'literature_refs': [
                {'value': {'id': 280}},
                {'value': {'id': 'uuid-string'}}
            ]
        })
        args = [ai_input]
        kwargs = {'user_id': 1}

        # 示例3: 动态计算参数
        resource_ids = list(range(270, 285))  # 生成ID列表
        kwargs = {
            'resource_ids': resource_ids,
            'resource_type': ResourceType.LITERATURE,
            'user_id': 1
        }
        ```

        **安全限制**:
        - 禁用文件操作、网络访问等危险功能
        - 只能使用预定义的安全函数和导入的类型
        - 代码执行超时保护

        Args:
            request: 包含Bean标识、方法名和Python代码的请求

        Returns:
            BeanCallResponse: 方法调用结果，包含代码执行信息

        Raises:
            HTTPException: 当调试功能未启用、代码执行失败或方法调用失败时
        """
        self._ensure_debug_enabled()

        try:
            # 执行代码生成参数
            logger.info(
                "执行参数生成代码: %s",
                request.code[:100] + "..." if len(request.code) > 100 else request.code,
            )

            code_result = self._execute_parameter_code(request.code)
            args = code_result['args']
            kwargs = code_result['kwargs']

            logger.info("代码执行成功，生成参数: args=%s, kwargs=%s", args, kwargs)

            # 获取Bean实例和信息
            bean_instance, bean_info = self._get_bean_by_identifier(
                request.bean_name, request.bean_type
            )

            # 检查方法是否存在
            if not hasattr(bean_instance, request.method):
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
                )

            method_to_call = getattr(bean_instance, request.method)

            # 检查是否为可调用对象
            if not callable(method_to_call):
                raise HTTPException(
                    status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
                )

            # 调用方法
            logger.info(
                "调用Bean方法: %s.%s(args=%s, kwargs=%s)",
                bean_info['name'],
                request.method,
                args,
                kwargs,
            )

            # 检查是否为协程函数，兼容异步和同步方法
            if asyncio.iscoroutinefunction(
                method_to_call
            ) or inspect.iscoroutinefunction(method_to_call):
                result = await method_to_call(*args, **kwargs)
            else:
                result = method_to_call(*args, **kwargs)

            # 序列化结果
            serialized_result = self._serialize_result(result)

            # 添加代码执行信息
            response_data = {
                'success': True,
                'bean_info': bean_info,
                'code_execution': {'generated_args': args, 'generated_kwargs': kwargs},
                **serialized_result,
            }

            return BeanCallResponse(**response_data)

        except HTTPException:
            # 重新抛出HTTP异常
            raise
        except Exception as e:
            # 捕获并处理其他异常
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            logger.error("通过代码调用Bean方法时发生错误: %s", error_msg)
            logger.debug("错误堆栈: %s", error_traceback)

            return BeanCallResponse(
                success=False,
                error=error_msg,
                traceback=error_traceback,
                bean_info=getattr(locals(), 'bean_info', None),
                code_execution=getattr(locals(), 'code_result', None),
            )
