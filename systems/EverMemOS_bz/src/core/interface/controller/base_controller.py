import inspect
from abc import ABC
from typing import Any, Callable, List, Optional, Union, get_origin, get_args

from fastapi import APIRouter, FastAPI
from fastapi.openapi.utils import get_openapi

# 导入授权相关模块已移动到方法内部按需导入


def _create_route_decorator(http_method: str) -> Callable:
    """
    内部辅助函数，用于创建 FastAPI 的路由装饰器 (get, post, put, delete 等)。

    Args:
        http_method (str): HTTP 方法名 (例如, "GET", "POST").

    Returns:
        Callable: 一个装饰器，它接收 path 和其他 APIRouter.add_api_route 的参数。
    """

    def decorator(
        path: str, extra_models: Optional[List[Any]] = None, **kwargs: Any
    ) -> Callable:
        """
        response_class: 决定了响应的**"传输方式"和"底层类型"**。它控制 FastAPI 最终用哪个类来打包和发送HTTP响应。
        response_model: 决定了响应体的**"数据结构"和"验证规则"**。它用于数据过滤、格式转换，并自动生成API文档中的 schema。
        summary 和 responses: 完全用于**"API文档（OpenAPI）"**。它们不影响任何运行时代码的行为，只用来丰富和精确化生成的文档（如 Swagger UI 或 ReDoc）。
        """

        def wrapper(func: Callable) -> Callable:
            # 使用一个特殊的属性来标记函数，并存储路由信息
            # 这避免了全局注册表，使每个控制器都是自包含的
            setattr(func, "__route_info__", (path, [http_method], kwargs))
            # 存储extra_models用于后续OpenAPI生成
            setattr(func, "__extra_models__", extra_models or [])
            return func

        return wrapper

    return decorator


get = _create_route_decorator("GET")
post = _create_route_decorator("POST")
put = _create_route_decorator("PUT")
delete = _create_route_decorator("DELETE")
patch = _create_route_decorator("PATCH")
head = _create_route_decorator("HEAD")
options = _create_route_decorator("OPTIONS")


class BaseController(ABC):
    """
    控制器基类，支持通过装饰器自动注册路由。

    继承此类并使用 @get, @post 等装饰器来定义您的 API 端点。
    控制器初始化时会自动收集所有被装饰的路由。

    使用示例:
    ```python
    # a_controller.py
    from .base_controller import BaseController, get

    class UserController(BaseController):
        def __init__(self):
            super().__init__(prefix="/users", tags=["Users"])

        @get("/")
        def list_users(self):
            return [{"id": 1, "name": "user1"}]

    # app.py
    from fastapi import FastAPI
    from .a_controller import UserController

    app = FastAPI()
    controllers = [UserController()] # "扫描" 到的控制器列表

    for controller in controllers:
        controller.register_to_app(app)
    ```
    """

    # 类级别的安全配置提供者，子类可以重写此属性
    _security_config_provider: Optional[Callable[[], List[dict]]] = None

    def __init__(
        self,
        prefix: str = "",
        tags: Optional[List[str]] = None,
        default_auth: str = "require_user",
        **kwargs: Any,
    ):
        """
        初始化控制器。

        Args:
            prefix (str, optional): 此控制器下所有路由的公共前缀。
            tags (Optional[List[str]], optional): 用于 OpenAPI 文档分组的标签。
            default_auth (str, optional): 默认授权策略，可选值：
                - "require_user": 默认需要用户认证（默认值）
                - "require_anonymous": 默认允许匿名访问
                - "require_admin": 默认需要管理员权限
                - "require_signature": 默认需要HMAC签名验证
                - "none": 不应用任何默认授权
            **kwargs: 其他传递给 FastAPI APIRouter 的参数。
        """
        self.router = APIRouter(prefix=prefix, tags=tags, **kwargs)
        self._app: Optional[FastAPI] = None
        self._extra_models: List[Any] = []
        self._auth_routes: List[str] = []  # 存储需要认证的路由路径
        self._default_auth = default_auth  # 存储默认授权策略
        self._collect_routes()

    def _collect_routes(self):
        """
        遍历类的所有成员，查找并注册被路由装饰器标记的方法。
        根据 default_auth 参数应用相应的默认授权策略。
        """
        for _member_name, member in inspect.getmembers(self):
            if callable(member) and hasattr(member, "__route_info__"):
                path, methods, route_kwargs = getattr(member, "__route_info__")

                # 收集extra_models
                extra_models = getattr(member, "__extra_models__", [])
                self._extra_models.extend(extra_models)

                # 应用默认授权（如果还没有授权装饰器）
                authorized_member = self._apply_default_auth(member)

                # 检查是否需要认证
                if self._needs_authentication(authorized_member):
                    # 记录需要认证的路由路径，去掉路径参数中的类型说明
                    full_path = (
                        f"{self.router.prefix}{path}" if self.router.prefix else path
                    )
                    # 移除路径参数中的类型说明，例如 {resource_id:int} -> {resource_id}
                    clean_path = self._clean_path_types(full_path)
                    self._auth_routes.append(clean_path)

                self.router.add_api_route(
                    path, endpoint=authorized_member, methods=methods, **route_kwargs
                )

    def _apply_default_auth(self, func: Callable) -> Callable:
        """
        根据 default_auth 参数应用默认授权策略

        Args:
            func: 要检查的函数

        Returns:
            Callable: 应用了默认授权的函数（如果还没有授权装饰器）
        """
        # 检查函数是否已经有授权装饰器
        if hasattr(func, '__authorization_context__'):
            return func

        # 如果是绑定方法，需要获取原始函数
        if hasattr(func, '__func__'):
            # 这是一个绑定方法，检查原始函数是否已有授权装饰器
            if hasattr(func.__func__, '__authorization_context__'):
                return func

            # 获取原始函数并应用装饰器
            original_func = func.__func__
            decorated_func = self._get_auth_decorator()(original_func)
            # 重新绑定到实例
            return decorated_func.__get__(func.__self__, type(func.__self__))
        else:
            # 这是一个未绑定函数，直接应用装饰器
            return self._get_auth_decorator()(func)

    def _get_auth_decorator(self):
        """获取对应的授权装饰器"""
        if self._default_auth == "require_user":
            from core.authorize.decorators import require_user

            return require_user
        elif self._default_auth == "require_anonymous":
            from core.authorize.decorators import require_anonymous

            return require_anonymous
        elif self._default_auth == "require_admin":
            from core.authorize.decorators import require_admin

            return require_admin
        elif self._default_auth == "require_signature":
            from core.authorize.decorators import require_signature

            return require_signature
        elif self._default_auth == "none":
            # 不应用任何默认授权，返回一个身份装饰器
            return lambda x: x
        else:
            # 未知的授权策略，默认使用 require_user
            from core.authorize.decorators import require_user

            return require_user

    def _needs_authentication(self, func: Callable) -> bool:
        """
        检查函数是否需要认证

        Args:
            func: 要检查的函数

        Returns:
            bool: 是否需要认证
        """
        # 检查函数是否直接有授权上下文
        if hasattr(func, '__authorization_context__'):
            auth_context = func.__authorization_context__
            return auth_context.need_auth()

        return False

    def _clean_path_types(self, path: str) -> str:
        """
        清理路径参数中的类型说明

        将 {resource_id:int} 转换为 {resource_id}
        将 {user_id:str} 转换为 {user_id}

        Args:
            path: 包含类型说明的路径

        Returns:
            str: 清理后的路径
        """
        import re

        # 使用正则表达式匹配 {parameter:type} 格式并替换为 {parameter}
        return re.sub(r'\{([^}:]+):[^}]+\}', r'{\1}', path)

    def _get_security_config(self) -> List[dict]:
        """
        获取安全配置

        Returns:
            List[dict]: 安全配置列表
        """
        # 优先使用类级别的安全配置提供者
        if self._security_config_provider is not None:
            return self._security_config_provider()

        # 尝试从全局配置获取
        try:
            from capabilities.auth.supabase_auth.supabase_auth_openapi import (
                get_security_config,
            )

            return get_security_config()
        except ImportError:
            return [{"OAuth2PasswordBearer": []}]

    def _is_union_type(self, model: Any) -> bool:
        """检查是否为Union类型"""
        return get_origin(model) is Union

    def _get_union_args(self, union_type: Any) -> tuple:
        """获取Union类型的参数"""
        return get_args(union_type)

    def _get_model_name(self, model: Any) -> str:
        """获取模型名称"""
        if hasattr(model, '__name__'):
            return model.__name__
        elif hasattr(model, '_name'):
            return model._name
        else:
            return str(model)

    def _generate_union_schema(self, union_type: Any, union_name: str) -> dict:
        """
        为Union类型生成oneOf schema结构

        Args:
            union_type: Union类型
            union_name: Union类型的名称

        Returns:
            包含oneOf和discriminator的schema定义
        """
        union_args = self._get_union_args(union_type)

        # 生成oneOf数组
        one_of = []
        discriminator_mapping = {}

        for arg in union_args:
            if hasattr(arg, '__name__'):
                model_name = arg.__name__
                one_of.append({"$ref": f"#/components/schemas/{model_name}"})

                # 尝试获取discriminator字段值
                if hasattr(arg, 'model_fields') and 'type' in arg.model_fields:
                    # 获取type字段的literal值或enum值
                    type_field = arg.model_fields['type']
                    if (
                        hasattr(type_field, 'default')
                        and type_field.default is not None
                    ):
                        discriminator_mapping[type_field.default] = (
                            f"#/components/schemas/{model_name}"
                        )
                    elif hasattr(type_field.annotation, '__args__'):
                        # 处理Literal类型
                        literal_values = getattr(type_field.annotation, '__args__', ())
                        if literal_values:
                            discriminator_mapping[literal_values[0]] = (
                                f"#/components/schemas/{model_name}"
                            )

        schema = {"oneOf": one_of}

        # 只有在有discriminator映射时才添加discriminator
        if discriminator_mapping:
            schema["discriminator"] = {
                "propertyName": "type",
                "mapping": discriminator_mapping,
            }

        return schema

    def _custom_openapi_generator(self, app: FastAPI):
        """
        自定义OpenAPI生成器，处理extra_models和认证路由
        """

        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema

            # 生成基本的OpenAPI schema
            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                summary=getattr(app, 'summary', None),
                description=app.description,
                routes=app.routes,
            )

            # 确保components存在
            if "components" not in openapi_schema:
                openapi_schema["components"] = {}
            if "schemas" not in openapi_schema["components"]:
                openapi_schema["components"]["schemas"] = {}
            if "securitySchemes" not in openapi_schema["components"]:
                openapi_schema["components"]["securitySchemes"] = {}

            # 收集所有BaseController实例的extra_models和认证路由
            controllers = []

            # 遍历所有路由来查找BaseController实例
            def collect_controllers_from_routes(routes):
                for route in routes:
                    if hasattr(route, 'router') and hasattr(route.router, 'routes'):
                        # 这是一个include_router的情况，递归处理
                        collect_controllers_from_routes(route.router.routes)
                    elif hasattr(route, 'endpoint') and hasattr(
                        route.endpoint, '__self__'
                    ):
                        # 这是一个绑定方法，检查是否是BaseController实例
                        controller = route.endpoint.__self__
                        if (
                            isinstance(controller, BaseController)
                            and controller not in controllers
                        ):
                            controllers.append(controller)

            collect_controllers_from_routes(app.routes)

            # 处理所有控制器的extra_models
            for controller in controllers:
                self._process_controller_extra_models(controller, openapi_schema)

            # 添加安全模式定义到 OpenAPI schema
            self._add_security_schemes_to_openapi(controllers, openapi_schema)

            # 为需要认证的路由添加安全配置
            self._add_security_to_auth_routes(controllers, openapi_schema)

            app.openapi_schema = openapi_schema
            return app.openapi_schema

        return custom_openapi

    def _add_security_schemes_to_openapi(
        self, controllers: List['BaseController'], openapi_schema: dict
    ):
        """
        添加安全模式定义到 OpenAPI schema

        Args:
            controllers: 所有BaseController实例列表
            openapi_schema: OpenAPI schema字典
        """
        # 收集所有控制器使用的安全模式定义
        security_schemes = {}

        for controller in controllers:
            # 检查控制器是否有自定义的安全配置提供者
            if (
                hasattr(controller, '_security_config_provider')
                and controller._security_config_provider is not None
            ):
                try:
                    # 尝试获取安全模式定义（如果控制器支持的话）
                    if hasattr(controller, '_get_security_schemes'):
                        schemes = controller._get_security_schemes()
                        if schemes:
                            security_schemes.update(schemes)
                    else:
                        # 检查是否是 HMAC 签名认证
                        security_config = controller._security_config_provider()
                        if security_config and any(
                            "HMACSignature" in config for config in security_config
                        ):
                            # 导入 HMAC 安全模式定义
                            try:
                                from core.middleware.hmac_signature_middleware import (
                                    get_hmac_openapi_security_schemes,
                                )

                                hmac_schemes = get_hmac_openapi_security_schemes()
                                security_schemes.update(hmac_schemes)
                            except ImportError:
                                # 如果导入失败，使用默认的 HMAC 定义
                                security_schemes["HMACSignature"] = {
                                    "type": "apiKey",
                                    "in": "header",
                                    "name": "X-Signature",
                                    "description": "HMAC 签名认证",
                                }
                except Exception as e:
                    # 如果获取安全模式失败，记录错误但不影响其他功能
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"获取控制器 {controller.__class__.__name__} 的安全模式定义失败: {str(e)}"
                    )

        # 将安全模式定义添加到 OpenAPI schema
        if security_schemes:
            openapi_schema["components"]["securitySchemes"].update(security_schemes)

    def _add_security_to_auth_routes(
        self, controllers: List['BaseController'], openapi_schema: dict
    ):
        """
        为需要认证的路由添加安全配置

        Args:
            controllers: 所有BaseController实例列表
            openapi_schema: OpenAPI schema字典
        """
        # 收集所有需要认证的路由路径
        all_auth_routes = []
        for controller in controllers:
            if hasattr(controller, '_auth_routes'):
                all_auth_routes.extend(controller._auth_routes)

        # 获取安全配置
        security_config = self._get_security_config()

        # 为需要认证的路由添加安全配置
        if "paths" in openapi_schema:
            for path, path_item in openapi_schema["paths"].items():
                # 检查当前路径是否需要认证
                if path in all_auth_routes:
                    # 为所有HTTP方法添加安全配置
                    for method in [
                        "get",
                        "post",
                        "put",
                        "delete",
                        "patch",
                        "head",
                        "options",
                    ]:
                        if method in path_item:
                            path_item[method]["security"] = security_config

    def _process_controller_extra_models(self, controller, openapi_schema):
        """
        处理单个控制器的extra_models
        """
        if not hasattr(controller, '_extra_models'):
            return

        for model in controller._extra_models:
            if self._is_union_type(model):
                # 对于Union类型，我们需要查找其原始名称
                # 从控制器的模块中查找这个Union类型的变量名
                model_name = None
                if hasattr(controller, '__class__') and hasattr(
                    controller.__class__, '__module__'
                ):
                    import sys

                    module = sys.modules.get(controller.__class__.__module__)
                    if module:
                        for attr_name in dir(module):
                            attr_value = getattr(module, attr_name)
                            if attr_value is model:
                                model_name = attr_name
                                break

                # 如果还是找不到，使用默认名称
                if not model_name:
                    model_name = "Union"

                # 处理Union类型
                union_schema = self._generate_union_schema(model, model_name)
                openapi_schema["components"]["schemas"][model_name] = union_schema

                # 同时添加Union成员的schema
                union_args = self._get_union_args(model)
                for arg in union_args:
                    if hasattr(arg, 'model_json_schema'):
                        arg_name = self._get_model_name(arg)
                        if arg_name not in openapi_schema["components"]["schemas"]:
                            # 生成单个模型的schema
                            arg_schema = arg.model_json_schema(
                                ref_template="#/components/schemas/{model}"
                            )
                            # 提取$defs中的schemas
                            if '$defs' in arg_schema:
                                openapi_schema["components"]["schemas"].update(
                                    arg_schema['$defs']
                                )
                                del arg_schema['$defs']
                            # 添加主模型schema
                            openapi_schema["components"]["schemas"][
                                arg_name
                            ] = arg_schema
            else:
                # 处理普通模型
                model_name = self._get_model_name(model)
                if hasattr(model, 'model_json_schema'):
                    if model_name not in openapi_schema["components"]["schemas"]:
                        model_schema = model.model_json_schema(
                            ref_template="#/components/schemas/{model}"
                        )
                        # 提取$defs中的schemas
                        if '$defs' in model_schema:
                            openapi_schema["components"]["schemas"].update(
                                model_schema['$defs']
                            )
                            del model_schema['$defs']
                        # 添加主模型schema
                        openapi_schema["components"]["schemas"][
                            model_name
                        ] = model_schema

    def register_to_app(self, app: FastAPI):
        """
        将此控制器的路由注册到 FastAPI 应用实例中。

        Args:
            app (FastAPI): FastAPI 的主应用实例。
        """
        self._app = app
        app.include_router(self.router)

        # 每次注册控制器时都重新设置自定义OpenAPI生成器
        # 这样可以确保所有控制器的extra_models都被正确处理
        app.openapi = self._custom_openapi_generator(app)
        # 清除已缓存的schema，强制重新生成
        app.openapi_schema = None


# 使用示例：
#
# 1. 使用默认的 FastAPI Users 认证配置：
# class UserController(BaseController):
#     def __init__(self):
#         super().__init__(prefix="/users", tags=["Users"])
#
#     @get("/")
#     @require_user  # 需要认证
#     def list_users(self):
#         return [{"id": 1, "name": "user1"}]
#
# 2. 自定义安全配置提供者：
# class CustomAuthController(BaseController):
#     # 自定义安全配置提供者
#     _security_config_provider = lambda: [
#         {
#             "CustomAuth": []
#         }
#     ]
#
#     def __init__(self):
#         super().__init__(prefix="/custom", tags=["Custom"])
#
#     @get("/")
#     @require_user
#     def custom_endpoint(self):
#         return {"message": "Custom authenticated endpoint"}
#
# 3. 动态安全配置：
# class DynamicAuthController(BaseController):
#     def __init__(self, auth_type: str = "OAuth2PasswordBearer"):
#         super().__init__(prefix="/dynamic", tags=["Dynamic"])
#         self.auth_type = auth_type
#         # 动态设置安全配置提供者
#         self._security_config_provider = lambda: [
#             {
#                 self.auth_type: []
#             }
#         ]
