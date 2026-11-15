from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from fastapi import Request, HTTPException

from core.di.decorators import component


class AuthProvider(ABC):
    """认证提供者接口，负责处理authorization header和用户上下文"""

    @abstractmethod
    async def get_optional_user_data_from_request(
        self, request: Request
    ) -> Optional[Dict[str, Any]]:
        """
        从请求中提取完整的用户数据（可选）

        Args:
            request: FastAPI请求对象

        Returns:
            Optional[Dict[str, Any]]: 用户数据，包含user_id、role等信息，如果不存在或无效则返回None
        """


@component(name="auth_provider")
class TestAuthProviderImpl(AuthProvider):
    """认证提供者实现，负责处理authorization header和用户上下文"""

    def __init__(self):
        """初始化认证提供者"""

    async def get_user_id_from_request(self, request: Request) -> int:
        """
        从请求中提取用户ID

        目前实现：直接从authorization header中获取用户ID（临时方案）
        未来扩展：可以支持JWT token解析等

        Args:
            request: FastAPI请求对象

        Returns:
            int: 用户ID

        Raises:
            HTTPException: 当authorization header缺失或无效时
        """
        # 从authorization header中获取用户ID
        auth_header = request.headers.get("authorization")

        if not auth_header:
            raise HTTPException(status_code=401, detail="缺少authorization header")

        # 移除可能的"Bearer "前缀
        user_id_str = auth_header.replace("Bearer ", "").strip()

        try:
            user_id = int(user_id_str)
            if user_id <= 0:
                raise ValueError("用户ID必须是正整数")
            return user_id
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="authorization header中的用户ID格式无效，应该是正整数",
            )

    async def get_optional_user_data_from_request(
        self, request: Request
    ) -> Optional[Dict[str, Any]]:
        """
        从请求中提取完整的用户数据（可选）

        Args:
            request: FastAPI请求对象

        Returns:
            Optional[Dict[str, Any]]: 用户数据，包含user_id、role等信息，如果不存在或无效则返回None
        """
        try:
            user_id = await self.get_user_id_from_request(request)
            # 导入 Role 枚举
            from core.authorize.enums import Role

            return {"user_id": user_id, "role": Role.USER.value}
        except HTTPException:
            return None
