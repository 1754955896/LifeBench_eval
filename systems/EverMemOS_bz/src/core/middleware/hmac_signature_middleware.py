import hmac
import hashlib
import time
import os
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.authorize.enums import Role
from core.context.context import set_current_user_info, clear_current_user_context
from core.observation.logger import get_logger
from component.redis_provider import RedisProvider
from core.di import get_bean_by_type

logger = get_logger(__name__)


class HMACSignatureMiddleware(BaseHTTPMiddleware):
    """
    HMAC签名验证中间件

    验证请求的HMAC签名，确保请求的完整性和真实性。
    使用HTTP方法、URL路径和时间戳作为签名数据。
    时间窗口为5分钟，超过时间窗口的请求将被拒绝。
    """

    def __init__(
        self,
        app: ASGIApp,
        secret_key: str,
        time_window_minutes: int = 5,
        redis_provider: Optional[RedisProvider] = None,
    ):
        """
        初始化HMAC签名中间件

        Args:
            app: ASGI应用实例
            secret_key: HMAC签名的密钥
            time_window_minutes: 时间窗口（分钟），默认5分钟
            redis_provider: Redis提供者，用于防重放攻击
        """
        super().__init__(app)
        self.secret_key = secret_key.encode('utf-8')
        self.time_window_seconds = time_window_minutes * 60
        self._redis_provider = redis_provider

    @property
    def redis_provider(self) -> RedisProvider:
        return self._redis_provider or get_bean_by_type(RedisProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求的HMAC签名验证并设置用户上下文

        期望的请求头：
        - X-Timestamp: Unix时间戳（秒）
        - X-Nonce: 随机数（用于防重放攻击）
        - X-Signature: HMAC-SHA256签名（十六进制格式）

        签名数据格式：{METHOD}|{URL_PATH}|{TIMESTAMP}|{NONCE}

        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: 响应对象
        """
        # 清除可能存在的用户上下文
        clear_current_user_context()

        # 设置用户上下文token
        token = None

        # 第一步：尝试进行HMAC签名验证并设置用户上下文
        try:
            # 获取请求头中的时间戳、nonce和签名
            timestamp_header = request.headers.get("X-Timestamp")
            nonce_header = request.headers.get("X-Nonce")
            signature_header = request.headers.get("X-Signature")

            # 如果有签名相关的头部，进行验证
            if (
                timestamp_header and nonce_header and signature_header
            ) or signature_header == "1234567890":
                # 验证HMAC签名
                is_valid_signature = await self._verify_hmac_signature(
                    request, timestamp_header, nonce_header, signature_header
                )

                if is_valid_signature:
                    # 签名验证成功，设置SIGNATURE角色的用户上下文
                    user_data = {"user_id": -1, "role": Role.SIGNATURE.value}
                    token = set_current_user_info(user_data)
                    logger.info("已设置HMAC签名用户上下文: role=SIGNATURE")
                else:
                    logger.info("HMAC签名验证失败，未设置用户上下文")
            else:
                logger.info("未找到HMAC签名头部，跳过签名验证")

        except Exception as e:
            logger.error(
                "HMAC签名验证时发生异常: %s, "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s'",
                str(e),
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
            )
            # 签名验证失败不影响请求继续处理
            # 具体的权限检查由各个endpoint负责

        # 第二步：执行业务逻辑
        try:
            response = await call_next(request)
            return response

        except Exception as e:
            logger.error(f"业务逻辑处理异常: {str(e)}")
            # 业务逻辑异常，重新抛出让上层处理
            raise

        finally:
            # 清理用户上下文
            if token is not None:
                try:
                    clear_current_user_context(token)
                    logger.debug("已清理HMAC签名用户上下文")
                except Exception as reset_error:
                    logger.warning(
                        f"清理HMAC签名用户上下文时发生错误: {str(reset_error)}"
                    )

    async def _verify_hmac_signature(
        self,
        request: Request,
        timestamp_header: str,
        nonce_header: str,
        signature_header: str,
    ) -> bool:
        """
        验证HMAC签名

        Args:
            request: FastAPI请求对象
            timestamp_header: 时间戳头部值
            nonce_header: 随机数头部值
            signature_header: 签名头部值

        Returns:
            bool: 签名是否有效
        """

        if signature_header == "1234567890" and os.getenv("ENV") == "dev":
            return True

        try:
            # 解析时间戳
            request_timestamp = int(timestamp_header)
        except ValueError:
            logger.error(
                "HMAC签名验证失败 - 无效的时间戳格式: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s'",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
            )
            return False

        # 验证nonce不能为空
        if not nonce_header or not nonce_header.strip():
            logger.error(
                "HMAC签名验证失败 - X-Nonce头部为空或无效: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s'",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
            )
            return False

        # 防重放攻击：使用原子操作检查并存储nonce
        nonce_key = f"nonce:{nonce_header}"
        expire_seconds = self.time_window_seconds * 2

        if self.redis_provider:
            try:
                # 使用Redis的SET NX EX命令进行原子操作：如果key不存在则设置，否则返回False
                # 这样可以在一个原子操作中完成检查和存储，避免竞态条件
                nonce_stored = await self.redis_provider.set(
                    nonce_key, str(request_timestamp), ex=expire_seconds, nx=True
                )
                if not nonce_stored:
                    logger.error(
                        "HMAC签名验证失败 - 检测到重放攻击，nonce已被使用: "
                        "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                        "method='%s', url_path='%s', request_timestamp=%d, current_time=%d",
                        timestamp_header,
                        nonce_header,
                        signature_header,
                        request.method,
                        request.url.path,
                        request_timestamp,
                        int(time.time()),
                    )
                    return False
                logger.debug(
                    "nonce已存储到Redis: %s, 过期时间=%d秒",
                    nonce_header,
                    expire_seconds,
                )
            except Exception as e:
                logger.error("检查和存储nonce时发生错误: %s", str(e))
                # Redis不可用时，记录警告但不阻止请求（降级处理）
                logger.warning("Redis不可用，跳过nonce重放检查")

        # 验证时间窗口
        current_time = int(time.time())
        time_diff = abs(current_time - request_timestamp)

        if time_diff > self.time_window_seconds:
            logger.error(
                "HMAC签名验证失败 - 请求超出时间窗口: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s', current_time=%d, request_timestamp=%d, "
                "time_diff=%d秒, time_window=%d秒",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
                current_time,
                request_timestamp,
                time_diff,
                self.time_window_seconds,
            )
            return False

        # 构建签名数据
        method = request.method
        url_path = request.url.path
        signature_data = f"{method}|{url_path}|{request_timestamp}|{nonce_header}"

        # 计算预期的签名
        expected_signature = hmac.new(
            self.secret_key, signature_data.encode('utf-8'), hashlib.sha256
        ).hexdigest()

        # 验证签名
        if not hmac.compare_digest(signature_header, expected_signature):
            logger.error(
                "HMAC签名验证失败 - 签名不匹配: "
                "timestamp_header='%s', nonce_header='%s', signature_header='%s', "
                "method='%s', url_path='%s', signature_data='%s', "
                "expected_signature='%s', secret_key_length=%d",
                timestamp_header,
                nonce_header,
                signature_header,
                request.method,
                request.url.path,
                signature_data,
                expected_signature,
                len(self.secret_key),
            )
            return False

        logger.debug("HMAC签名验证成功: %s %s", method, url_path)
        return True


def get_hmac_security_config():
    """
    获取 HMAC 签名认证的 OpenAPI 安全配置

    Returns:
        List[dict]: OpenAPI 安全配置列表，定义了 HMAC 签名认证的要求
    """
    return [{"HMACSignature": []}]


def get_hmac_openapi_security_schemes():
    """
    获取 HMAC 签名认证的 OpenAPI 安全模式定义

    这个函数返回用于 OpenAPI components.securitySchemes 的配置，
    定义了 HMAC 签名认证的具体实现方式和参数。

    Returns:
        dict: OpenAPI securitySchemes 配置
    """
    return {
        "HMACSignature": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Signature",
            "description": """**HMAC 签名认证**

使用 HMAC-SHA256 算法对请求进行签名验证，确保请求的完整性和真实性。

**签名算法：**
1. 构建签名数据：`{HTTP_METHOD}|{URL_PATH}|{TIMESTAMP}|{NONCE}`
   - 示例：`POST|/finance/storage/sign/download|1755572417|7fe6a3edabb9c1383b6d75a72ffce2e5`
2. 使用 HMAC-SHA256 算法和共享密钥计算签名
3. 将签名转换为十六进制字符串

**必需的请求头：**
- `X-Timestamp`: Unix 时间戳（秒），用于防重放攻击
- `X-Nonce`: 随机数（用于防重放攻击，每次请求都应不同）
- `X-Signature`: HMAC-SHA256 签名（十六进制格式）

**防重放攻击机制：**
- 请求时间戳与服务器时间差不能超过 5 分钟
- 每个nonce只能使用一次，服务器会在Redis中记录已使用的nonce
- nonce在Redis中的过期时间为时间窗口的两倍（10分钟），确保安全性
- 重复使用相同nonce的请求将被拒绝

**签名示例：**
```bash
# 使用HMAC签名生成器工具生成签名
python tests/hmac_signature_generator.py -m POST -p "/finance/storage/sign/download" -k "abc-12345"

# 或者手动生成签名（假设密钥为 "abc-12345"）
TIMESTAMP=$(date +%s)
NONCE=$(openssl rand -hex 16)
SIGNATURE_DATA="POST|/finance/storage/sign/download|${TIMESTAMP}|${NONCE}"
SIGNATURE=$(echo -n "$SIGNATURE_DATA" | openssl dgst -sha256 -hmac "abc-12345" | cut -d' ' -f2)

curl -X POST "https://api.example.com/finance/storage/sign/download" \\
     -H "X-Timestamp: ${TIMESTAMP}" \\
     -H "X-Nonce: ${NONCE}" \\
     -H "X-Signature: ${SIGNATURE}"
```

**Python 代码示例：**
```python
import hmac
import hashlib
import secrets
import time

# 生成签名的完整示例
method = "POST"
url_path = "/finance/storage/sign/download"
timestamp = int(time.time())
nonce = secrets.token_hex(16)
secret_key = "abc-12345"

# 构建签名数据
signature_data = f"{method}|{url_path}|{timestamp}|{nonce}"
signature = hmac.new(
    secret_key.encode('utf-8'),
    signature_data.encode('utf-8'),
    hashlib.sha256
).hexdigest()

# 请求头
headers = {
    "X-Timestamp": str(timestamp),
    "X-Nonce": nonce,
    "X-Signature": signature
}
```

**开发环境快捷方式：**
- 当环境变量 `ENV=dev` 时，可以使用 `X-Signature: 1234567890` 作为测试签名
- 生产环境必须使用正确的 HMAC 签名

**安全注意事项：**
- 密钥必须保密，不得泄露
- 建议定期轮换密钥
- 确保客户端时钟与服务器时钟同步
- nonce应该使用加密安全的随机数生成器生成
- Redis用于存储nonce，确保Redis的安全性和可用性""",
            "x-example": {
                "X-Timestamp": "1755572417",
                "X-Nonce": "7fe6a3edabb9c1383b6d75a72ffce2e5",
                "X-Signature": "6c17b2d568d42b9e0a9df422133f3e84bf4c3aa9bed04400843822586f25e4cd",
            },
        }
    }


def create_hmac_middleware(
    secret_key: str,
    time_window_minutes: int = 5,
    redis_provider: Optional[RedisProvider] = None,
):
    """
    创建HMAC签名中间件的工厂函数

    Args:
        secret_key: HMAC签名的密钥
        time_window_minutes: 时间窗口（分钟），默认5分钟
        redis_provider: Redis提供者，用于防重放攻击

    Returns:
        Callable: 中间件构造函数
    """

    def middleware_factory(app: ASGIApp) -> HMACSignatureMiddleware:
        return HMACSignatureMiddleware(
            app, secret_key, time_window_minutes, redis_provider
        )

    return middleware_factory
