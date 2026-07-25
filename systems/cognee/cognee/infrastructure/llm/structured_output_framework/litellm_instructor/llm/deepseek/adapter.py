"""
Custom DeepSeek-style LLM adapter — bypasses litellm + instructor entirely.

Calls /chat/completions directly with aiohttp using OpenAI's
`response_format: {type: json_schema, ...}` to get structured output, and
auto-injects `extra_body: {thinking: {type: disabled}}` so DeepSeek's
reasoning mode doesn't fight structured-output calls.

This sidesteps the four-layer abstraction tangle (litellm + instructor +
llm_provider enum + pydantic-settings Mode validation) that otherwise
broke every time we tried to plug a Chinese OpenAI-compatible provider
into cognee's default LLMGateway path.

Activated by setting `LLM_PROVIDER=default` (sentinel) in env — see
`get_llm_client.py`.
"""
import json
import logging
import os
from typing import Any, TypeVar

import aiohttp
from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound="BaseModel | str")


class DeepSeekAdapter(LLMInterface):
    """Direct OpenAI-compatible HTTP adapter for DeepSeek and similar providers."""

    max_completion_tokens: int

    def __init__(
        self,
        model: str,
        api_key: str,
        endpoint: str,
        max_completion_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        # Strip 'provider/' prefix (e.g. 'openai/deepseek-v4-flash' →
        # 'deepseek-v4-flash') so the name sent to the API is always clean.
        self.model = model.rsplit("/", 1)[-1] if "/" in model else model
        self.api_key = api_key or ""
        self.endpoint = (endpoint or "").rstrip("/")
        # Match cognee convention: cap by LLMConfig.llm_max_completion_tokens.
        configured = int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", "32768"))
        self.max_completion_tokens = min(max_completion_tokens, configured)
        self.temperature = temperature

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        # DeepSeek / SiliconFlow / OpenRouter all expose OpenAI shape at
        # {endpoint}/chat/completions. If endpoint already ends in a path,
        # trust the user; otherwise append the default.
        base = self.endpoint or "https://api.openai.com/v1"
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        """Strip ```json ... ``` wrappers some providers add around JSON.

        DeepSeek generally returns raw JSON, but other OpenAI-compatible
        providers occasionally wrap; tolerate both.
        """
        if not content:
            return content
        s = content.strip()
        if s.startswith("```"):
            # Remove leading ``` or ```json and trailing ```
            first_line, _, rest = s.partition("\n")
            s = (rest if "\n" in s else s[len(first_line):]).strip()
            if s.endswith("```"):
                s = s[: -3].rstrip()
        return s

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.endpoint:
            raise RuntimeError(
                "DeepSeekAdapter: LLM_ENDPOINT not configured. "
                "Set it via cognee.yaml (llm.base_url) or $LLM_ENDPOINT."
            )
        if not self.api_key:
            raise RuntimeError(
                "DeepSeekAdapter: LLM_API_KEY not configured. "
                "Set it via the harness .env or $LLM_API_KEY."
            )

        timeout = aiohttp.ClientTimeout(total=600)  # 10min — graph extraction can be slow
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._url(), json=payload, headers=self._headers()) as resp:
                body_text = await resp.text()
                if resp.status >= 400:
                    # Surface server-side errors verbosely for debugging.
                    raise RuntimeError(
                        f"DeepSeekAdapter HTTP {resp.status} from {self._url()}: "
                        f"{body_text[:500]}"
                    )
                return json.loads(body_text)

    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[T],
    ) -> T:
        """Generate structured output, returning either str or response_model instance."""
        is_str = response_model is str

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": text_input},
            ],
            "max_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
        }
        if not is_str:
            # DeepSeek's API supports `response_format: {type: json_object}`
            # universally, but `response_format: {type: json_schema, schema: ...}`
            # is rejected by some model endpoints ("This response_format type is
            # unavailable now"). To stay portable, we ask for plain JSON
            # output and validate against the Pydantic model client-side. The
            # schema is summarized into the system prompt so the model knows
            # what shape to emit.
            schema = response_model.model_json_schema()
            fields = ", ".join(
                f'"{k}": <{_json_type(v)}>' for k, v in schema.get("properties", {}).items()
            )
            structured_prompt = (
                (system_prompt or "You are a helpful assistant.")
                + (
                    "\n\nRespond ONLY with valid JSON matching this schema: "
                    f"{{ {fields} }}. Do not include any other text."
                )
            )
            payload["messages"][0]["content"] = structured_prompt
            payload["response_format"] = {"type": "json_object"}

        # Always disable thinking-mode for DeepSeek-style reasoning models.
        payload["extra_body"] = {"thinking": {"type": "disabled"}}

        data = await self._post_chat(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"DeepSeekAdapter: unexpected response shape from {self._url()}: "
                f"{str(data)[:500]}"
            ) from exc

        if is_str:
            return content  # type: ignore[return-value]

        content = self._strip_code_fence(content)
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"DeepSeekAdapter: model returned non-JSON content. "
                f"raw={content[:300]!r}"
            ) from exc
        return response_model.model_validate(obj)

    async def create_transcript(self, input) -> TranscriptionReturnType | None:
        # DeepSeek doesn't do audio; cognee may call this but it's a no-op for us.
        logger.debug(
            "DeepSeekAdapter.create_transcript called — not implemented; returning None."
        )
        return None

    async def transcribe_image(self, input: str):
        # cognee's LLMInterface declares this abstract too. DeepSeek doesn't
        # have a vision-OCR path; cognee may not even reach here for our
        # LoCoMo eval, but we satisfy the abstract-method check.
        logger.debug("DeepSeekAdapter.transcribe_image called — not implemented.")
        return None


def _safe_schema_name(name: str) -> str:
    """OpenAI / DeepSeek require `json_schema.name` to match ^[a-zA-Z0-9_-]+$."""
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    s = "".join(safe).strip("_-") or "Response"
    if s[0].isdigit():
        s = "Model_" + s
    return s[:64]


def _json_type(prop_schema: dict) -> str:
    """Render a JSON Schema property into a short type-name token for the system prompt.

    Examples:
        {"type": "string"}                                 ->  "string"
        {"type": "array", "items": {"type": "integer"}}    ->  "array[int]"
        {"$ref": "#/$defs/Foo"}                            ->  "any"
    """
    t = prop_schema.get("type")
    if t:
        if t == "array":
            inner = prop_schema.get("items") or {}
            inner_t = inner.get("type") or _json_type(inner)
            return f"array[{inner_t}]"
        return t
    if "$ref" in prop_schema:
        return "any"
    if "enum" in prop_schema and prop_schema["enum"]:
        return " | ".join(repr(e) for e in prop_schema["enum"][:5])
    return "any"
