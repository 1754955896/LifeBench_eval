"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import re
import typing
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import EmptyResponseError, RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'

StructuredOutputMode = Literal['json_schema', 'json_object']


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    This client targets any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
    vLLM, llama.cpp, Ollama, DeepSeek, Together, etc.). It defaults to native
    ``json_schema`` structured output (constrained decoding) and can fall back to
    ``json_object`` for the minority of providers that do not support ``json_schema``.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        structured_output_mode (StructuredOutputMode): How structured output is requested.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
        structured_output_mode: StructuredOutputMode = 'json_schema',
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.
            structured_output_mode (StructuredOutputMode): Whether to request structured
                output via native ``json_schema`` (the default, uses constrained decoding)
                or to fall back to ``json_object``. Set to ``'json_object'`` for providers
                that do not support the ``json_schema`` response format (e.g. DeepSeek); in
                that mode the schema is injected into the prompt instead of being enforced
                by the API.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens
        self.structured_output_mode: StructuredOutputMode = structured_output_mode

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    @staticmethod
    def _build_example_and_fields(response_model: type[BaseModel]) -> tuple[str, str]:
        """Build an example JSON and field descriptions from a Pydantic model's JSON Schema.

        Produces a concrete example (e.g. ``{"items": [], "count": 0}``) and a numbered
        list of field descriptions, which is much easier for LLMs to follow than a raw
        JSON Schema document.  Raw schemas confuse models (especially in ``json_object``
        mode) because ``{"properties": ..., "type": "object"}`` looks like output the
        model should produce, causing schema echo.
        """
        full_schema = response_model.model_json_schema()
        properties = full_schema.get('properties', {})

        def _resolve_ref(ref: str) -> dict:
            """Resolve a ``$ref`` pointer against the full schema."""
            # ref format: "#/$defs/TypeName"
            parts = ref.split('/')
            node = full_schema
            for part in parts:
                if part == '#':
                    continue
                node = node.get(part, {})
            return node

        def _example_for(prop_schema: dict) -> Any:
            # Resolve $ref first
            if '$ref' in prop_schema:
                prop_schema = _resolve_ref(prop_schema['$ref'])

            # anyOf: use the first option (typically a single $ref)
            if 'anyOf' in prop_schema:
                prop_schema = prop_schema['anyOf'][0]
                if '$ref' in prop_schema:
                    prop_schema = _resolve_ref(prop_schema['$ref'])

            typ = prop_schema.get('type', 'object')

            if typ == 'array':
                item_schema = prop_schema.get('items', {})
                return [_example_for(item_schema)] if item_schema else []
            if typ == 'integer' or typ == 'number':
                return 0
            if typ == 'boolean':
                return False
            if typ == 'object':
                return _example_from_props(prop_schema.get('properties', {}))
            return ''

        def _example_from_props(props: dict) -> dict:
            return {name: _example_for(s) for name, s in props.items()}

        example = _example_from_props(properties)
        example_json = json.dumps(example, indent=2, ensure_ascii=False)

        desc_lines = []
        for name, prop_schema in properties.items():
            # Resolve $ref / anyOf for type display in descriptions
            display = prop_schema
            if '$ref' in display:
                display = _resolve_ref(display['$ref'])
            if 'anyOf' in display:
                display = display['anyOf'][0]
                if '$ref' in display:
                    display = _resolve_ref(display['$ref'])

            desc = display.get('description', '')
            typ = display.get('type', 'object')
            item_typ = ''
            if typ == 'array':
                items = display.get('items', {})
                # Resolve item type through $ref too
                if '$ref' in items:
                    resolved = _resolve_ref(items['$ref'])
                    item_typ = f' of {resolved.get("title", "object")}'
                elif 'anyOf' in items:
                    opt = items['anyOf'][0]
                    if '$ref' in opt:
                        resolved = _resolve_ref(opt['$ref'])
                        item_typ = f' of {resolved.get("title", "object")}'
                else:
                    item_typ = f' of {items.get("type", "string")}'
            desc_lines.append(f'- {name} ({typ}{item_typ}): {desc}')

        return example_json, '\n'.join(desc_lines)

    def _build_response_format(self, response_model: type[BaseModel] | None) -> dict[str, Any]:
        """Build the ``response_format`` payload for the chat completion request.

        Uses native ``json_schema`` when a response model is provided and the client is in
        ``json_schema`` mode; otherwise falls back to ``json_object``. In ``json_object``
        mode the schema is not enforced by the API — ``generate_response`` injects it into
        the prompt instead.
        """
        if response_model is None or self.structured_output_mode == 'json_object':
            return {'type': 'json_object'}

        # Native json_schema. We intentionally omit "strict": true — strict mode requires
        # the schema to meet OpenAI's strict subset (additionalProperties: false, every
        # field required), which raw model_json_schema() routinely violates (that's why the
        # dedicated OpenAIClient uses responses.parse() instead). So adherence is best-effort
        # on OpenAI-proper; constrained-decoding servers (vLLM, llama.cpp) still enforce it.
        return {
            'type': 'json_schema',
            'json_schema': {
                'name': getattr(response_model, '__name__', 'structured_response'),
                'schema': response_model.model_json_schema(),
            },
        }

    @staticmethod
    def _safe_json_parse(text: str) -> dict[str, Any]:
        """Parse JSON with progressive repair for common LLM output issues.

        Handles trailing commas, extra text before/after the JSON object,
        and truncated outputs. Code fences are assumed already stripped.
        """
        text = text.strip()
        # Try raw parse first (most common path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Extract JSON object from surrounding text
        obj_start = text.find('{')
        arr_start = text.find('[')
        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            start = obj_start
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        text = text[start:i + 1]
                        break
        # Remove trailing commas (common LLM mistake)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(text)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip a wrapping markdown code fence from a JSON payload.

        OpenAI-compatible models served via Ollama/llama.cpp etc. frequently wrap their
        output in a ```json … ``` fence even when a json_schema/json_object response_format
        is requested, which breaks a bare ``json.loads``. No-op when there is no fence.
        """
        stripped = text.strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```[a-zA-Z0-9_-]*[ \t]*\r?\n?', '', stripped)
            stripped = re.sub(r'\r?\n?```[ \t]*$', '', stripped)
        return stripped.strip()

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=self._build_response_format(response_model),  # type: ignore[arg-type]
            )
            result = response.choices[0].message.content or ''
            # An empty body (refusal, length finish_reason, or a flaky endpoint) would make
            # json.loads raise a cryptic JSONDecodeError; surface a clear error instead.
            if not result:
                raise EmptyResponseError('LLM returned an empty response')
            # Many OpenAI-compatible/local models wrap JSON in a ```json fence even under a
            # structured response_format; strip it before parsing.
            parsed = self._safe_json_parse(self._strip_code_fences(result))

            # Detect JSON Schema echo: some models (DeepSeek etc.) in json_object mode
            # sometimes return the schema itself instead of data conforming to it.
            # A schema dict has "type" and "properties" keys; real data does not.
            if (
                isinstance(parsed, dict)
                and parsed.get('type') == 'object'
                and 'properties' in parsed
                and response_model is not None
            ):
                raise EmptyResponseError(
                    'Model returned JSON Schema instead of data — retrying'
                )

            return parsed
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, typing.Any]:
        self._apply_attribute_extraction_preamble(messages, attribute_extraction)
        if max_tokens is None:
            max_tokens = self.max_tokens

        # In json_object fallback mode the API does not enforce the schema. Instead of
        # injecting the raw JSON Schema (which confuses models — they echo back the
        # {"properties": ..., "type": "object"} structure instead of producing data),
        # we provide a concrete example and field descriptions.
        if response_model is not None and self.structured_output_mode == 'json_object':
            example_json, fields_desc = self._build_example_and_fields(response_model)
            messages[-1].content += (
                f'\n\nRespond with a JSON object that follows this structure:\n'
                f'{example_json}\n\n'
                f'Field descriptions:\n{fields_desc}\n\n'
                f'IMPORTANT: Output ONLY a JSON object with the fields shown above. '
                f'Do NOT include "properties", "type", or "required" keys. '
                f'Do NOT wrap in markdown code fences.'
            )

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'openai',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            try:
                # Delegate to the base tenacity wrapper so transient JSONDecodeError /
                # RateLimitError get backoff-retried (4 attempts) — most relevant in the
                # json_object fallback path for less-reliable providers. This is the clean
                # retry mechanism (same pattern as Gliner2Client); the old hand-rolled
                # re-prompt loop is intentionally not reinstated.
                return await self._generate_response_with_retry(
                    messages, response_model, max_tokens=max_tokens, model_size=model_size
                )
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise
