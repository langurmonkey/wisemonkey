"""Model API router for Wisemonkey.

Provides a unified interface over multiple LLM providers using their
native Python SDKs:
- OpenAI (api.openai.com) — openai SDK
- Anthropic (api.anthropic.com) — anthropic SDK
- Ollama (localhost:11434) — ollama SDK
- LM Studio (localhost:1234) — openai SDK (OpenAI-compatible)

Supported features:
- Messages (inference) with streaming enabled/disabled
- Temperature control
- Thinking/reasoning effort (provider-specific)
- List available models

Provider detection:
1. Explicit `model.provider` in config.yaml (e.g. "openai", "anthropic", etc.)
2. Auto-detection from well-known base_urls
3. Falls back to "generic" (OpenAI-compatible)
"""

import json
import os
from enum import Enum
from typing import Any, Optional

from agent.config import get_config


class Provider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    GENERIC = "generic"  # Unknown OpenAI-compatible endpoint


# Known base URLs for autodetection (normalized, no trailing slash)
_PROVIDER_URLS = [
    ("https://api.openai.com/v1", Provider.OPENAI),
    ("https://api.anthropic.com", Provider.ANTHROPIC),
    ("http://localhost:11434", Provider.OLLAMA),
    ("http://127.0.0.1:11434", Provider.OLLAMA),
    ("http://localhost:1234/v1", Provider.LMSTUDIO),
    ("http://127.0.0.1:1234/v1", Provider.LMSTUDIO),
]

# Provider-specific API key environment variables
_PROVIDER_KEYS = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OLLAMA: None,    # No key needed
    Provider.LMSTUDIO: None,  # No key needed
    Provider.GENERIC: "OPENAI_API_KEY",
}

# Anthropic thinking effort -> budget_tokens mapping
_THINKING_BUDGET = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
}


def _detect_provider(base_url: str) -> Provider:
    """Auto-detect provider from base URL."""
    normalized = base_url.rstrip("/")
    for url, provider in _PROVIDER_URLS:
        if normalized == url.rstrip("/"):
            return provider
    return Provider.GENERIC


def _get_api_key(provider: Provider) -> str:
    """Get the appropriate API key for the provider."""
    env_var = _PROVIDER_KEYS.get(provider)
    if env_var is None:
        return ""
    return os.environ.get(env_var, "")


# ---------------------------------------------------------------------------
# Response wrappers — make non-OpenAI providers look like OpenAI responses
# so the rest of the agent (stream handler, tool parsing) works unchanged.
# ---------------------------------------------------------------------------

class _Delta:
    def __init__(self, data: dict):
        self.content = data.get("content")
        self.role = data.get("role")
        self.tool_calls = data.get("tool_calls")
        self.reasoning_content = data.get("reasoning_content")


class _StreamChoice:
    def __init__(self, data: dict):
        self.delta = _Delta(data.get("delta", {}))
        self.finish_reason = data.get("finish_reason")
        self.index = data.get("index", 0)


class _StreamChunk:
    """Mimics OpenAI ChatCompletionChunk for streaming responses."""
    def __init__(self, data: dict):
        self.choices = [_StreamChoice(c) for c in data.get("choices", [])]
        self.model = data.get("model", "")


class _ToolCallFunction:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments  # keep as-is; caller decides the type


class _ToolCallDelta:
    def __init__(self, index: int = 0, id: str = "",
                 function_name: str = "", function_arguments: str = ""):
        self.index = index
        self.id = id
        self.type = "function"
        self.function = _ToolCallFunction(function_name, function_arguments)


class _Message:
    def __init__(self, content: str, tool_calls: list | None = None):
        self.content = content
        self.role = "assistant"
        self.tool_calls = tool_calls


class _ResponseChoice:
    def __init__(self, content: str, tool_calls: list | None = None):
        self.message = _Message(content, tool_calls)
        self.finish_reason = "stop"
        self.index = 0


class _Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Response:
    """Mimics OpenAI ChatCompletion for non-streaming responses."""
    def __init__(self, content: str, model: str = "",
                 prompt_tokens=0, completion_tokens=0,
                 tool_calls: list | None = None):
        self.model = model
        self.choices = [_ResponseChoice(content, tool_calls)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class ModelRouter:
    """Unified interface for multiple LLM providers.

    Creates the appropriate SDK client for the detected provider and
    normalises responses into an OpenAI-like shape so the rest of the
    agent doesn't need to care about which provider is behind it.

    Usage:
        router = ModelRouter(config)
        ok, err = router.initialize()

        # Streaming chat
        for chunk in router.chat(messages, stream=True):
            ...

        # Non-streaming chat
        response = router.chat(messages, stream=False)

        # List models
        models = router.list_models()
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.provider: Provider = Provider.GENERIC

        # SDK clients — only the one matching self.provider will be created
        self._openai_client: Optional[Any] = None
        self._anthropic_client: Optional[Any] = None
        self._ollama_client: Optional[Any] = None

        # Config-derived settings
        self._model_name: str = ""
        self._base_url: str = ""
        self._temperature: float = 0.8
        self._thinking_effort: str = "medium"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> tuple[bool, str | None]:
        """Initialise the router and create the provider-specific client.

        Returns:
            (ok, error_message) tuple.
        """
        self._base_url = self.config.get("model.base_url", "").strip()
        self._model_name = self.config.get("model.name", "")
        self._temperature = self.config.get("model.temperature", 0.8)
        self._thinking_effort = self.config.get("model.thinking.effort", "medium")

        # 1. Detect provider
        explicit = self.config.get("model.provider", "").strip().lower()
        if explicit:
            try:
                self.provider = Provider(explicit)
            except ValueError:
                return False, f"Unknown provider: {explicit}"
        else:
            self.provider = _detect_provider(self._base_url)

        # 2. Resolve API key
        api_key = _get_api_key(self.provider)
        if not api_key and self.provider in (Provider.OPENAI, Provider.ANTHROPIC):
            env_var = _PROVIDER_KEYS[self.provider]
            return False, (
                f"Missing API key for {self.provider.value}. "
                f"Set {env_var} in your environment or .env file."
            )

        # 3. Create the *single* client needed for this provider
        try:
            if self.provider == Provider.ANTHROPIC:
                import anthropic as anthropic_sdk
                self._anthropic_client = anthropic_sdk.Anthropic(
                    api_key=api_key,
                )

            elif self.provider == Provider.OLLAMA:
                import ollama as ollama_sdk
                base = self._base_url or "http://localhost:11434"
                self._ollama_client = ollama_sdk.Client(host=base)

            else:
                # OpenAI, LM Studio, Generic — all use the openai SDK
                base_url = self._base_url
                if base_url and not base_url.endswith("/v1"):
                    base_url = base_url.rstrip("/") + "/v1"
                import openai as openai_sdk
                self._openai_client = openai_sdk.OpenAI(
                    api_key=api_key or "dummy",
                    base_url=base_url or None,
                )

        except Exception as e:
            return False, f"Failed to create {self.provider.value} client: {e}"

        return True, None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        self._model_name = name
        self.config.set("model.name", name)

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = value
        self.config.set("model.temperature", value)

    @property
    def thinking_effort(self) -> str:
        return self._thinking_effort

    @thinking_effort.setter
    def thinking_effort(self, effort: str):
        self._thinking_effort = effort
        self.config.set("model.thinking.effort", effort)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], stream: bool = True,
             tools: list[dict] | None = None,
             model: str | None = None,
             temperature: float | None = None,
             thinking_effort: str | None = None) -> Any:
        """Send a chat completion request.

        Args:
            messages: List of message dicts (role, content).
            stream: If True, returns a generator of _StreamChunk objects.
            tools: Optional tool definitions for function calling.
            model: Override the model name.
            temperature: Override the temperature.
            thinking_effort: Override the thinking effort.

        Returns:
            If stream=True: generator of _StreamChunk objects.
            If stream=False: _Response object.
        """
        model_name = model if model is not None else self._model_name
        temp = temperature if temperature is not None else self._temperature
        thinking = thinking_effort if thinking_effort is not None else self._thinking_effort

        if self.provider == Provider.ANTHROPIC:
            return self._chat_anthropic(messages, model_name, temp, thinking,
                                        stream, tools)
        elif self.provider == Provider.OLLAMA:
            return self._chat_ollama(messages, model_name, temp, thinking,
                                     stream, tools)
        else:
            # OpenAI, LM Studio, Generic
            return self._chat_openai(messages, model_name, temp, thinking,
                                     stream, tools)

    # ------------------------------------------------------------------
    # OpenAI / LM Studio / Generic
    # ------------------------------------------------------------------

    def _chat_openai(self, messages, model_name, temp, thinking,
                     stream, tools) -> Any:
        """OpenAI-compatible chat (OpenAI, LM Studio, Generic)."""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not initialised")

        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "stream": stream,
        }
        
        kwargs["extra_headers"] = {
            "HTTP-Referer": "https://wisemonkey.ai", # Must be a full URL
            "X-Title": "Wisemonkey Agent",
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Extra body params (reasoning_effort, etc.)
        kwargs["extra_body"] = {
            "reasoning_effort": thinking
        }

        return self._openai_client.chat.completions.create(**kwargs)



    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _adapt_messages_for_anthropic(self, messages: list[dict]
                                      ) -> tuple[list[dict], str | None]:
        """Extract system messages (Anthropic uses a separate 'system' param)."""
        system_parts = []
        chat_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)
            else:
                chat_messages.append(msg)
        system_text = "\n\n".join(system_parts) if system_parts else None
        return chat_messages, system_text

    def _convert_tools_for_anthropic(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI tool format to Anthropic tool format."""
        if not tools:
            return None
        result = []
        for t in tools:
            if t.get("type") == "function":
                fn = t.get("function", {})
                result.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
        return result if result else None

    def _chat_anthropic(self, messages, model_name, temp, thinking,
                        stream, tools):
        """Call Anthropic's Messages API via the official SDK."""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not initialised")

        # Adapt messages: extract system prompt
        chat_msgs, system = self._adapt_messages_for_anthropic(messages)
        anthropic_tools = self._convert_tools_for_anthropic(tools)

        # Anthropic requires temperature=1 when thinking is enabled.
        thinking_enabled = thinking and thinking != "none"
        effective_temp = 1.0 if thinking_enabled else temp

        kwargs: dict = {
            "model": model_name,
            "max_tokens": 8192,
            "messages": chat_msgs,
            "temperature": effective_temp,
        }

        if system:
            kwargs["system"] = system

        if thinking_enabled:
            budget = _THINKING_BUDGET.get(thinking, 4096)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if stream:
            return self._anthropic_stream(kwargs)
        else:
            return self._anthropic_non_stream(kwargs)

    def _anthropic_stream(self, kwargs):
        """Stream from Anthropic, yielding _StreamChunk objects.

        Handles text, thinking blocks, and tool use by iterating raw events
        rather than the text-only stream helper, so nothing is silently dropped.
        """
        # Accumulate tool call inputs per block index as they stream in
        tool_input_buffers: dict[int, dict] = {}  # index -> {id, name, json_buf}

        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not initialised")

        with self._anthropic_client.messages.stream(**kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", None)

                # --- Thinking delta ---
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if not delta:
                        continue

                    delta_type = getattr(delta, "type", None)

                    if delta_type == "thinking_delta":
                        chunk_data = {
                            "choices": [{
                                "delta": {"reasoning_content": delta.thinking},
                                "finish_reason": None,
                                "index": 0,
                            }],
                        }
                        yield _StreamChunk(chunk_data)

                    elif delta_type == "text_delta":
                        chunk_data = {
                            "choices": [{
                                "delta": {"content": delta.text},
                                "finish_reason": None,
                                "index": 0,
                            }],
                        }
                        yield _StreamChunk(chunk_data)

                    elif delta_type == "input_json_delta":
                        # Accumulate partial JSON for this tool block
                        idx = event.index
                        if idx in tool_input_buffers:
                            tool_input_buffers[idx]["json_buf"] += delta.partial_json

                # --- Track tool block openings so we know id/name ---
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if not block:
                        continue

                    if getattr(block, "type", None) == "tool_use":
                        tool_input_buffers[event.index] = {
                            "id": block.id,
                            "name": block.name,
                            "json_buf": "",
                        }

                # --- Emit completed tool call when its block closes ---
                elif event_type == "content_block_stop":
                    idx = event.index
                    if idx in tool_input_buffers:
                        buf = tool_input_buffers.pop(idx)
                        tc = _ToolCallDelta(
                            index=idx,
                            id=buf["id"],
                            function_name=buf["name"],
                            function_arguments=buf["json_buf"],
                        )
                        chunk_data = {
                            "choices": [{
                                "delta": {"tool_calls": [tc]},
                                "finish_reason": None,
                                "index": 0,
                            }],
                        }
                        yield _StreamChunk(chunk_data)

        # Final chunk with finish_reason
        final_msg = stream.get_final_message()
        chunk_data = {
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
                "index": 0,
            }],
            "model": final_msg.model or kwargs.get("model", ""),
        }
        yield _StreamChunk(chunk_data)

    def _anthropic_non_stream(self, kwargs):
        """Non-streaming call to Anthropic, returns _Response."""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not initialised")

        message = self._anthropic_client.messages.create(**kwargs)
        text = ""
        thinking_text = ""
        tool_calls = []
        for block in message.content:
            if block.type == "thinking":
                thinking_text += block.thinking
            elif block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
        return _Response(
            content=text,
            model=message.model,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            tool_calls=tool_calls or None,
        )

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def _chat_ollama(self, messages, model_name, temp, thinking, stream, tools):
        if self._ollama_client is None:
            raise RuntimeError("Ollama client not initialised")

        # Ollama's Pydantic Message model requires tool_calls[].function.arguments
        # to be a dict, not a JSON string. Deserialise before sending.
        adapted = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                fixed_tcs = []
                for tc in msg["tool_calls"]:
                    args = tc.get("function", {}).get("arguments", {})
                    fixed_tcs.append({
                        **tc,
                        "function": {
                            **tc["function"],
                            "arguments": json.loads(args) if isinstance(args, str) else args,
                        },
                    })
                adapted.append({**msg, "tool_calls": fixed_tcs})
            else:
                adapted.append(msg)

        options = {"temperature": temp}
        if thinking and thinking != "none":
            options["reasoning_effort"] = thinking

        kwargs = {
            "model": model_name,
            "messages": adapted,   # <-- use adapted, not messages
            "options": options,
        }
        if tools:
            kwargs["tools"] = tools

        if stream:
            return self._ollama_stream(**kwargs)
        else:
            return self._ollama_non_stream(**kwargs)

    def _ollama_stream(self, **kwargs):
        """Stream from Ollama, yielding _StreamChunk objects."""
        if self._ollama_client is None:
            raise RuntimeError("Ollama client not initialised")

        response = self._ollama_client.chat(stream=True, **kwargs)
        for part in response:
            delta = {}
            if part.get("message", {}).get("content"):
                delta["content"] = part["message"]["content"]

            # Detect tool calls in streaming
            tool_calls_data = part.get("message", {}).get("tool_calls")
            tool_calls_deltas = None
            if tool_calls_data:
                tc_list = []
                for i, tc in enumerate(tool_calls_data):
                    tc_list.append(_ToolCallDelta(
                        index=i,
                        id=tc.get("id", ""),
                        function_name=tc.get("function", {}).get("name", ""),
                        function_arguments=tc.get("function", {}).get("arguments", {}),
                    ))
                tool_calls_deltas = tc_list

            chunk_data = {
                "choices": [{
                    "delta": delta,
                    "finish_reason": None,
                    "index": 0,
                }],
                "model": part.get("model", ""),
            }
            if tool_calls_deltas:
                chunk_data["choices"][0]["delta"]["tool_calls"] = tool_calls_deltas

            yield _StreamChunk(chunk_data)

            if part.get("done"):
                chunk_data = {
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop",
                        "index": 0,
                    }],
                    "model": part.get("model", ""),
                }
                yield _StreamChunk(chunk_data)

    def _ollama_non_stream(self, **kwargs):
        """Non-streaming call to Ollama, returns _Response."""
        if self._ollama_client is None:
            raise RuntimeError("Ollama client not initialised")

        response = self._ollama_client.chat(stream=False, **kwargs)
        msg = response.get("message", {})
        content = msg.get("content", "")
        tool_calls_data = msg.get("tool_calls")
        tool_calls = None
        if tool_calls_data:
            tool_calls = []
            for tc in tool_calls_data:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", {}),
                    },
                })
        return _Response(
            content=content,
            model=response.get("model", ""),
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def chat_raw(self, messages: list[dict],
                 tools: list[dict] | None = None,
                 model: str | None = None,
                 temperature: float | None = None,
                 thinking_effort: str | None = None) -> Any:
        """Non-streaming chat. Returns a _Response object.

        Args:
            messages: List of message dicts (role, content).
            tools: Optional tool definitions for function calling.
            model: Override the configured model name.
            temperature: Override the configured temperature.
            thinking_effort: Override the configured thinking effort.
        """
        return self.chat(messages, stream=False, tools=tools,
                         model=model, temperature=temperature,
                         thinking_effort=thinking_effort)

    # ------------------------------------------------------------------
    # List models
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """List available models from the provider.

        Returns:
            List of dicts with at least 'id' key, sorted alphabetically.

        Note:
            Anthropic does not expose a model listing endpoint. The returned
            list is hardcoded and may lag behind newly released models.
            Check https://docs.anthropic.com/en/docs/about-claude/models for
            the canonical list.
        """
        try:
            if self.provider == Provider.ANTHROPIC:
                # TODO: replace with a live endpoint if Anthropic ever exposes one.
                models = [
                    {"id": "claude-opus-4-20250514"},
                    {"id": "claude-sonnet-4-20250514"},
                    {"id": "claude-haiku-4-20250514"},
                    {"id": "claude-3-7-sonnet-20250219"},
                    {"id": "claude-3-5-sonnet-20241022"},
                    {"id": "claude-3-5-haiku-20241022"},
                ]

            elif self.provider == Provider.OLLAMA:
                if self._ollama_client is None:
                    raise RuntimeError("Ollama client not initialised")

                models_raw = self._ollama_client.list()
                models = []
                for m in models_raw.get("models", []):
                    name = m.model if hasattr(m, "model") else m.get("model") if isinstance(m, dict) else None
                    if name:
                        models.append({"id": name})

            else:
                if self._openai_client is None:
                    raise RuntimeError("OpenAI client not initialised")

                # OpenAI, LM Studio, Generic
                response = self._openai_client.models.list()
                models = [{"id": m.id} for m in response]

            return sorted(models, key=lambda m: m["id"])

        except Exception as e:
            raise RuntimeError(
                f"Error listing models from {self.provider.value}: {e}"
            ) from e

    def has_model(self, model_name: str) -> bool:
        """Check if a model is available."""
        models = self.list_models()
        return any(m["id"] == model_name for m in models)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        """Return a human-readable status string."""
        key_env = _PROVIDER_KEYS.get(self.provider)
        key_set = bool(os.environ.get(key_env)) if key_env else True
        thinking_enabled = self._thinking_effort and self._thinking_effort != "none"
        budget = _THINKING_BUDGET.get(self._thinking_effort, "n/a") if thinking_enabled else "off"
        return (
            f"Provider: {self.provider.value} | "
            f"Model: {self._model_name} | "
            f"URL: {self._base_url} | "
            f"Temp: {self._temperature} | "
            f"Thinking: {self._thinking_effort} (budget: {budget} tokens) | "
            f"Key: {'set' if key_set else 'missing'}"
        )
