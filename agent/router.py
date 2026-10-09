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


def _parse_args(args: str) -> dict:
    """Parse a tool-call ``arguments`` string, tolerating malformed JSON.

    A truncated or invalid call (e.g. a bare ``{``) would otherwise raise
    here and abort the whole request. Fall back to the raw string wrapped
    in a dict so the model still sees what it emitted.
    """
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {"__raw__": args}



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

# API key resolution, per provider.
#
# One config value must serve several very different endpoints: a user who
# mostly drives Wisemonkey through routers (OpenRouter, OpenCode, ...) flips
# between them, and all of them speak the OpenAI dialect. A single global
# WM_API_KEY would have to be rewritten on every switch -- ending up in shell
# history -- and, worse, would send one endpoint's key to another the moment
# the provider changed. So resolution is an ordered chain, per provider, and
# the first non-empty entry wins:
#
#   1. ``model.api_key_env``  -- the env var *name* from config.yaml. Explicit
#      wins, and naming a var rather than storing a secret keeps the key out of
#      a config file that may be shared or committed.
#   2. ``WM_API_KEY_<PROVIDER>`` -- namespaced, so several endpoint keys can
#      coexist. This is what actually solves the router case.
#   3. The provider's conventional var (``OPENAI_API_KEY`` / ...), so the
#      ``openai``/``anthropic`` SDKs' own auto-discovery keeps working and
#      existing setups do not break.
#
# Local providers need nothing.
_PROVIDER_KEY_CHAINS: dict[Provider, tuple[str, ...]] = {
    Provider.OPENAI: ("WM_API_KEY_OPENAI", "OPENAI_API_KEY"),
    Provider.ANTHROPIC: ("WM_API_KEY_ANTHROPIC", "ANTHROPIC_API_KEY"),
    Provider.OLLAMA: (),
    Provider.LMSTUDIO: (),
    Provider.GENERIC: ("WM_API_KEY_GENERIC", "OPENAI_API_KEY"),
}

# Last-resort convenience var, tried last for every key-bearing provider. Never
# the only source: it must not be able to leak one provider's key to another.
WM_FALLBACK_KEY_ENV = "WM_API_KEY"

# Ordered effort vocabulary. ``config.yaml`` offers one ladder and each
# provider supports a different subset of it, so the configured value is
# clamped rather than dropped: asking for more than a provider has must not
# silently turn reasoning off.
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Anthropic thinking effort -> budget_tokens. Covers the whole config
# vocabulary so an effort like ``max`` no longer falls back to 4096.
_THINKING_BUDGET = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 24576,
    "max": 32768,
}

# Anthropic requires budget_tokens < max_tokens, so max_tokens is derived from
# the budget rather than fixed: with a hard-coded 8192, ``high`` (16384) was a
# guaranteed 400 on every request.
_ANTHROPIC_MAX_TOKENS = 8192
_ANTHROPIC_RESPONSE_MARGIN = 4096


def _clamp_effort(effort: str, allowed: tuple[str, ...]) -> str:
    """Clamp a configured effort into the range ``allowed`` supports.

    Providers accept different subsets of the config vocabulary: OpenAI takes
    minimal..high, Ollama only low|medium|high. Asking for a level a provider
    does not know must not raise and must not silently mean "off", so a known
    level is snapped to the nearest one the provider *does* accept (down or
    up). An unknown value (a vendor extension, say ``ultra``) is returned
    unchanged so an endpoint that does understand it still receives it.
    """
    if effort not in _EFFORT_ORDER:
        return effort
    if effort in allowed:
        return effort
    idx = _EFFORT_ORDER.index(effort)
    lo, hi = _EFFORT_ORDER.index(allowed[0]), _EFFORT_ORDER.index(allowed[-1])
    return _EFFORT_ORDER[min(max(idx, lo), hi)]


def _match_provider_url(base_url: str) -> Provider | None:
    """Return the provider for a *known* base URL, or None if unrecognised.

    ``_detect_provider`` collapses everything unknown to ``GENERIC``, which is
    the right answer for routing but hides the distinction the endpoint
    configurator needs: a canonical host decides the provider outright, while an
    unknown host (OpenRouter, OpenCode, vLLM) has to ask.
    """
    normalized = (base_url or "").rstrip("/")
    for url, provider in _PROVIDER_URLS:
        if normalized and normalized == url.rstrip("/"):
            return provider
    return None


def _detect_provider(base_url: str) -> Provider:
    """Auto-detect provider from base URL."""
    return _match_provider_url(base_url) or Provider.GENERIC


# Host fragment -> the env var name a user of that endpoint most likely keeps
# its key in. This is a *hint*, never part of the resolution order: the key for
# an OpenAI-compatible host is genuinely arbitrary, so the point is only to put
# the right name in front of the user instead of making them recall it. Matched
# as a substring of the base URL, longest-first, so "api.openrouter.ai" wins
# over a bare "openrouter".
_HOST_KEY_HINTS: tuple[tuple[str, str], ...] = (
    ("opencode.ai", "OPENCODE_API_KEY"),
    ("openrouter.ai", "OPENROUTER_API_KEY"),
    ("localhost:1234", "LMSTUDIO_API_KEY"),
    ("localhost:3001", "FREELLMAPI_API_KEY"),
    ("api.openai.com", "OPENAI_API_KEY"),
    ("api.anthropic.com", "ANTHROPIC_API_KEY"),
)


def suggest_key_env(base_url: str) -> str:
    """Return the env var name most likely to hold this endpoint's key.

    Empty string when nothing matches. Advisory only: it never affects
    :func:`_get_api_key`, which resolves strictly through
    :func:`api_key_candidates`.
    """
    url = (base_url or "").lower()
    for fragment, env_name in sorted(
        _HOST_KEY_HINTS, key=lambda kv: -len(kv[0])
    ):
        if fragment in url:
            return env_name
    return ""


def api_key_candidates(provider: Provider, explicit_env: str = "") -> list[str]:
    """Return the env var names tried for ``provider``, in order.

    Order is "most explicit wins": the configured ``model.api_key_env``, then
    the namespaced ``WM_API_KEY_<PROVIDER>``, then the provider's conventional
    var, then the generic ``WM_API_KEY`` fallback. Exposed so the "missing key"
    error can name every name it looked for instead of guessing.
    """
    if provider not in (Provider.OPENAI, Provider.ANTHROPIC, Provider.GENERIC):
        return []
    explicit = (explicit_env or "").strip()
    chain: list[str] = []
    if explicit:
        chain.append(explicit)
    chain += [n for n in _PROVIDER_KEY_CHAINS[provider] if n != explicit]
    chain.append(WM_FALLBACK_KEY_ENV)
    # de-dupe, preserving order (an explicit name may repeat a chain entry)
    return list(dict.fromkeys(chain))


def _get_api_key(provider: Provider, explicit_env: str = "") -> str:
    """Resolve the API key for ``provider`` from the environment chain.

    The first non-empty variable in :func:`api_key_candidates` wins; the value
    itself is never stored in config.
    """
    for name in api_key_candidates(provider, explicit_env):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


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
        # Token usage, when the provider reports it. Only the final chunk of
        # a stream carries it (Anthropic's message_delta, Ollama's done
        # part); OpenAI delivers it as a choices-less chunk when
        # `stream_options.include_usage` is set. None means the provider
        # did not report usage and the caller falls back to estimating.
        self.usage = data.get("usage")


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

    def __init__(self, config=None, session=None):
        self.config = config or get_config()
        self.session = session
        self.provider: Provider = Provider.GENERIC

        # SDK clients \u2014 only the one matching self.provider will be created
        self._openai_client: Optional[Any] = None
        self._anthropic_client: Optional[Any] = None
        self._ollama_client: Optional[Any] = None

        # Config-derived settings
        self._model_name: str = ""
        self._base_url: str = ""
        self._temperature: float = 0.8
        self._thinking_effort: str = "low"
        # Env var *name* (never the key itself) naming this endpoint's key
        self._key_env: str = ""
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
        self._temperature = self.config.get("model.temperature", 0.6)
        effort = str(self.config.get("model.thinking.effort", "low") or "").strip().lower()
        self._thinking_effort = effort or "none"

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
        explicit_env = self.config.get("model.api_key_env", "") or ""
        self._key_env = str(explicit_env).strip()
        api_key = _get_api_key(self.provider, self._key_env)
        if not api_key and self.provider in (Provider.OPENAI, Provider.ANTHROPIC):
            tried = ", ".join(api_key_candidates(self.provider, explicit_env))
            return False, (
                f"Missing API key for {self.provider.value}. Tried: {tried}. "
                f"Set one in your environment or .env file, or point "
                f"model.api_key_env at a variable holding the key, or run "
                f"/onboard."
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
            "HTTP-Referer": "https://tonisagrista.com/projects/wisemonkey/",
            "X-Title": "Wisemonkey Agent",
        }
        # Add the OpenCode session header only for OpenCode endpoints.
        if "opencode.ai" in self._base_url.lower():
            session_id = self.session

            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError(
                    "OpenCode API requires a non-empty session ID. "
                    "Pass a stable session ID to ModelRouter."
                )

            kwargs["extra_headers"]["x-opencode-session"] = session_id.strip()
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Ask for token usage on streamed responses. Without this the
        # OpenAI-compatible API reports no usage at all and the agent can
        # only estimate from the response text. Servers that predate the
        # option ignore it, so sending it is safe.
        if stream:
            kwargs["stream_options"] = {"include_usage": True}

# Extra body params (reasoning_effort, etc.). Only sent when thinking
        # is actually on: `none` is valid on recent OpenAI models but is a 400
        # on older ones and on strict OpenAI-compatible proxies, and an empty
        # effort would be meaningless. Clamped to the levels the OpenAI family
        # accepts so the config-only `xhigh`/`max` do not 400 either.
        if thinking and thinking != "none":
            effort = _clamp_effort(thinking, ("minimal", "low", "medium", "high"))
            kwargs["extra_body"] = {"reasoning_effort": effort}

        return self._openai_client.chat.completions.create(**kwargs)



    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _adapt_messages_for_anthropic(self, messages: list[dict]
                                      ) -> tuple[list[dict], str | None]:
        """Extract system messages and convert image blocks to Anthropic format.

        Anthropic uses a separate 'system' param and a different image block
        format than OpenAI.  This method:
        1. Extracts system messages into a separate string.
        2. Converts OpenAI-style image_url content blocks to Anthropic's
           image source format, in tool results and in user messages (the
           pending-image path attaches images to user prompts too).
        """
        system_parts = []
        chat_messages = []
        for msg in messages:
            # DeepSeek's thinking mode requires `reasoning_content` back on the
            # assistant message, so core.py puts it there. Anthropic names the
            # same thing differently (thinking blocks) and its SDK rejects
            # unknown message keys, so it is dropped on this path.
            if msg.get("reasoning_content"):
                msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)
            else:
                # Convert image blocks from OpenAI to Anthropic format. Tool
                # results carry screenshots; user messages carry images
                # attached by the pending-image path or /attach-image.
                if msg.get("role") in ("tool", "user") and isinstance(msg.get("content"), list):
                    converted_content = []
                    text_parts = []
                    for block in msg["content"]:
                        if block.get("type") == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            # Extract base64 data from data URL
                            if url.startswith("data:"):
                                header, data = url.split(",", 1)
                                mime = header.split(";")[0].split(":")[1]
                                converted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime,
                                        "data": data,
                                    },
                                })
                            else:
                                converted_content.append(block)
                        elif block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        else:
                            converted_content.append(block)
                    # Merge text parts into a single text block before the image
                    if text_parts:
                        converted_content.insert(0, {
                            "type": "text",
                            "text": "\n".join(text_parts),
                        })
                    msg = {**msg, "content": converted_content}
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
        thinking_enabled = bool(thinking) and thinking != "none"
        effective_temp = 1.0 if thinking_enabled else temp
        # Anthropic has no effort ladder: it takes an absolute token budget and
        # requires budget_tokens < max_tokens. max_tokens is therefore derived
        # from the budget -- hard-coding 8192 made every budget above it (all
        # of `high`/`xhigh`/`max`) a guaranteed 400.
        budget = _THINKING_BUDGET.get(thinking, 8192) if thinking_enabled else 0
        max_tokens = _ANTHROPIC_MAX_TOKENS
        if thinking_enabled:
            max_tokens = max(max_tokens, budget + _ANTHROPIC_RESPONSE_MARGIN)

        kwargs: dict = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": chat_msgs,
            "temperature": effective_temp,
        }

        if system:
            kwargs["system"] = system

        if thinking_enabled:
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

        # Final chunk with finish_reason. The message_delta event carries the
        # cumulative output tokens and the message_start event the input
        # count; both are surfaced here so the footer can show real usage
        # instead of an estimate.
        final_msg = stream.get_final_message()
        usage = getattr(final_msg, "usage", None)
        usage_data = None
        if usage is not None:
            usage_data = {
                "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            }
        chunk_data = {
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
                "index": 0,
            }],
            "model": final_msg.model or kwargs.get("model", ""),
            "usage": usage_data,
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
                            # A malformed call from a previous turn would
                            # otherwise raise here and abort the whole request.
                            "arguments": _parse_args(args) if isinstance(args, str) else args,
                        },
                    })
                adapted.append({**msg, "tool_calls": fixed_tcs})
                continue

            # Ollama has no image_url content blocks: images go in a separate
            # `images` list of raw base64 strings and content must be plain
            # text. Extract the base64 data out of OpenAI-style blocks
            # (tool results from screenshot/read_image, user messages with a
            # pending image) so the vision decoder actually receives them.
            content = msg.get("content")
            if msg.get("role") in ("tool", "user") and isinstance(content, list):
                images = list(msg.get("images", []) or [])
                text_parts = []
                for block in content:
                    if block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            _, data = url.split(",", 1)
                            images.append(data)
                    elif block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                msg = {**msg, "content": "\n".join(text_parts)}
                if images:
                    msg["images"] = images
            adapted.append(msg)

        options = {"temperature": temp}

        kwargs = {
            "model": model_name,
            "messages": adapted,   # <-- use adapted, not messages
            "options": options,
        }
        # Ollama's thinking switch is a *top-level* `think` parameter
        # (bool | "low" | "medium" | "high"), not an option. Stuffing it into
        # `options` looked right but was silently dropped: `options` is
        # serialised through the Pydantic `Options` model, which has no such
        # field, so Ollama thinking never actually turned on. Clamped to the
        # three levels Ollama accepts.
        if thinking and thinking != "none":
            kwargs["think"] = _clamp_effort(thinking, ("low", "medium", "high"))
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
                    # Ollama reports usage on the final part. Absent on
                    # some backends, in which case the caller estimates.
                    "usage": {
                        "prompt_tokens": part.get("prompt_eval_count", 0) or 0,
                        "completion_tokens": part.get("eval_count", 0) or 0,
                    },
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
        # Report the variable that will actually supply the key, so the status
        # line names the same source the resolver uses.
        chain = api_key_candidates(self.provider, self._key_env)
        key_env = chain[0] if chain else "not needed"
        key_set = bool(_get_api_key(self.provider, self._key_env))
        effort = self._thinking_effort
        if not effort or effort == "none":
            thinking_desc = "off"
        elif self.provider == Provider.ANTHROPIC:
            # Anthropic has no ladder: the config effort becomes a token
            # budget, and max_tokens is sized off it.
            thinking_desc = (f"{effort} "
                             f"(budget: {_THINKING_BUDGET.get(effort, 8192)} tokens)")
        elif self.provider == Provider.OLLAMA:
            thinking_desc = _clamp_effort(effort, ("low", "medium", "high"))
        else:
            thinking_desc = _clamp_effort(
                effort, ("minimal", "low", "medium", "high"))
        return (
            f"Provider: {self.provider.value} | "
            f"Model: {self._model_name} | "
            f"URL: {self._base_url} | "
            f"Temp: {self._temperature} | "
            f"Thinking: {thinking_desc} | "
            f"Key: {'set' if key_set else 'missing'}"
        )
