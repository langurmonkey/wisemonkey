"""Tests for reasoning/thinking effort translation across providers."""

import unittest

from agent.router import ModelRouter, Provider, _clamp_effort


class _Config:
    def get(self, key, default=None):
        return default


class _OpenAIClient:
    """Mirrors the SDK's client.chat.completions.create chain."""

    def __init__(self):
        self.kwargs = None
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                return object()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _AnthropicClient:
    def __init__(self):
        self.kwargs = None
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                return object()

        self.messages = _Messages()


class _OllamaClient:
    def __init__(self):
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return object()


def _router(provider, effort):
    router = ModelRouter(config=_Config())
    router.provider = provider
    router._thinking_effort = effort
    router._model_name = "m"
    router._temperature = 0.5
    return router


MESSAGES = [{"role": "user", "content": "hi"}]
LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


class TestClampEffort(unittest.TestCase):
    OPENAI = ("minimal", "low", "medium", "high")
    OLLAMA = ("low", "medium", "high")

    def test_known_levels_pass_and_others_snap_to_the_nearest(self):
        assert _clamp_effort("low", self.OPENAI) == "low"
        assert _clamp_effort("medium", self.OLLAMA) == "medium"
        # Asking for more must cap at the ceiling, not silently mean "off".
        assert _clamp_effort("max", self.OPENAI) == "high"
        assert _clamp_effort("xhigh", self.OPENAI) == "high"
        assert _clamp_effort("minimal", self.OLLAMA) == "low"
        # A vendor extension is passed through untouched.
        assert _clamp_effort("ultra", self.OPENAI) == "ultra"


class TestOpenAIEffort(unittest.TestCase):
    def _call(self, effort):
        router = _router(Provider.OPENAI, effort)
        router._openai_client = _OpenAIClient()
        router._chat_openai(MESSAGES, "m", 0.5, effort, False, None)
        return router._openai_client.kwargs

    def test_effort_travels_in_extra_body_and_is_clamped(self):
        assert self._call("high")["extra_body"] == {"reasoning_effort": "high"}
        assert self._call("max")["extra_body"] == {"reasoning_effort": "high"}

    def test_none_and_empty_send_no_field(self):
        # `none` is a 400 on older models and strict proxies, so it is omitted
        # rather than sent: the one value every endpoint accepts.
        for effort in ("none", ""):
            assert "extra_body" not in self._call(effort), effort


class TestOllamaEffort(unittest.TestCase):
    def _call(self, effort):
        router = _router(Provider.OLLAMA, effort)
        router._ollama_client = _OllamaClient()
        captured: dict = {}
        router._ollama_stream = lambda **kwargs: captured.update(kwargs)
        router._chat_ollama(MESSAGES, "m", 0.5, effort, True, None)
        return captured

    def test_effort_is_a_top_level_think_parameter(self):
        # In `options` the Pydantic Options model silently dropped it, so
        # Ollama thinking never turned on.
        kwargs = self._call("medium")
        assert kwargs["think"] == "medium"
        assert "reasoning_effort" not in kwargs["options"]

    def test_none_sends_nothing_and_levels_are_clamped(self):
        assert "think" not in self._call("none")
        assert self._call("minimal")["think"] == "low"
        assert self._call("max")["think"] == "high"


class TestAnthropicEffort(unittest.TestCase):
    def _call(self, effort):
        router = _router(Provider.ANTHROPIC, effort)
        router._anthropic_client = _AnthropicClient()
        router._anthropic_non_stream = lambda kwargs: kwargs
        return router._chat_anthropic(MESSAGES, "m", 0.5, effort, False, None)

    def test_every_level_gets_a_real_budget_that_fits_max_tokens(self):
        for effort in LEVELS:
            kwargs = self._call(effort)
            assert kwargs["thinking"]["type"] == "enabled", effort
            # budget_tokens < max_tokens is required, or it is a guaranteed 400.
            assert kwargs["thinking"]["budget_tokens"] < kwargs["max_tokens"], effort
        assert self._call("high")["thinking"] == {
            "type": "enabled", "budget_tokens": 16384,
        }

    def test_thinking_forces_temperature_one_and_none_disables_it(self):
        assert self._call("medium")["temperature"] == 1.0
        kwargs = self._call("none")
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.5

    def test_reasoning_content_is_stripped_from_the_message(self):
        # The SDK rejects unknown keys, and a thinking block is signed, so the
        # text cannot be turned back into one: the key itself must go.
        router = _router(Provider.ANTHROPIC, "none")
        for value in ("my thoughts", ""):
            adapted, _ = router._adapt_messages_for_anthropic(
                [{"role": "assistant", "content": "hi", "reasoning_content": value}]
            )
            assert "reasoning_content" not in adapted[0], value


if __name__ == "__main__":
    unittest.main()
