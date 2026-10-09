"""Tests for reasoning/thinking effort translation across providers.

One config vocabulary (``model.thinking.effort``) meets three providers with
different dialects: OpenAI takes a ``reasoning_effort`` in extra_body, Ollama
takes a top-level ``think``, Anthropic takes an absolute ``budget_tokens`` with
``max_tokens > budget``. These tests pin the wire kwargs for each.
"""

import unittest

from agent.router import (
    Provider,
    _clamp_effort,
    ModelRouter,
)


class _Config:
    """Minimal config stub for ModelRouter."""

    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _Capture:
    """Records the kwargs it is called with and returns a stub response."""

    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return object()


class _OpenAIClient:
    """Mirrors the SDK's client.chat.completions.create chain."""

    kwargs: dict | None = None

    def __init__(self):
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                return object()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _Messages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return object()

    def stream(self, **kwargs):
        raise NotImplementedError


class _AnthropicClient:
    def __init__(self):
        self.messages = _Messages()


class _OllamaClient:
    def __init__(self):
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return object()


def _router(provider, effort):
    r = ModelRouter(config=_Config())
    r.provider = provider
    r._thinking_effort = effort
    r._model_name = "m"
    r._temperature = 0.5
    return r


MESSAGES = [{"role": "user", "content": "hi"}]


class TestClampEffort(unittest.TestCase):
    OPENAI = ("minimal", "low", "medium", "high")
    OLLAMA = ("low", "medium", "high")

    def test_known_level_inside_range_passes_through(self):
        self.assertEqual(_clamp_effort("low", self.OPENAI), "low")
        self.assertEqual(_clamp_effort("medium", self.OLLAMA), "medium")

    def test_level_above_ceiling_is_capped_not_dropped(self):
        # max -> high, not "off": asking for more must not disable reasoning.
        self.assertEqual(_clamp_effort("max", self.OPENAI), "high")
        self.assertEqual(_clamp_effort("xhigh", self.OPENAI), "high")

    def test_level_below_floor_raises_to_floor(self):
        # Ollama has no `minimal`; snapping up keeps thinking on.
        self.assertEqual(_clamp_effort("minimal", self.OLLAMA), "low")

    def test_unknown_vendor_value_is_untouched(self):
        self.assertEqual(_clamp_effort("ultra", self.OPENAI), "ultra")


class TestOpenAIEffort(unittest.TestCase):
    def _call(self, effort):
        r = _router(Provider.OPENAI, effort)
        r._openai_client = _OpenAIClient()
        r._chat_openai(MESSAGES, "m", 0.5, effort, False, None)
        return r._openai_client.kwargs

    def test_effort_travels_in_extra_body(self):
        self.assertEqual(self._call("high")["extra_body"],
                         {"reasoning_effort": "high"})

    def test_none_sends_no_field(self):
        # `none` is valid on recent OpenAI models but a 400 on older ones and
        # on strict proxies; omitting it is accepted everywhere.
        self.assertNotIn("extra_body", self._call("none"))

    def test_empty_effort_sends_no_field(self):
        self.assertNotIn("extra_body", self._call(""))

    def test_config_only_level_is_clamped(self):
        self.assertEqual(self._call("max")["extra_body"],
                         {"reasoning_effort": "high"})


class TestOllamaEffort(unittest.TestCase):
    def _call(self, effort):
        r = _router(Provider.OLLAMA, effort)
        r._ollama_client = _OllamaClient()
        captured: dict = {}
        r._ollama_stream = lambda **kw: captured.update(kw)
        r._chat_ollama(MESSAGES, "m", 0.5, effort, True, None)
        return captured

    def test_effort_travels_as_top_level_think(self):
        # It used to be stuffed into `options`, where the Pydantic Options
        # model silently dropped it, so Ollama thinking never turned on.
        kwargs = self._call("medium")
        self.assertEqual(kwargs["think"], "medium")
        self.assertNotIn("reasoning_effort", kwargs["options"])

    def test_none_sends_no_think(self):
        self.assertNotIn("think", self._call("none"))

    def test_clamped_to_ollama_levels(self):
        self.assertEqual(self._call("minimal")["think"], "low")
        self.assertEqual(self._call("max")["think"], "high")


class TestAnthropicEffort(unittest.TestCase):
    def _call(self, effort):
        r = _router(Provider.ANTHROPIC, effort)
        r._anthropic_client = _AnthropicClient()
        r._anthropic_non_stream = lambda kw: kw
        return r._chat_anthropic(MESSAGES, "m", 0.5, effort, False, None)

    def test_budget_always_fits_in_max_tokens(self):
        # budget_tokens < max_tokens is required by the API; a fixed 8192 made
        # every budget above it a guaranteed 400.
        for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
            kw = self._call(effort)
            with self.subTest(effort=effort):
                self.assertLess(kw["thinking"]["budget_tokens"],
                                kw["max_tokens"])

    def test_high_gets_its_documented_budget(self):
        kw = self._call("high")
        self.assertEqual(kw["thinking"],
                         {"type": "enabled", "budget_tokens": 16384})

    def test_every_config_level_has_a_real_budget(self):
        # Previously minimal/xhigh/max fell through to a silent 4096.
        for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
            kw = self._call(effort)
            with self.subTest(effort=effort):
                self.assertEqual(kw["thinking"]["type"], "enabled")

    def test_none_disables_thinking_and_keeps_temperature(self):
        kw = self._call("none")
        self.assertNotIn("thinking", kw)
        self.assertEqual(kw["temperature"], 0.5)

    def test_thinking_forces_temperature_one(self):
        self.assertEqual(self._call("medium")["temperature"], 1.0)


class TestReasoningContentStrippedForAnthropic(unittest.TestCase):
    def test_reasoning_content_is_removed(self):
        # Anthropic's SDK rejects unknown message keys, and its thinking blocks
        # are signed, so text cannot be reconstructed into one.
        r = _router(Provider.ANTHROPIC, "none")
        msgs = [{"role": "assistant", "content": "hi",
                 "reasoning_content": "my thoughts"}]
        adapted, _ = r._adapt_messages_for_anthropic(msgs)
        self.assertNotIn("reasoning_content", adapted[0])


if __name__ == "__main__":
    unittest.main()