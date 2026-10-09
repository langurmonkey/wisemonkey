"""Tests for API key resolution across providers and endpoints.

One config value serves several very different endpoints (OpenRouter, OpenCode,
OpenAI, Anthropic, ...), so the key cannot come from a single global variable.
Resolution is an ordered, per-provider chain; these tests pin both the order
and the isolation property that makes several routers usable at once.
"""

import os
import unittest
from unittest.mock import patch

from agent.router import (
    Provider,
    WM_FALLBACK_KEY_ENV,
    _get_api_key,
    api_key_candidates,
    suggest_key_env,
)


ALL_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "WM_API_KEY_OPENAI",
    "WM_API_KEY_ANTHROPIC",
    "WM_API_KEY_GENERIC",
    WM_FALLBACK_KEY_ENV,
    "OPENROUTER_API_KEY",
    "OPENCODE_API_KEY",
)


class KeyEnvTestCase(unittest.TestCase):
    """Starts every test from a clean environment."""

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in ALL_KEY_VARS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def set_env(self, **values):
        for k, v in values.items():
            os.environ[k] = v


class TestCandidateOrder(KeyEnvTestCase):
    def test_generic_chain_is_namespaced_then_conventional(self):
        self.assertEqual(
            api_key_candidates(Provider.GENERIC),
            ["WM_API_KEY_GENERIC", "OPENAI_API_KEY", WM_FALLBACK_KEY_ENV],
        )

    def test_anthropic_chain_is_anthropic_only(self):
        self.assertEqual(
            api_key_candidates(Provider.ANTHROPIC),
            ["WM_API_KEY_ANTHROPIC", "ANTHROPIC_API_KEY", WM_FALLBACK_KEY_ENV],
        )

    def test_explicit_env_comes_first(self):
        self.assertEqual(
            api_key_candidates(Provider.GENERIC, "OPENROUTER_API_KEY")[0],
            "OPENROUTER_API_KEY",
        )

    def test_explicit_name_is_not_repeated_later_in_the_chain(self):
        chain = api_key_candidates(Provider.GENERIC, "OPENAI_API_KEY")
        self.assertEqual(chain, ["OPENAI_API_KEY", "WM_API_KEY_GENERIC",
                                 WM_FALLBACK_KEY_ENV])

    def test_local_providers_need_no_key(self):
        self.assertEqual(api_key_candidates(Provider.OLLAMA), [])
        self.assertEqual(api_key_candidates(Provider.LMSTUDIO), [])


class TestKeySuggestions(unittest.TestCase):
    """``suggest_key_env`` is a hint for the /model prompt, never resolution."""

    def test_known_router_hosts_get_their_conventional_name(self):
        self.assertEqual(
            suggest_key_env("https://opencode.ai/zen/go/v1"),
            "OPENCODE_API_KEY",
        )
        self.assertEqual(
            suggest_key_env("https://openrouter.ai/api/v1"),
            "OPENROUTER_API_KEY",
        )

    def test_matching_ignores_case_and_path(self):
        self.assertEqual(
            suggest_key_env("HTTPS://API.OPENAI.COM/v1/"),
            "OPENAI_API_KEY",
        )

    def test_unknown_host_has_no_suggestion(self):
        self.assertEqual(suggest_key_env("https://llm.internal/v1"), "")
        self.assertEqual(suggest_key_env(""), "")

    def test_suggestion_never_changes_resolution(self):
        # The key for an OpenAI-compatible host is arbitrary; a hint that
        # leaked into the chain would resolve a variable the user never set up.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=False):
            for name in ("OPENCODE_API_KEY", "OPENROUTER_API_KEY"):
                os.environ.pop(name, None)
            self.assertNotIn(
                "OPENCODE_API_KEY",
                api_key_candidates(Provider.GENERIC),
            )


class TestKeyResolution(KeyEnvTestCase):
    def test_first_non_empty_wins(self):
        self.set_env(WM_API_KEY_GENERIC="namespaced", OPENAI_API_KEY="conventional")
        self.assertEqual(_get_api_key(Provider.GENERIC), "namespaced")

    def test_falls_through_empty_values(self):
        self.set_env(WM_API_KEY_GENERIC="   ", OPENAI_API_KEY="conventional")
        self.assertEqual(_get_api_key(Provider.GENERIC), "conventional")

    def test_generic_fallback_is_last_resort(self):
        self.set_env(**{WM_FALLBACK_KEY_ENV: "catchall"})
        self.assertEqual(_get_api_key(Provider.GENERIC), "catchall")

    def test_missing_key_is_empty_string(self):
        self.assertEqual(_get_api_key(Provider.GENERIC), "")

    def test_local_provider_resolves_to_nothing(self):
        self.set_env(OPENAI_API_KEY="irrelevant")
        self.assertEqual(_get_api_key(Provider.OLLAMA), "")


class TestKeyIsolation(KeyEnvTestCase):
    """The property that makes several routers usable side by side."""

    def test_switching_provider_does_not_reuse_another_providers_key(self):
        self.set_env(WM_API_KEY_GENERIC="openrouter-key",
                     WM_API_KEY_ANTHROPIC="anthropic-key")
        self.assertEqual(_get_api_key(Provider.GENERIC), "openrouter-key")
        self.assertEqual(_get_api_key(Provider.ANTHROPIC), "anthropic-key")

    def test_a_generic_only_key_does_not_satisfy_anthropic_by_default(self):
        # WM_API_KEY_* is provider-scoped; only the explicit WM_API_KEY
        # fallback is shared.
        self.set_env(WM_API_KEY_GENERIC="router-key")
        self.assertEqual(_get_api_key(Provider.ANTHROPIC), "")

    def test_explicit_env_selects_a_specific_router(self):
        self.set_env(OPENROUTER_API_KEY="or-key",
                     OPENCODE_API_KEY="oc-key",
                     OPENAI_API_KEY="openai-key")
        self.assertEqual(
            _get_api_key(Provider.GENERIC, "OPENROUTER_API_KEY"), "or-key")
        self.assertEqual(
            _get_api_key(Provider.GENERIC, "OPENCODE_API_KEY"), "oc-key")

    def test_empty_explicit_env_falls_through_the_chain(self):
        # An unset variable is skipped like any other empty entry; the chain
        # still ends at a resolvable key rather than erroring.
        self.set_env(OPENROUTER_API_KEY="", OPENAI_API_KEY="openai-key")
        self.assertEqual(
            _get_api_key(Provider.GENERIC, "OPENROUTER_API_KEY"), "openai-key")


if __name__ == "__main__":
    unittest.main()
