"""Tests for API key resolution across providers and endpoints."""

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
    def test_namespaced_then_conventional_then_the_shared_fallback(self):
        assert api_key_candidates(Provider.GENERIC) == [
            "WM_API_KEY_GENERIC", "OPENAI_API_KEY", WM_FALLBACK_KEY_ENV,
        ]
        assert api_key_candidates(Provider.ANTHROPIC) == [
            "WM_API_KEY_ANTHROPIC", "ANTHROPIC_API_KEY", WM_FALLBACK_KEY_ENV,
        ]
        # An explicit name leads and is not repeated later in the chain.
        assert api_key_candidates(Provider.GENERIC, "OPENAI_API_KEY") == [
            "OPENAI_API_KEY", "WM_API_KEY_GENERIC", WM_FALLBACK_KEY_ENV,
        ]
        # Local providers need no key at all.
        assert api_key_candidates(Provider.OLLAMA) == []
        assert api_key_candidates(Provider.LMSTUDIO) == []


class TestKeyResolution(KeyEnvTestCase):
    def test_first_non_empty_wins_and_empty_values_are_skipped(self):
        self.set_env(WM_API_KEY_GENERIC="namespaced", OPENAI_API_KEY="conventional")
        assert _get_api_key(Provider.GENERIC) == "namespaced"
        self.set_env(WM_API_KEY_GENERIC="   ")
        assert _get_api_key(Provider.GENERIC) == "conventional"
        self.set_env(OPENAI_API_KEY="")
        assert _get_api_key(Provider.GENERIC) == ""

    def test_the_shared_fallback_is_a_last_resort(self):
        self.set_env(**{WM_FALLBACK_KEY_ENV: "catchall", "OPENAI_API_KEY": "direct"})
        assert _get_api_key(Provider.GENERIC) == "direct"
        os.environ.pop("OPENAI_API_KEY")
        assert _get_api_key(Provider.GENERIC) == "catchall"

    def test_a_local_provider_resolves_to_nothing(self):
        self.set_env(OPENAI_API_KEY="irrelevant")
        assert _get_api_key(Provider.OLLAMA) == ""


class TestKeyIsolation(KeyEnvTestCase):
    """The property that makes several routers usable side by side."""

    def test_keys_are_provider_scoped_and_explicit_names_select_a_router(self):
        self.set_env(WM_API_KEY_GENERIC="openrouter-key",
                     WM_API_KEY_ANTHROPIC="anthropic-key",
                     OPENCODE_API_KEY="oc-key")
        assert _get_api_key(Provider.GENERIC) == "openrouter-key"
        assert _get_api_key(Provider.ANTHROPIC) == "anthropic-key"
        assert _get_api_key(Provider.GENERIC, "OPENCODE_API_KEY") == "oc-key"

        # WM_API_KEY_* is provider-scoped; only WM_API_KEY is shared.
        self.set_env(WM_API_KEY_ANTHROPIC="")
        assert _get_api_key(Provider.ANTHROPIC) == ""

        # An unset explicit variable is skipped like any other empty entry.
        self.set_env(OPENAI_API_KEY="openai-key", WM_API_KEY_GENERIC="")
        assert _get_api_key(Provider.GENERIC, "OPENROUTER_API_KEY") == "openai-key"


class TestKeySuggestions(KeyEnvTestCase):
    """``suggest_key_env`` is a hint for the /model prompt, never resolution."""

    def test_known_hosts_are_recognised_without_case_or_path(self):
        assert suggest_key_env("https://opencode.ai/zen/go/v1") == "OPENCODE_API_KEY"
        assert suggest_key_env("HTTPS://API.OPENAI.COM/v1/") == "OPENAI_API_KEY"
        assert suggest_key_env("https://llm.internal/v1") == ""
        assert suggest_key_env("") == ""

    def test_a_suggestion_never_enters_the_resolution_chain(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=False):
            chain = api_key_candidates(Provider.GENERIC)
        assert "OPENROUTER_API_KEY" not in chain and "OPENCODE_API_KEY" not in chain
