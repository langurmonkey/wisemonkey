"""Tests for the merged /model (endpoint) flow.

URL, provider, API key variable and model are configured in one pass. The
behaviours worth pinning:

- a canonical URL determines the provider without asking;
- a local endpoint asks no key question at all;
- an explicit variable name is stored, and an unset one is an actionable error;
- ``/url`` is a true alias, and ``/config`` no longer runs the flow twice.
"""

import os
import unittest
from unittest.mock import Mock, patch

from agent.commands import registry


class _ScriptedUI:
    """Replays scripted answers and records what was asked."""

    def __init__(self, strings=None, choices=None):
        self._strings = list(strings or [])
        self._choices = list(choices or [])
        self.string_prompts = []
        self.choice_prompts = []

    def ask_string(self, message, default=""):
        self.string_prompts.append(message)
        return self._strings.pop(0) if self._strings else default

    def ask_choice(self, message=None, options=None, default=None):
        self.choice_prompts.append(message)
        return self._choices.pop(0) if self._choices else default

    def ask_bool(self, message=None, default=False):
        return False


class _Config:
    def __init__(self, **values):
        self.values = dict(values)
        self.set_calls = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.set_calls.append((key, value))
        self.values[key] = value


def _core(models=None):
    core = Mock()
    core.initialize_router.return_value = (True, "ok")
    core.get_models.return_value = models if models is not None else [
        {"id": "m1"}
    ]
    core.router.model_name = "m1"
    core.set_model.return_value = True
    return core


def _run(config, ui, core=None):
    core = core or _core()
    command = registry.lookup(["/model"])[0]
    assert command is not None
    with patch("agent.config.get_config", return_value=config), patch(
        "pubsub.pub.sendMessage"
    ):
        return command.handler(core, [], ui)


class TestProviderFromURL(unittest.TestCase):
    def test_known_url_sets_provider_without_asking(self):
        # api.anthropic.com must resolve to anthropic, not be a question:
        # asking is how a generic host ends up receiving the Anthropic key.
        config = _Config(**{"model.base_url": "", "model.provider": "generic"})
        ui = _ScriptedUI(
            strings=["https://api.anthropic.com"], choices=["", "m1"]
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}, clear=False):
            ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(config.get("model.provider"), "anthropic")
        # No *provider* question was asked: only the key variable and model.
        self.assertEqual(ui.choice_prompts, [
            " Environment variable holding the API key for anthropic:",
            "Choose a model:",
        ])

    def test_local_url_asks_no_key_variable(self):
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=["http://localhost:11434"], choices=["m1"]
        )

        ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(config.get("model.provider"), "ollama")
        self.assertEqual(ui.string_prompts, [
            " Enter the endpoint URL"
        ])  # no key question at all

    def test_unknown_url_asks_provider_defaulting_to_generic(self):
        config = _Config(**{"model.base_url": "", "model.provider": "generic"})
        # choices: provider, key variable, model
        ui = _ScriptedUI(choices=["generic", "", "m1"])

        with patch.dict(
            os.environ, {"WM_API_KEY_GENERIC": "k"}, clear=False
        ):
            ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(len(ui.choice_prompts), 3)  # provider + key + model


class TestKeyVariableStep(unittest.TestCase):
    def test_explicit_name_is_stored(self):
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=["https://openrouter.ai/api/v1"],
            choices=["generic", "OPENROUTER_API_KEY", "m1"],
        )

        with patch.dict(
            os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False
        ):
            ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(config.get("model.api_key_env"), "OPENROUTER_API_KEY")

    def test_unset_explicit_name_is_an_actionable_error(self):
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=["https://openrouter.ai/api/v1"],
            choices=["generic", "NOT_SET_ANYWHERE"],
        )

        core = _core()
        ok, msg, _c, _m = _run(config, ui, core)

        self.assertFalse(ok)
        # Bailing out before the router is built is the point: a missing key
        # surfaces here, not as an opaque failure listing models.
        core.initialize_router.assert_not_called()
        self.assertIn("NOT_SET_ANYWHERE", msg)
        self.assertIn("auto-detect", msg)

    def test_blank_defers_to_the_chain(self):
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=["https://openrouter.ai/api/v1"],
            choices=["generic", "", "m1"],
        )

        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "k"}, clear=False
        ):
            ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(config.get("model.api_key_env"), "")


class TestUrlAlias(unittest.TestCase):
    def test_url_runs_the_same_flow(self):
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=["http://localhost:11434"], choices=["m1"]
        )
        core = _core()

        command = registry.lookup(["/url"])[0]
        assert command is not None
        with patch("agent.config.get_config", return_value=config), patch(
            "pubsub.pub.sendMessage"
        ):
            ok, _msg, _c, _m = command.handler(core, [], ui)

        self.assertTrue(ok)
        self.assertEqual(config.get("model.base_url"), "http://localhost:11434")
        core.set_model.assert_called_once_with("m1")

    def test_config_does_not_run_the_endpoint_flow_twice(self):
        # /config used to run /url *and* /model; now that they are one command
        # it must run the merged flow exactly once.
        import inspect

        command = registry.lookup(["/config"])[0]
        assert command is not None
        src = inspect.getsource(command.handler)
        self.assertNotIn('"/url"', src)
        self.assertIn('"/model"', src)


class TestSuccessMessage(unittest.TestCase):
    def test_message_names_endpoint_key_and_model(self):
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=["https://openrouter.ai/api/v1"],
            choices=["generic", "OPENROUTER_API_KEY", "deepseek-v4"],
        )

        with patch.dict(
            os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False
        ):
            _ok, msg, _c, _m = _run(config, ui)

        self.assertIn("openrouter.ai", msg)
        self.assertIn("OPENROUTER_API_KEY", msg)
        self.assertIn("deepseek-v4", msg)


if __name__ == "__main__":
    unittest.main()