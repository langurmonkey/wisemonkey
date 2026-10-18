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


class TestEndpointList(unittest.TestCase):
    """``model.endpoints`` turns the URL question into a picker.

    The point of the list is that a host one uses regularly is one selection
    away, and that a selection can carry the provider and the key variable with
    it -- those are properties of the *endpoint*, not of the model.
    """

    ENTRIES = [
        {
            "name": "Local",
            "base_url": "http://127.0.0.1:1234/v1",
            "provider": "lmstudio",
        },
        {
            "name": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "generic",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    ]

    def test_entry_supplies_url_provider_and_key_variable(self):
        config = _Config(
            **{"model.base_url": "", "model.endpoints": self.ENTRIES}
        )
        # choices: the entry, the key variable, the model -- no URL string.
        ui = _ScriptedUI(choices=["1", "OPENROUTER_API_KEY", "m1"])

        with patch.dict(
            os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False
        ):
            ok, msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(ui.string_prompts, [])  # URL never typed by hand
        self.assertEqual(
            config.get("model.base_url"), "https://openrouter.ai/api/v1"
        )
        self.assertEqual(config.get("model.provider"), "generic")
        self.assertEqual(config.get("model.api_key_env"), "OPENROUTER_API_KEY")
        self.assertIn("OPENROUTER_API_KEY", msg)

    def test_new_url_is_still_one_option_of_the_list(self):
        config = _Config(
            **{"model.base_url": "", "model.endpoints": self.ENTRIES}
        )
        ui = _ScriptedUI(
            strings=["https://example.test/v1"],
            choices=["__new_endpoint__", "generic", "", "m1"],
        )

        with patch.dict(os.environ, {"WM_API_KEY_GENERIC": "k"}, clear=False):
            ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(ui.string_prompts, [" Enter the endpoint URL"])
        self.assertEqual(config.get("model.base_url"), "https://example.test/v1")

    def test_current_url_is_the_default_selection(self):
        config = _Config(
            **{
                "model.base_url": "http://127.0.0.1:1234/v1",
                "model.endpoints": self.ENTRIES,
            }
        )
        ui = _ScriptedUI()  # every answer takes its default
        ui.ask_choice = Mock(side_effect=lambda **kw: kw["default"])

        _run(config, ui)

        picked = ui.ask_choice.call_args_list[0].kwargs
        self.assertEqual(picked["default"], "0")
        self.assertEqual(picked["options"][-1][0], "__new_endpoint__")

    def test_malformed_entries_do_not_break_the_command(self):
        # A typo in config.yaml must not make /model unusable.
        config = _Config(
            **{
                "model.base_url": "",
                "model.endpoints": [None, 42, {}, {"name": "no url"}],
            }
        )
        ui = _ScriptedUI(
            strings=["http://localhost:11434"], choices=["m1"]
        )

        ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(config.get("model.base_url"), "http://localhost:11434")


class TestApiKeyListing(unittest.TestCase):
    """Every ``*_API_KEY`` on disk is offered, not a fixed list of presets."""

    def test_env_file_names_are_listed(self):
        import tempfile
        from pathlib import Path

        from agent.config import api_key_env_names

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text(
                "# a comment\n"
                "OPENROUTER_API_KEY=or\n"
                "export CEREBRAS_API_KEY=\"c\"\n"
                "SOME_OTHER_SETTING=1\n"
            )
            with patch("agent.config.env_file_paths", return_value=[env]):
                names = api_key_env_names()

        self.assertIn("OPENROUTER_API_KEY", names)
        self.assertIn("CEREBRAS_API_KEY", names)
        self.assertNotIn("SOME_OTHER_SETTING", names)

    def test_environment_names_are_listed_too(self):
        from agent.config import api_key_env_names

        with patch("agent.config.env_file_paths", return_value=[]), patch.dict(
            os.environ, {"MY_PROVIDER_API_KEY": "k"}, clear=False
        ):
            self.assertIn("MY_PROVIDER_API_KEY", api_key_env_names())

    def test_picker_offers_every_key_found(self):
        from agent.commands import _key_variable_options

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("CEREBRAS_API_KEY=c\n")
            with patch("agent.config.env_file_paths", return_value=[env]):
                options = _key_variable_options(
                    base_url="https://openrouter.ai/api/v1",
                    current_env="",
                    entry_env="OPENROUTER_API_KEY",
                    suggested_env="OPENROUTER_API_KEY",
                    candidates=["OPENROUTER_API_KEY", "WM_API_KEY"],
                )

        values = [v for v, _ in options]
        self.assertIn("CEREBRAS_API_KEY", values)
        self.assertEqual(values[0], "")  # auto-detect stays first
        self.assertEqual(values[-2:], ["__new_key_name__", "__new_key__"])

    def test_key_found_in_env_file_can_be_selected(self):
        import tempfile
        from pathlib import Path

        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=["https://openrouter.ai/api/v1"],
            choices=["generic", "CEREBRAS_API_KEY", "m1"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("CEREBRAS_API_KEY=c\n")
            with patch("agent.config.env_file_paths", return_value=[env]), patch.dict(
                os.environ, {"CEREBRAS_API_KEY": "c"}, clear=False
            ):
                ok, _msg, _c, _m = _run(config, ui)

        self.assertTrue(ok)
        self.assertEqual(config.get("model.api_key_env"), "CEREBRAS_API_KEY")


class TestNewKeyEntry(unittest.TestCase):
    def _run_typed_key(self, url, key):
        """Drive the flow to the "enter a new key" branch; return what was stored."""
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(
            strings=[url, key],
            choices=["generic", "__new_key__", "m1"],
        )
        ui.ask_confirm = lambda message, default=False: True

        stored = {}

        def fake_set(name, value, path=None):
            stored[name] = value
            os.environ[name] = value
            return "/tmp/fake/.env"

        with patch("agent.config.set_env_var", side_effect=fake_set), patch(
            "agent.config.env_file_paths", return_value=[]
        ):
            ok, msg, _c, _m = _run(config, ui)

        for name in stored:
            os.environ.pop(name, None)
        return config, ok, msg, stored

    def test_typing_a_key_stores_it_and_uses_it(self):
        config, ok, msg, stored = self._run_typed_key(
            "https://openrouter.ai/api/v1", "sk-brand-new"
        )

        self.assertTrue(ok)
        # A known host gets its conventional variable, not the generic fallback:
        # the key then also works for tools that read OPENROUTER_API_KEY.
        self.assertEqual(stored, {"OPENROUTER_API_KEY": "sk-brand-new"})
        self.assertEqual(config.get("model.api_key_env"), "OPENROUTER_API_KEY")
        self.assertIn("/tmp/fake/.env", msg)

    def test_unknown_host_falls_back_to_the_namespaced_variable(self):
        _config, ok, _msg, stored = self._run_typed_key(
            "https://example.test/v1", "sk-x"
        )

        self.assertTrue(ok)
        self.assertEqual(stored, {"WM_API_KEY_GENERIC": "sk-x"})


class TestSetEnvVar(unittest.TestCase):
    def test_replaces_in_place_and_is_private(self):
        import tempfile
        from pathlib import Path

        from agent.config import set_env_var

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("OTHER=1\nFOO_API_KEY=old\n")
            set_env_var("FOO_API_KEY", "new", path=env)

            text = env.read_text()
            self.assertIn("FOO_API_KEY=new", text)
            self.assertNotIn("old", text)
            self.assertIn("OTHER=1", text)
            self.assertEqual(text.count("FOO_API_KEY"), 1)
            self.assertEqual(env.stat().st_mode & 0o777, 0o600)

        os.environ.pop("FOO_API_KEY", None)

    def test_values_needing_quotes_are_quoted(self):
        import tempfile
        from pathlib import Path

        from agent.config import set_env_var

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            set_env_var("ODD_API_KEY", "a b#c", path=env)
            self.assertIn('ODD_API_KEY="a b#c"', env.read_text())

        os.environ.pop("ODD_API_KEY", None)


if __name__ == "__main__":
    unittest.main()