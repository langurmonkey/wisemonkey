"""Tests for the merged /model (endpoint) flow: URL, provider, key and model."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent.commands import registry


class _ScriptedUI:
    """Replays scripted answers and records what was asked."""

    def __init__(self, strings=None, choices=None, confirm=False):
        self._strings = list(strings or [])
        self._choices = list(choices or [])
        self._confirm = confirm
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

    def ask_confirm(self, message=None, default=False):
        return self._confirm


class _Config:
    def __init__(self, **values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def _core(models=None):
    core = Mock()
    core.initialize_router.return_value = (True, "ok")
    core.get_models.return_value = models if models is not None else [{"id": "m1"}]
    core.router.model_name = "m1"
    core.set_model.return_value = True
    return core


def _run(config, ui, core=None, name="/model"):
    core = core or _core()
    command = registry.lookup([name])[0]
    assert command is not None
    with patch("agent.config.get_config", return_value=config), patch(
        "pubsub.pub.sendMessage"
    ):
        return command.handler(core, [], ui)


class TestProviderFromURL(unittest.TestCase):
    def test_a_known_url_decides_the_provider_and_a_local_one_asks_no_key(self):
        """Asking is how an Anthropic key ends up offered to a generic host."""
        config = _Config(**{"model.base_url": "", "model.provider": "generic"})
        ui = _ScriptedUI(strings=["https://api.anthropic.com"], choices=["", "m1"])
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}, clear=False):
            ok, _msg, _c, _m = _run(config, ui)
        assert ok and config.get("model.provider") == "anthropic"
        # Only the key variable and the model were offered as choices.
        assert len(ui.choice_prompts) == 2

        # A local endpoint has no key chain at all, so no key question.
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(strings=["http://localhost:11434"], choices=["m1"])
        ok, _msg, _c, _m = _run(config, ui)
        assert ok and config.get("model.provider") == "ollama"
        assert ui.string_prompts == [" Enter the endpoint URL"]

    def test_an_unknown_url_asks_the_provider(self):
        config = _Config(**{"model.base_url": "", "model.provider": "generic"})
        ui = _ScriptedUI(choices=["generic", "", "m1"])
        with patch.dict(os.environ, {"WM_API_KEY_GENERIC": "k"}, clear=False):
            assert _run(config, ui)[0] is True
        assert len(ui.choice_prompts) == 3  # provider + key + model


class TestKeyVariableStep(unittest.TestCase):
    def test_an_explicit_name_is_stored_and_a_blank_defers_to_the_chain(self):
        for choices, environ, stored in (
            (["generic", "OPENROUTER_API_KEY", "m1"], "or-key", "OPENROUTER_API_KEY"),
            (["generic", "", "m1"], "k", ""),
        ):
            config = _Config(**{"model.base_url": ""})
            ui = _ScriptedUI(strings=["https://openrouter.ai/api/v1"], choices=choices)
            env = {"OPENROUTER_API_KEY": environ} if stored else {"OPENAI_API_KEY": environ}
            with patch.dict(os.environ, env, clear=False):
                ok, _msg, _c, _m = _run(config, ui)
            assert ok and config.get("model.api_key_env") == stored, choices

    def test_an_unset_name_fails_before_the_router_is_built(self):
        """A missing key must surface here, not as an opaque model-list failure."""
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(strings=["https://openrouter.ai/api/v1"],
                         choices=["generic", "NOT_SET_ANYWHERE"])
        core = _core()
        ok, msg, _c, _m = _run(config, ui, core)
        assert ok is False
        core.initialize_router.assert_not_called()
        assert "NOT_SET_ANYWHERE" in msg and "auto-detect" in msg


class TestSuccessMessageAndAlias(unittest.TestCase):
    def test_the_message_names_the_endpoint_and_url_is_an_alias(self):
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(strings=["https://openrouter.ai/api/v1"],
                         choices=["generic", "OPENROUTER_API_KEY", "deepseek-v4"])
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False):
            _ok, msg, _c, _m = _run(config, ui)
        for fragment in ("openrouter.ai", "OPENROUTER_API_KEY", "deepseek-v4"):
            assert fragment in msg, fragment

        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(strings=["http://localhost:11434"], choices=["m1"])
        core = _core()
        ok, _msg, _c, _m = _run(config, ui, core, name="/url")
        assert ok and config.get("model.base_url") == "http://localhost:11434"
        core.set_model.assert_called_once_with("m1")


class TestEndpointList(unittest.TestCase):
    """``model.endpoints`` turns the URL question into a picker."""

    ENTRIES = [
        {"name": "Local", "base_url": "http://127.0.0.1:1234/v1",
         "provider": "lmstudio"},
        {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
         "provider": "generic", "api_key_env": "OPENROUTER_API_KEY"},
    ]

    def test_an_entry_supplies_url_provider_and_key_and_the_current_one_defaults(self):
        config = _Config(**{"model.base_url": "", "model.endpoints": self.ENTRIES})
        ui = _ScriptedUI(choices=["1", "OPENROUTER_API_KEY", "m1"])
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False):
            ok, msg, _c, _m = _run(config, ui)
        assert ok
        assert ui.string_prompts == []  # the URL is never typed by hand
        assert config.get("model.base_url") == "https://openrouter.ai/api/v1"
        assert config.get("model.provider") == "generic"
        assert config.get("model.api_key_env") == "OPENROUTER_API_KEY"
        assert "OPENROUTER_API_KEY" in msg

        # The configured URL is the default, and a new one is still offered.
        config = _Config(**{"model.base_url": "http://127.0.0.1:1234/v1",
                            "model.endpoints": self.ENTRIES})
        ui = _ScriptedUI()
        ui.ask_choice = Mock(side_effect=lambda **kw: kw["default"])
        _run(config, ui)
        first = ui.ask_choice.call_args_list[0].kwargs
        assert first["default"] == "0"
        assert first["options"][-1][0] == "__new_endpoint__"

    def test_malformed_entries_fall_back_to_a_typed_url(self):
        """A typo in config.yaml must not make /model unusable."""
        config = _Config(**{
            "model.base_url": "",
            "model.endpoints": [None, 42, {}, {"name": "no url"}],
        })
        ui = _ScriptedUI(strings=["http://localhost:11434"], choices=["m1"])
        assert _run(config, ui)[0] is True
        assert config.get("model.base_url") == "http://localhost:11434"


class TestApiKeyListing(unittest.TestCase):
    """Every ``*_API_KEY`` on disk is offered, not a fixed list of presets."""

    def test_names_come_from_the_env_file_and_the_environment_and_auto_detect_leads(self):
        from agent.commands import _key_variable_options
        from agent.config import api_key_env_names

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text(
                "# a comment\n"
                "OPENROUTER_API_KEY=or\n"
                'export CEREBRAS_API_KEY="c"\n'
                "SOME_OTHER_SETTING=1\n"
            )
            with patch("agent.config.env_file_paths", return_value=[env]):
                names = api_key_env_names()
                options = _key_variable_options(
                    base_url="https://openrouter.ai/api/v1",
                    current_env="",
                    entry_env="OPENROUTER_API_KEY",
                    suggested_env="OPENROUTER_API_KEY",
                    candidates=["OPENROUTER_API_KEY", "WM_API_KEY"],
                )

        assert "OPENROUTER_API_KEY" in names and "CEREBRAS_API_KEY" in names
        assert "SOME_OTHER_SETTING" not in names
        with patch("agent.config.env_file_paths", return_value=[]), patch.dict(
            os.environ, {"MY_PROVIDER_API_KEY": "k"}, clear=False
        ):
            assert "MY_PROVIDER_API_KEY" in api_key_env_names()

        values = [value for value, _ in options]
        assert "CEREBRAS_API_KEY" in values
        assert values[0] == ""  # auto-detect stays the default
        assert values[-2:] == ["__new_key_name__", "__new_key__"]


class TestNewKeyEntry(unittest.TestCase):
    def _run_typed_key(self, url, key):
        """Drive the flow to the "enter a new key" branch; return what was stored."""
        config = _Config(**{"model.base_url": ""})
        ui = _ScriptedUI(strings=[url, key],
                         choices=["generic", "__new_key__", "m1"], confirm=True)
        stored = {}

        def fake_set(name, value, path=None):
            stored[name] = value
            os.environ[name] = value
            return "/tmp/fake/.env"

        with patch("agent.config.set_env_var", side_effect=fake_set), patch(
            "agent.config.env_file_paths", return_value=[]
        ):
            ok, _msg, _c, _m = _run(config, ui)
        for name in stored:
            os.environ.pop(name, None)
        return config, ok, stored

    def test_a_typed_key_gets_the_hosts_name_and_an_unknown_host_the_namespaced_one(self):
        """So a router key also works for tools reading OPENROUTER_API_KEY."""
        config, ok, stored = self._run_typed_key(
            "https://openrouter.ai/api/v1", "sk-brand-new"
        )
        assert ok and stored == {"OPENROUTER_API_KEY": "sk-brand-new"}
        assert config.get("model.api_key_env") == "OPENROUTER_API_KEY"

        _config, ok, stored = self._run_typed_key("https://example.test/v1", "sk-x")
        assert ok and stored == {"WM_API_KEY_GENERIC": "sk-x"}


class TestSetEnvVar(unittest.TestCase):
    def test_replaces_in_place_privately_and_quotes_when_needed(self):
        from agent.config import set_env_var

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("OTHER=1\nFOO_API_KEY=old\n")
            set_env_var("FOO_API_KEY", "new", path=env)
            text = env.read_text()
            assert "FOO_API_KEY=new" in text and "old" not in text
            assert "OTHER=1" in text and text.count("FOO_API_KEY") == 1
            assert env.stat().st_mode & 0o777 == 0o600

            set_env_var("ODD_API_KEY", "a b#c", path=env)
            assert 'ODD_API_KEY="a b#c"' in env.read_text()

        for name in ("FOO_API_KEY", "ODD_API_KEY"):
            os.environ.pop(name, None)


if __name__ == "__main__":
    unittest.main()
