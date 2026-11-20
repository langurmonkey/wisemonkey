"""Tests for agent/config.py — singleton, dot-notation, load/save/reload."""

from tests.conftest import BaseTest
from agent.config import Config, get_config


class TestConfigSingleton(BaseTest):
    def test_one_instance_and_reset_reloads_the_defaults(self):
        assert Config() is Config() is get_config()
        config = Config()
        config._config = {"changed": True}
        config.reset()
        assert {"agent", "model"} <= set(config.to_dict())
        # set()/has()/to_dict() operate on a copy of the nested dicts.
        snapshot = config.to_dict()
        snapshot["agent"]["max_turns"] = 999
        assert config.get("agent.max_turns") != 999


class TestGetSet(BaseTest):
    def test_dot_notation_get_set_and_has(self):
        config = Config()
        config.reset()
        assert config.get("agent.max_turns") == 50
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "fallback") == "fallback"

        config.set_no_save("agent.max_turns", 99)
        config.set_no_save("a.b.c.d", 42)  # creates the intermediate keys
        assert config.get("agent.max_turns") == 99
        assert config.get("a.b.c.d") == 42
        assert config.has("agent.max_turns") is True
        assert config.has("a.b.c.d.e") is False
        assert config.has("agent.nonexistent") is False


class TestLoadSave(BaseTest):
    def test_load_reload_and_save_round_trip(self):
        config = Config()
        cfg = self._default_config()
        cfg["agent"]["max_turns"] = 12
        path = self._write_config(cfg)
        config.load(path)
        assert config.config_path == path
        assert config.get("agent.max_turns") == 12

        cfg["agent"]["max_turns"] = 27
        self._write_config(cfg)
        config.reload()
        assert config.get("agent.max_turns") == 27

        config.set_no_save("agent.max_turns", 88)
        config.save()
        # A fresh instance reads the saved value back.
        del Config._instance
        reloaded = Config()
        reloaded.load(path)
        assert reloaded.get("agent.max_turns") == 88

    def test_a_missing_file_falls_back_to_the_repo_config(self):
        config = Config()
        config.load(self._tmpdir / "nonexistent.yaml")
        assert config.get("agent.max_turns") is not None
