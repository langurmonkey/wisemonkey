"""Tests for agent/config.py — Config singleton, get/set, load/save, defaults."""

import os
import shutil
import tempfile
import yaml

from pathlib import Path

from tests.conftest import BaseTest
from agent.config import Config, get_config, _get_defaults


class TestConfigDefaults(BaseTest):
    """Test that defaults are returned correctly."""

    def test_get_defaults_returns_dict(self):
        defaults = _get_defaults()
        assert isinstance(defaults, dict)

    def test_get_defaults_has_model_section(self):
        defaults = _get_defaults()
        assert "model" in defaults
        assert "name" in defaults["model"]

    def test_get_defaults_has_agent_section(self):
        defaults = _get_defaults()
        assert "agent" in defaults
        assert "context_files" in defaults["agent"]
        assert defaults["agent"]["context_files"] == ["AGENTS.md"]

    def test_get_defaults_has_embedding_section(self):
        defaults = _get_defaults()
        assert "embedding" in defaults


class TestConfigSingleton(BaseTest):
    """Test the Config singleton pattern."""

    def test_same_instance_returned(self):
        a = Config()
        b = Config()
        assert a is b

    def test_get_config_returns_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_clears_config(self):
        c = Config()
        c._config = {"changed": True}
        c.reset()
        assert c.to_dict() == _get_defaults()


class TestConfigGetSet(BaseTest):
    """Test dot-notation get and set."""

    def setUp(self):
        super().setUp()
        self.config = Config()
        self.config.reset()
        assert self.config.get("agent.max_turns") == 50

    def test_get_missing_key_returns_default(self):
        assert self.config.get("nonexistent.key", "fallback") == "fallback"

    def test_get_missing_key_returns_none(self):
        assert self.config.get("nonexistent.key") is None

    def test_set_updates_value(self):
        self.config.set("agent.max_turns", 99)
        assert self.config.get("agent.max_turns") == 99

    def test_set_creates_nested_keys(self):
        self.config.set("agent.new_field", "hello")
        assert self.config.get("agent.new_field") == "hello"

    def test_set_deeply_nested(self):
        self.config.set("a.b.c.d", 42)
        assert self.config.get("a.b.c.d") == 42

    def test_has_returns_true_for_existing(self):
        assert self.config.has("agent.max_turns") is True

    def test_has_returns_false_for_missing(self):
        assert self.config.has("agent.nonexistent") is False

    def test_has_returns_false_for_deeply_missing(self):
        assert self.config.has("a.b.c.d.e") is False

    def test_to_dict_returns_copy(self):
        d = self.config.to_dict()
        d["agent"]["max_turns"] = 999
        assert self.config.get("agent.max_turns") == 50


class TestConfigLoadSave(BaseTest):
    """Test loading and saving config from/to YAML files."""

    def setUp(self):
        super().setUp()
        self.config = Config()

    def test_load_from_file(self):
        cfg = self._default_config()
        cfg["agent"]["max_turns"] = 77
        path = self._write_config(cfg)
        self.config.load(path)
        assert self.config.get("agent.max_turns") == 77

    def test_load_missing_file_uses_defaults(self):
        fake_path = self._tmpdir / "nonexistent.yaml"
        self.config.load(fake_path)
        assert self.config.to_dict() == _get_defaults()

    def test_save_creates_file(self):
        path = self._tmpdir / "test_config.yaml"
        self.config._config_path = path
        self.config.save()
        assert path.exists()

    def test_save_round_trip(self):
        cfg = self._default_config()
        cfg["agent"]["max_turns"] = 42
        path = self._write_config(cfg)
        self.config.load(path)
        assert self.config.get("agent.max_turns") == 42

        self.config.set("agent.max_turns", 88)
        self.config.save()

        # Reload from disk
        del Config._instance
        c2 = Config()
        c2.load(path)
        assert c2.get("agent.max_turns") == 88
