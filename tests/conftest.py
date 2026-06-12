"""Shared test fixtures for wisemonkey tests."""

import json
import shutil
import tempfile
import unittest

from pathlib import Path


def reset_singletons():
    """Reset all singleton instances between tests."""
    import agent.config as config_mod
    import agent.memory as memory_mod

    # Config uses a custom __new__ that checks hasattr(cls, "_instance").
    # Setting _instance = None still makes hasattr return True, so __new__
    # returns None. We must delete the attribute entirely.
    if hasattr(config_mod.Config, "_instance"):
        del config_mod.Config._instance
    config_mod._config_instance = None

    if hasattr(memory_mod.Memory, "_instance"):
        del memory_mod.Memory._instance
    memory_mod._instance = None


class BaseTest(unittest.TestCase):
    """Base test class that handles singleton reset and temp directory cleanup."""

    def setUp(self):
        reset_singletons()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="wisemonkey-test-"))

    def tearDown(self):
        reset_singletons()
        if self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_file(self, name: str, content: str) -> Path:
        """Write a file inside the temp directory and return its path."""
        path = self._tmpdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _write_config(self, config: dict) -> Path:
        """Write a YAML config file and return its path."""
        import yaml
        return self._write_file("config.yaml", yaml.dump(config))

    def _write_json(self, name: str, data) -> Path:
        """Write a JSON file and return its path."""
        return self._write_file(name, json.dumps(data, indent=2))

    @staticmethod
    def _default_config() -> dict:
        """Return a minimal valid config dict matching the project defaults."""
        return {
            "model": {
                "name": "test-model",
                "base_url": "http://localhost:1234/v1",
                "temperature": 0.8,
                "thinking": {
                    "effort": "medium",
                    "display": False,
                },
            },
            "embedding": {
                "name": "test-embedding",
                "base_url": "",
            },
            "agent": {
                "max_turns": 10,
                "system_prompt": "You are a test assistant.",
                "markdown": False,
                "max_chat_history": 1000,
                "context_files": ["AGENTS.md"],
                "vi_mode": False,
            },
        }
