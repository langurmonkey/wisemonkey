"""Tests for reloading config into a running Core."""

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from agent.core import Core


class _Config:
    def __init__(self) -> None:
        self._config = {
            "model": {"name": "updated"},
            "agent": {
                "system_prompt": "Updated prompt",
                "max_chat_history": 1234,
                "skills": False,
            },
        }
        self.reloaded = False

    def to_dict(self) -> dict:
        return dict(self._config)

    def reload(self) -> None:
        self.reloaded = True

    def get(self, key: str, default=None):
        values = {
            "agent.system_prompt": self._config["agent"]["system_prompt"],
            "agent.max_chat_history": self._config["agent"]["max_chat_history"],
            "agent.skills": self._config["agent"]["skills"],
        }
        return values.get(key, default)


class TestCoreConfigReload(unittest.TestCase):
    def _core(self) -> tuple[Core, _Config, Mock]:
        core = cast(Any, Core.__new__(Core))
        config = _Config()
        history = Mock()
        core.config = config
        core.router = object()
        core.system_prompt = "old prompt"
        core.memory = SimpleNamespace(_chat_history=history)
        core.skills = object()
        core._context_files_cache = "stale context"
        core._soul_files_cache = "stale soul"
        return core, config, history

    def test_reload_refreshes_runtime_settings_and_caches(self) -> None:
        core, config, history = self._core()
        new_router = Mock()
        new_router.initialize.return_value = (True, None)

        with patch("agent.core.ModelRouter", return_value=new_router), patch(
            "agent.core.SkillLoader", return_value="new skills"
        ):
            ok, message = core.reload_config()

        assert ok is True
        assert message is None
        assert config.reloaded
        assert core.router is new_router
        assert core.system_prompt == "Updated prompt"
        assert history.max_tokens == 1234
        history._trim.assert_called_once_with()
        assert core.skills == "new skills"
        assert not hasattr(core, "_context_files_cache")
        assert not hasattr(core, "_soul_files_cache")

    def test_failed_router_reload_restores_previous_config_and_router(self) -> None:
        core, config, _history = self._core()
        previous_config = config.to_dict()
        previous_router = core.router
        new_router = Mock()
        new_router.initialize.return_value = (False, "invalid model configuration")

        with patch("agent.core.ModelRouter", return_value=new_router):
            ok, message = core.reload_config()

        assert ok is False
        assert message == "invalid model configuration"
        assert config._config == previous_config
        assert core.router is previous_router
        assert core.system_prompt == "old prompt"
