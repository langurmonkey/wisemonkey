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

    def to_dict(self) -> dict:
        return dict(self._config)

    def reload(self) -> None:
        self.reloaded = True

    def get(self, key: str, default=None):
        section, _, leaf = key.partition(".")
        return self._config.get(section, {}).get(leaf, default)


class TestCoreConfigReload(unittest.TestCase):
    def _core(self) -> tuple[Core, _Config, Mock]:
        core = cast(Any, Core.__new__(Core))
        config = _Config()
        history = Mock()
        core.config = config
        core.session = "ses_test123"
        core.router = object()
        core.system_prompt = "old prompt"
        core.memory = SimpleNamespace(_chat_history=history)
        core.skills = object()
        core._context_files_cache = "stale context"
        core._soul_files_cache = "stale soul"
        return core, config, history

    def _reload(self, core, initialized=(True, None)):
        router = Mock()
        router.initialize.return_value = initialized
        with patch("agent.core.ModelRouter", return_value=router), patch(
            "agent.core.SkillLoader", return_value="new skills"
        ):
            return core.reload_config(), router

    def test_reload_refreshes_settings_and_drops_the_caches(self) -> None:
        core, config, history = self._core()
        (ok, message), router = self._reload(core)
        assert (ok, message) == (True, None)
        assert config.reloaded and core.router is router
        assert core.system_prompt == "Updated prompt"
        assert core.skills == "new skills"
        assert history.max_tokens == 1234
        history._trim.assert_called_once_with()
        assert not hasattr(core, "_context_files_cache")
        assert not hasattr(core, "_soul_files_cache")

    def test_a_failed_router_restores_config_and_router(self) -> None:
        core, config, _history = self._core()
        previous_config = config.to_dict()
        previous_router = core.router
        (ok, message), _router = self._reload(
            core, initialized=(False, "invalid model configuration")
        )
        assert (ok, message) == (False, "invalid model configuration")
        assert config._config == previous_config
        assert core.router is previous_router
        assert core.system_prompt == "old prompt"
