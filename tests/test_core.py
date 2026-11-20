"""Tests for agent/core.py — workspace discovery, context/soul loading, prompt order."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from agent.config import Config
from agent.core import Core
from tests.conftest import BaseTest


class _CoreCase(BaseTest):
    """A Core built without __init__ (no router, no MCP) over the real Config."""

    def setUp(self):
        super().setUp()
        self.config = Config()
        self.core = Core.__new__(Core)
        self.core.config = self.config

    def _cwd(self):
        return patch("agent.core.Path.cwd", return_value=self._tmpdir)


class TestWorkspaceRoot(_CoreCase):
    def test_walks_up_to_the_deepest_directory_holding_a_context_file(self):
        sub = self._tmpdir / "a" / "b"
        sub.mkdir(parents=True)
        (self._tmpdir / "AGENTS.md").write_text("# Root", encoding="utf-8")
        assert self.core._find_workspace_root(sub) == self._tmpdir
        # The deepest match wins, not the outermost one.
        (sub / "AGENTS.md").write_text("# Sub", encoding="utf-8")
        assert self.core._find_workspace_root(sub) == sub

    def test_falls_back_to_the_start_directory_and_honours_custom_names(self):
        sub = self._tmpdir / "sub"
        sub.mkdir()
        assert self.core._find_workspace_root(sub) == sub
        self.config.set_no_save("agent.context_files", ["CUSTOM.md"])
        (self._tmpdir / "CUSTOM.md").write_text("# Custom", encoding="utf-8")
        assert self.core._find_workspace_root(sub) == self._tmpdir


class TestContextFiles(_CoreCase):
    def test_loads_configured_files_and_caches_them(self):
        self.config.set_no_save("agent.context_files", ["AGENTS.md", "MISSING.md"])
        (self._tmpdir / "AGENTS.md").write_text("# Hello", encoding="utf-8")
        with self._cwd():
            first = self.core._load_context_files()
        assert "## Workspace Instructions (AGENTS.md)" in first
        assert "# Hello" in first
        assert "MISSING" not in first  # an absent file is skipped, not an error
        # The second call is the cache: an edit on disk is not picked up.
        (self._tmpdir / "AGENTS.md").write_text("# Changed", encoding="utf-8")
        with self._cwd():
            assert self.core._load_context_files() == first

    def test_no_files_yields_an_empty_string(self):
        self.config.set_no_save("agent.context_files", ["NONEXISTENT.md"])
        with self._cwd():
            assert self.core._load_context_files() == ""


class TestSoulFiles(_CoreCase):
    def test_global_soul_precedes_the_workspace_soul(self):
        global_dir = self._tmpdir / "global"
        global_dir.mkdir()
        (global_dir / "SOUL.md").write_text("# Global", encoding="utf-8")
        (self._tmpdir / "SOUL.md").write_text("# Workspace", encoding="utf-8")
        with patch("agent.core.BASE_CONFIG_DIR", global_dir), self._cwd():
            result = self.core._load_soul_files()
        assert result.index("# Global") < result.index("# Workspace")
        assert "## Global Soul (SOUL.md)" in result
        assert "## Workspace Soul (SOUL.md)" in result

    def test_an_empty_config_name_disables_that_file(self):
        for key in ("agent.soul_file", "agent.global_soul_file"):
            self.config.set_no_save(key, "")
        (self._tmpdir / "SOUL.md").write_text("# Should not load", encoding="utf-8")
        with self._cwd():
            assert self.core._load_soul_files() == ""

    def test_a_custom_workspace_soul_name_is_used(self):
        self.config.set_no_save("agent.soul_file", "IDENTITY.md")
        (self._tmpdir / "IDENTITY.md").write_text("# Custom", encoding="utf-8")
        with patch("agent.core.BASE_CONFIG_DIR", self._tmpdir / "nope"), self._cwd():
            result = self.core._load_soul_files()
        assert "## Workspace Soul (IDENTITY.md)" in result


class TestBuildSystemPrompt(_CoreCase):
    """Chat history is the tail, so the static prefix stays cacheable."""

    def _prompt(self, **overrides):
        core = cast(Any, self.core)
        core.system_prompt = "BASE"
        core._load_soul_files = overrides.get("soul", lambda: "SOUL")
        core._load_context_files = overrides.get("context", lambda: "CONTEXT")
        core.memory = type("M", (), {
            "get_user_profile_formatted": lambda self, notes=False: "MEMORY",
            "get_chat_history_formatted": lambda self, timestamps=False: "CHAT",
        })()
        core.skills = type("S", (), {"load_all": lambda self: "SKILLS"})()
        return self.core._build_system_prompt()

    def test_sections_appear_in_a_stable_order(self):
        assert self._prompt().split("\n") == [
            "BASE", "SOUL", "CONTEXT", "MEMORY", "SKILLS", "CHAT",
        ]

    def test_empty_sections_are_omitted(self):
        prompt = self._prompt(soul=lambda: "", context=lambda: "")
        assert prompt.split("\n") == ["BASE", "MEMORY", "SKILLS", "CHAT"]
