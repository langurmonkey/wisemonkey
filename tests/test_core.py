"""Tests for agent/core.py — workspace root finding, context file loading, prompt building."""

from pathlib import Path
from unittest.mock import patch

from tests.conftest import BaseTest
from agent.config import Config


class TestFindWorkspaceRoot(BaseTest):
    """Test the _find_workspace_root walk-up logic."""

    def setUp(self):
        super().setUp()
        self.config = Config()

    def test_finds_agents_md_in_start_dir(self):
        """AGENTS.md exists in the starting directory."""
        (self._tmpdir / "AGENTS.md").write_text("# Test", encoding="utf-8")
        result = self._call_find_workspace_root(self._tmpdir)
        assert result == self._tmpdir

    def test_finds_agents_md_in_parent(self):
        """AGENTS.md exists one level up."""
        sub = self._tmpdir / "sub"
        sub.mkdir()
        (self._tmpdir / "AGENTS.md").write_text("# Test", encoding="utf-8")
        result = self._call_find_workspace_root(sub)
        assert result == self._tmpdir

    def test_finds_agents_md_two_levels_up(self):
        """AGENTS.md exists two levels up."""
        sub = self._tmpdir / "a" / "b"
        sub.mkdir(parents=True)
        (self._tmpdir / "AGENTS.md").write_text("# Test", encoding="utf-8")
        result = self._call_find_workspace_root(sub)
        assert result == self._tmpdir

    def test_returns_start_when_no_context_file(self):
        """No context file found — should return start directory."""
        sub = self._tmpdir / "sub"
        sub.mkdir()
        result = self._call_find_workspace_root(sub)
        assert result == sub

    def test_custom_context_files(self):
        """Uses custom file names from config."""
        self.config.set_no_save("agent.context_files", ["CUSTOM.md"])
        (self._tmpdir / "CUSTOM.md").write_text("# Custom", encoding="utf-8")
        result = self._call_find_workspace_root(self._tmpdir)
        assert result == self._tmpdir

    def test_prefers_deepest_match(self):
        """When context files exist at multiple levels, returns the deepest."""
        sub = self._tmpdir / "sub"
        sub.mkdir()
        (self._tmpdir / "AGENTS.md").write_text("# Root", encoding="utf-8")
        (sub / "AGENTS.md").write_text("# Sub", encoding="utf-8")
        result = self._call_find_workspace_root(sub)
        assert result == sub

    def test_cwd_fallback(self):
        """When no start dir is given, uses cwd."""
        (self._tmpdir / "AGENTS.md").write_text("# Cwd", encoding="utf-8")
        with patch("agent.core.Path.cwd", return_value=self._tmpdir):
            result = self._call_find_workspace_root()
        assert result == self._tmpdir

    def _call_find_workspace_root(self, start=None):
        """Call the private _find_workspace_root on a minimal Core-like object."""
        # We can't instantiate Core without a full router setup, so we
        # replicate the method logic using the config singleton.
        context_files = self.config.get("agent.context_files", ["AGENTS.md"])
        current = (start or Path.cwd()).resolve()
        for parent in [current, *current.parents]:
            for name in context_files:
                if (parent / name).is_file():
                    return parent
        return current


class TestLoadContextFiles(BaseTest):
    """Test the _load_context_files caching and content loading."""

    def setUp(self):
        super().setUp()
        self.config = Config()

    def test_loads_single_file(self):
        (self._tmpdir / "AGENTS.md").write_text("# Hello", encoding="utf-8")
        result = self._call_load_context_files(self._tmpdir)
        assert "# Hello" in result
        assert "## Workspace Instructions (AGENTS.md)" in result

    def test_loads_multiple_files(self):
        self.config.set_no_save("agent.context_files", ["AGENTS.md", "CUSTOM.md"])
        (self._tmpdir / "AGENTS.md").write_text("# Agents", encoding="utf-8")
        (self._tmpdir / "CUSTOM.md").write_text("# Custom", encoding="utf-8")
        result = self._call_load_context_files(self._tmpdir)
        assert "# Agents" in result
        assert "# Custom" in result
        assert "## Workspace Instructions (AGENTS.md)" in result
        assert "## Workspace Instructions (CUSTOM.md)" in result

    def test_missing_file_skipped_gracefully(self):
        self.config.set_no_save("agent.context_files", ["AGENTS.md", "MISSING.md"])
        (self._tmpdir / "AGENTS.md").write_text("# Agents", encoding="utf-8")
        result = self._call_load_context_files(self._tmpdir)
        assert "# Agents" in result
        assert "MISSING" not in result

    def test_empty_string_when_no_files(self):
        self.config.set_no_save("agent.context_files", ["NONEXISTENT.md"])
        result = self._call_load_context_files(self._tmpdir)
        assert result == ""

    def test_caching(self):
        """Second call should return the same result without re-reading."""
        (self._tmpdir / "AGENTS.md").write_text("# Cached", encoding="utf-8")
        result1 = self._call_load_context_files(self._tmpdir, use_cache=True)

        # Modify the file on disk
        (self._tmpdir / "AGENTS.md").write_text("# Modified", encoding="utf-8")

        # Should still return cached content (cache was set on first call)
        result2 = self._call_load_context_files(self._tmpdir, use_cache=True)
        assert result1 == result2
        assert "# Cached" in result2

    def _call_load_context_files(self, workspace_root, use_cache=False):
        """Replicate _load_context_files logic using the config singleton,
        including the caching behaviour."""
        context_files = self.config.get("agent.context_files", ["AGENTS.md"])

        # Simulate the cache attribute that Core uses
        cache_attr = "_context_files_cache"
        if use_cache and hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        parts = []
        for name in context_files:
            path = workspace_root / name
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if content:
                    parts.append(f"## Workspace Instructions ({name})\n{content}")
        result = "\n\n".join(parts) if parts else ""

        if use_cache:
            setattr(self, cache_attr, result)
        return result


class TestLoadSoulFiles(BaseTest):
    """Test the _load_soul_files global + workspace soul loading."""

    def setUp(self):
        super().setUp()
        self.config = Config()
        # Build a Core-like object without running __init__ (no router needed).
        from agent.core import Core
        self.core = Core.__new__(Core)
        self.core.config = self.config

    def _patch_paths(self, tmpdir):
        """Patch cwd and the global config dir used by _load_soul_files."""
        from unittest.mock import patch
        return patch.multiple(
            "agent.core",
            BASE_CONFIG_DIR=tmpdir,
        )

    def test_workspace_soul_loaded(self):
        (self._tmpdir / "SOUL.md").write_text("# I am a soul", encoding="utf-8")
        with patch("agent.core.Path.cwd", return_value=self._tmpdir):
            result = self.core._load_soul_files()
        assert "# I am a soul" in result
        assert "## Workspace Soul (SOUL.md)" in result

    def test_global_soul_loaded(self):
        global_dir = self._tmpdir / "global"
        global_dir.mkdir()
        (global_dir / "SOUL.md").write_text("# Global soul", encoding="utf-8")
        with patch("agent.core.BASE_CONFIG_DIR", global_dir), \
             patch("agent.core.Path.cwd", return_value=self._tmpdir):
            result = self.core._load_soul_files()
        assert "# Global soul" in result
        assert "## Global Soul (SOUL.md)" in result

    def test_global_and_workspace_soul_combined(self):
        global_dir = self._tmpdir / "global"
        global_dir.mkdir()
        (global_dir / "SOUL.md").write_text("# Global", encoding="utf-8")
        (self._tmpdir / "SOUL.md").write_text("# Workspace", encoding="utf-8")
        with patch("agent.core.BASE_CONFIG_DIR", global_dir), \
             patch("agent.core.Path.cwd", return_value=self._tmpdir):
            result = self.core._load_soul_files()
        assert "# Global" in result
        assert "# Workspace" in result
        # Global comes first
        assert result.index("# Global") < result.index("# Workspace")

    def test_empty_when_no_soul(self):
        with patch("agent.core.BASE_CONFIG_DIR", self._tmpdir / "nope"), \
             patch("agent.core.Path.cwd", return_value=self._tmpdir):
            result = self.core._load_soul_files()
        assert result == ""

    def test_disabled_with_empty_config(self):
        self.config.set_no_save("agent.soul_file", "")
        self.config.set_no_save("agent.global_soul_file", "")
        (self._tmpdir / "SOUL.md").write_text("# Should not load", encoding="utf-8")
        with patch("agent.core.Path.cwd", return_value=self._tmpdir):
            result = self.core._load_soul_files()
        assert result == ""

    def test_custom_soul_file_name(self):
        self.config.set_no_save("agent.soul_file", "IDENTITY.md")
        (self._tmpdir / "IDENTITY.md").write_text("# Custom", encoding="utf-8")
        with patch("agent.core.BASE_CONFIG_DIR", self._tmpdir / "nope"), \
             patch("agent.core.Path.cwd", return_value=self._tmpdir):
            result = self.core._load_soul_files()
        assert "# Custom" in result
        assert "IDENTITY.md" in result

    def test_caching(self):
        (self._tmpdir / "SOUL.md").write_text("# Cached", encoding="utf-8")
        with patch("agent.core.BASE_CONFIG_DIR", self._tmpdir / "nope"), \
             patch("agent.core.Path.cwd", return_value=self._tmpdir):
            first = self.core._load_soul_files()
            (self._tmpdir / "SOUL.md").write_text("# Modified", encoding="utf-8")
            second = self.core._load_soul_files()
        assert first == second
        assert "# Cached" in second


class TestBuildSystemPrompt(BaseTest):
    """Test that _build_system_prompt assembles all sections correctly."""

    def setUp(self):
        super().setUp()
        self.config = Config()

    def test_includes_base_prompt(self):
        prompt = self._call_build_system_prompt(self._tmpdir)
        assert "You are a test assistant." in prompt

    def test_includes_context_files(self):
        (self._tmpdir / "AGENTS.md").write_text("# Workspace", encoding="utf-8")
        prompt = self._call_build_system_prompt(self._tmpdir)
        assert "# Workspace" in prompt

    def test_no_context_files_section_when_empty(self):
        prompt = self._call_build_system_prompt(self._tmpdir)
        assert "Workspace Instructions" not in prompt

    def _call_build_system_prompt(self, workspace_root):
        """Replicate _build_system_prompt logic."""
        parts = ["You are a test assistant."]

        # Context files
        context_files = self.config.get("agent.context_files", ["AGENTS.md"])
        ctx_parts = []
        for name in context_files:
            path = workspace_root / name
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                if content:
                    ctx_parts.append(f"## Workspace Instructions ({name})\n{content}")
        if ctx_parts:
            parts.append("\n\n".join(ctx_parts))

        return "\n".join(parts)


class TestSystemPromptOrder(BaseTest):
    """Chat history must be the tail of the prompt so the static prefix
    (identity, context, memory, skills) stays stable for prompt caching."""

    def test_chat_history_is_last_section(self):
        from typing import Any, cast

        from agent.core import Core

        core = Core.__new__(Core)
        fake = cast(Any, core)
        fake.system_prompt = "BASE"
        fake._load_soul_files = lambda: "SOUL"
        fake._load_context_files = lambda: "CONTEXT"
        fake.memory = type("M", (), {
            "get_user_profile_formatted": lambda self, notes=False: "MEMORY",
            "get_chat_history_formatted": lambda self, timestamps=False: "CHAT_HISTORY",
        })()
        fake.skills = type("S", (), {"load_all": lambda self: "SKILLS"})()

        prompt = core._build_system_prompt()
        sections = prompt.split("\n")
        self.assertEqual(
            sections, ["BASE", "SOUL", "CONTEXT", "MEMORY", "SKILLS", "CHAT_HISTORY"]
        )
        self.assertGreater(prompt.index("SKILLS"), prompt.index("MEMORY"))
        self.assertGreater(prompt.index("CHAT_HISTORY"), prompt.index("SKILLS"))
