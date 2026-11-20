"""Tests for the startup banner's previous-conversation recap."""

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import agent.memory as memory_mod
from agent.console import monkee_theme
from agent.memory import Memory, SESSIONS_DIR as _REAL_SESSIONS_DIR
from agent.output import OutputAdapter
from agent.startup import startup_info
from rich.console import Console
from rich.panel import Panel
from tests.conftest import reset_singletons


class _RecordingOutput:
    """Collects what the panel code emits, without a terminal."""

    def __init__(self):
        self.rich: list[object] = []

    def print(self, *a, **kw):
        pass

    def print_rich(self, renderable, **kw):
        self.rich.append(renderable)

    def info(self, *a, **kw):
        pass

    def rule(self, *a, **kw):
        pass

    def newline(self, *a, **kw):
        pass

    def panel(self, *a, **kw):
        pass

    def ask_choice(self, *a, **kw):
        raise AssertionError("the startup recap must not prompt")

    ask_string = ask_choice


class TestStartupRecap(unittest.TestCase):
    def setUp(self):
        for p, kw in ((mock.patch("agent.startup.check_updates",
                                  return_value=(False, None, None)), {}),):
            p.start()
            self.addCleanup(p.stop)
        reset_singletons()
        self.addCleanup(reset_singletons)
        root = Path(tempfile.mkdtemp(prefix="wisemonkey-startup-"))
        memory_mod.SESSIONS_DIR = root / "sessions"
        memory_mod.SESSIONS_DIR.mkdir(parents=True)
        self.addCleanup(setattr, memory_mod, "SESSIONS_DIR", _REAL_SESSIONS_DIR)

    def _run(self, entries):
        memory = Memory(session="default")
        for role, content, *_extra in entries:
            memory._chat_history.add_exchange(None, role, content)
        core = SimpleNamespace(
            memory=memory,
            config=SimpleNamespace(get=lambda *a, **k: False),
            get_context_breakdown=lambda: [("System", 100)],
        )
        output = _RecordingOutput()
        startup_info(core, cast(OutputAdapter, output))
        panel = next((r for r in output.rich
                      if isinstance(r, Panel)
                      and "Previous conversation" in str(r.title)), None)
        return panel

    @staticmethod
    def _body(panel) -> str:
        console = Console(file=StringIO(), width=400, force_terminal=False,
                          no_color=True, theme=monkee_theme)
        console.print(panel)
        return cast(StringIO, console.file).getvalue()

    def test_a_tool_heavy_tail_recaps_as_one_question_and_one_answer(self):
        panel = self._run([
            ("user", "fix the failing test"),
            ("tool_call", ""),
            ("tool_result", "AssertionError: boom"),
            ("assistant", "Let me look at that."),
            ("assistant", "Fixed it; 594 tests pass."),
        ])
        assert panel is not None
        body = self._body(panel)
        assert "fix the failing test" in body
        assert "Fixed it; 594 tests pass." in body
        # Tool traffic and intermediate narration are not part of the recap.
        for noise in ("AssertionError", "Let me look at that"):
            assert noise not in body, noise
        assert "last 1 exchange" in str(panel.title)

    def test_only_the_most_recent_exchange_is_shown(self):
        panel = self._run([("user", "q1"), ("assistant", "a1"),
                           ("user", "q2"), ("assistant", "a2")])
        assert panel is not None
        body = self._body(panel)
        assert "q2" in body and "a2" in body
        assert "q1" not in body

    def test_no_history_means_no_panel(self):
        assert self._run([]) is None
