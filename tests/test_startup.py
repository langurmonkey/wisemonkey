"""Tests for the startup banner and the previous-conversation recap.

The recap is the one part of startup that reads stored state, so the tests
build a real session with a tool-heavy tail: that is the shape that made the
panel show two tool results and no question.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import agent.memory as memory_mod

from agent.console import monkee_theme
from agent.output import OutputAdapter

from agent.memory import Memory
from agent.memory import SESSIONS_DIR as _REAL_SESSIONS_DIR


def _memory_with_history(entries):
    """A Memory whose chat history is exactly *entries*.

    *entries* is a list of ``(role, content)`` or ``(role, content, extra)``.
    History is written to disk, because the recap only ever reads what a
    previous process left there.
    """
    from tests.conftest import reset_singletons

    reset_singletons()
    root = Path(tempfile.mkdtemp(prefix="wisemonkey-startup-"))
    sessions = root / "sessions"
    sessions.mkdir(parents=True)
    memory_mod.SESSIONS_DIR = sessions

    memory = Memory(session="default")
    for entry in entries:
        role, content, extra = (list(entry) + [{}])[:3]
        memory._chat_history.add_exchange(None, role, content, **extra)
    memory._chat_history.save()
    memory.save()
    return memory


def _restore_sessions_dir():
    memory_mod.SESSIONS_DIR = _REAL_SESSIONS_DIR


class _RecordingOutput:
    """Collects what the panel code emits, without a terminal."""

    def __init__(self):
        self.text: list[str] = []
        self.rich: list[object] = []

    def print(self, text="", end="\n", indent=0):
        self.text.append(text)

    def print_rich(self, renderable, **kw):
        self.rich.append(renderable)

    def info(self, text=""):
        self.text.append(text)

    def rule(self, *a, **kw):
        pass

    def newline(self):
        pass

    def panel(self, *a, **kw):
        pass

    def ask_choice(self, *a, **kw):
        raise AssertionError("the startup recap must not prompt")

    def ask_string(self, *a, **kw):
        raise AssertionError("the startup recap must not prompt")


def _recap_panel(output):
    """The Panel holding the previous-conversation recap, if any."""
    from rich.panel import Panel

    for renderable in output.rich:
        if isinstance(renderable, Panel) and "Previous conversation" in str(
            renderable.title
        ):
            return renderable
    return None


def _recap_body(panel):
    """The recap as rendered text, which is what the user actually reads.

    `rich.markdown.Markdown` keeps no plain-text copy of its source, so the
    panel is rendered through a narrow console instead of read off the object.
    Width is generous enough not to wrap mid-word and hide the assertions.
    """
    from io import StringIO

    from rich.console import Console

    console = Console(
        file=StringIO(),
        width=400,
        force_terminal=False,
        no_color=True,
        theme=monkee_theme,
    )
    console.print(panel)
    stream = console.file
    assert isinstance(stream, StringIO)
    return stream.getvalue()


class TestStartupRecap(unittest.TestCase):
    def setUp(self):
        from unittest import mock

        self._mock = mock
        patcher = mock.patch(
            "agent.startup.check_updates", return_value=(False, None, None)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        from tests.conftest import reset_singletons

        reset_singletons()
        _restore_sessions_dir()

    def _run(self, entries):
        memory = _memory_with_history(entries)
        core = SimpleNamespace(
            memory=memory,
            config=SimpleNamespace(get=lambda *a, **k: False),
            get_context_breakdown=lambda: [("System", 100)],
        )
        output = _RecordingOutput()
        from agent.startup import startup_info

        startup_info(core, cast(OutputAdapter, output))
        return _recap_panel(output), output

    def test_a_tool_heavy_tail_still_shows_the_question(self):
        panel, _ = self._run(
            [
                ("user", "fix the failing test"),
                ("tool_call", "", {"name": "run_command", "arguments": "{}"}),
                ("tool_result", "AssertionError: boom", {"name": "run_command"}),
                ("assistant", "Fixed it; 594 tests pass."),
                ("user", "now run it again"),
                ("tool_result", "Ran 594 tests, OK", {"name": "run_command"}),
            ]
        )
        self.assertIsNotNone(panel)
        body = _recap_body(panel)
        self.assertIn("now run it again", body)
        self.assertNotIn("AssertionError", body)
        self.assertNotIn("Ran 594", body)

    def test_one_exchange_is_one_question_and_one_answer(self):
        """A turn with several tool steps recaps as a single Q/A pair.

        Narration before each tool call is stored as its own assistant
        message, so a turn that ran three tools has three of them. Counting
        entries instead of turns filled the panel with Assistant blocks and
        pushed the question off the end.
        """
        panel, _ = self._run(
            [
                ("user", "the question"),
                ("assistant", "Let me look at that."),
                ("tool_call", "", {"name": "search_content", "arguments": "{}"}),
                ("tool_result", "matches", {"name": "search_content"}),
                ("assistant", "Now let me read it."),
                ("tool_call", "", {"name": "read_file", "arguments": "{}"}),
                ("tool_result", "the file", {"name": "read_file"}),
                ("assistant", "The final answer."),
            ]
        )
        self.assertIsNotNone(panel)
        body = _recap_body(panel)
        self.assertIn("the question", body)
        self.assertIn("The final answer.", body)
        self.assertNotIn("Let me look at that.", body)
        self.assertNotIn("Now let me read it.", body)
        self.assertNotIn("matches", body)
        self.assertIn("last 1 exchange", str(panel.title))

    def test_the_title_counts_what_is_actually_shown(self):
        panel, _ = self._run(
            [
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
                ("user", "q3"),
            ]
        )
        self.assertIsNotNone(panel)
        # The default is one turn. With a pending question and no answer yet,
        # that is just the question -- a recap is a "where were we" note, not
        # a transcript.
        self.assertIn("last 1 exchange", str(panel.title))
        body = _recap_body(panel)
        self.assertIn("q3", body)
        self.assertNotIn("q2", body)
        self.assertNotIn("q1", body)

    def test_the_whole_turn_keeps_only_its_last_answer(self):
        panel, _ = self._run(
            [
                ("user", "q1"),
                ("assistant", "intermediate narration"),
                ("tool_call", "", {"name": "run_command", "arguments": "{}"}),
                ("tool_result", "output", {"name": "run_command"}),
                ("assistant", "the real answer to q1."),
                ("user", "q2"),
                ("assistant", "the real answer to q2."),
            ]
        )
        body = _recap_body(panel)
        self.assertIn("q2", body)
        self.assertIn("the real answer to q2.", body)
        self.assertNotIn("q1", body)
        self.assertNotIn("intermediate narration", body)

    def test_no_history_means_no_panel(self):
        panel, _ = self._run([])
        self.assertIsNone(panel)


if __name__ == "__main__":
    unittest.main()