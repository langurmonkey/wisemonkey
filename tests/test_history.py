"""Tests for the prompt history and the /undo, /retry and /edit commands."""

import os
import subprocess
import tempfile
import unittest
from typing import Any, cast
from unittest import mock

from agent.history import PromptHistory


class _FakeMemory:
    """Minimal ChatMemory stand-in: a list of dicts."""

    def __init__(self, entries):
        self._exchanges = list(entries)

    def get_chat_history_unformatted(self):
        return self._exchanges

    def drop_last_exchanges(self, n):
        return self._drop_last(n)

    def last_user_prompt(self):
        for entry in reversed(self._exchanges):
            if entry.get("role") == "user":
                return entry.get("content", "")
        return ""

    # The real algorithm lives in ChatMemory; borrow it.
    def _drop_last(self, n):
        if not self._exchanges:
            return 0
        if n <= 0:
            return self._clear()
        starts = [i for i, e in enumerate(self._exchanges)
                  if e.get("role") == "user"]
        if len(starts) <= n:
            return self._clear()
        cutoff = starts[-n]
        removed = len(self._exchanges) - cutoff
        self._exchanges = self._exchanges[:cutoff]
        return removed

    def _clear(self):
        removed = len(self._exchanges)
        self._exchanges = []
        return removed


def _exchanges(*specs):
    return [{"role": role, "content": content} for role, content in specs]


def _run(core, command):
    from agent.commands import registry

    return registry.run_command(core, command)


def _core(*specs):
    core = cast(Any, mock.Mock())
    core.memory = _FakeMemory(_exchanges(*specs))
    core.pending_prompt = None
    return core


class TestPromptHistoryFiltering(unittest.TestCase):
    """Slash commands stay on disk but out of the navigable history."""

    def setUp(self) -> None:
        self.path = os.path.join(tempfile.mkdtemp(), "history.txt")

    def _history(self) -> PromptHistory:
        return PromptHistory(self.path)

    def test_commands_are_hidden_but_still_written(self):
        h = self._history()
        for line in ("/quit", "plot the data", "   /config", "/undo"):
            h.store_string(line)
        # prompt_toolkit's contract: newest first.
        assert self._history().load_history_strings() == ["plot the data"]
        with open(self.path, encoding="utf-8") as handle:
            assert "/quit" in handle.read()

        # A prompt that merely *contains* a slash line, or a bare path, is kept.
        self.path = os.path.join(tempfile.mkdtemp(), "history.txt")
        h = self._history()
        h.store_string("explain this:\n/quit means exit")
        h.store_string("agent/core.py")
        assert sorted(self._history().load_history_strings()) == [
            "agent/core.py", "explain this:\n/quit means exit",
        ]

    def test_an_empty_history_is_a_prompt_toolkit_history(self):
        from prompt_toolkit.history import History

        assert self._history().load_history_strings() == []
        assert isinstance(self._history(), History)


class TestUndoCommand(unittest.TestCase):
    """`/undo` drops whole exchanges from the tail."""

    SESSION = (
        ("user", "first"), ("assistant", "answer one"),
        ("user", "second"), ("assistant", "answer two"),
        ("tool_call", "{}"), ("tool_result", "42"),
    )

    def _roles(self, core):
        return [e["role"] for e in core.memory.get_chat_history_unformatted()]

    def test_drops_the_whole_exchange_including_its_tool_calls(self):
        core = _core(*self.SESSION)
        ok, msg, *_ = _run(core, "/undo")
        assert ok
        # Tool calls and results belong to their exchange: no orphan survives.
        assert self._roles(core) == ["user", "assistant"]
        assert "4" in msg  # four entries removed, not one exchange
        assert core.pending_prompt is None  # /undo never re-runs anything

    def test_counts_clearing_and_a_non_integer(self):
        core = _core(*self.SESSION)
        assert _run(core, "/undo 2")[0] is True
        assert self._roles(core) == []
        for core, command in ((_core(*self.SESSION), "/undo 99"),
                              (_core(), "/undo")):
            ok, msg, *_ = _run(core, command)
            assert ok is (command == "/undo 99"), command
            if ok:
                assert self._roles(core) == []
            else:
                assert "nothing to undo" in msg
        ok, msg, *_ = _run(_core(*self.SESSION), "/undo many")
        assert ok is False and "integer" in msg


class TestRetryCommand(unittest.TestCase):
    SESSION = (
        ("user", "plot it"), ("assistant", "here"),
        ("user", "wrong thing"), ("assistant", "bad answer"),
    )

    def test_re_runs_the_last_prompt_and_keeps_the_exchange(self):
        core = _core(*self.SESSION)
        assert _run(core, "/retry")[0] is True
        assert core.pending_prompt == "wrong thing"
        # Kept by default, so the model can see what it is being asked to redo.
        assert len(core.memory.get_chat_history_unformatted()) == 4

    def test_drop_removes_the_old_exchange_first_and_errors(self):
        """The re-run must see the context the original did."""
        core = _core(*self.SESSION)
        assert _run(core, "/retry --drop")[0] is True
        assert [e["content"]
                for e in core.memory.get_chat_history_unformatted()] == [
            "plot it", "here",
        ]
        ok, msg, *_ = _run(_core(*self.SESSION), "/retry --whatever")
        assert ok is False and "--whatever" in msg
        core = _core()
        ok, msg, *_ = _run(core, "/retry")
        assert ok is False and "no previous prompt" in msg
        assert core.pending_prompt is None


class TestEditCommand(unittest.TestCase):
    def _core(self):
        return _core(("user", "original prompt"), ("assistant", "answer"))

    def _edit(self, core, content=None):
        from agent import commands

        with mock.patch.object(commands, "_edit_in_editor", return_value=content):
            return _run(core, "/edit")

    def test_the_edited_prompt_replaces_the_old_exchange(self):
        core = self._core()
        ok, *_ = self._edit(core, "a better prompt\n")
        assert ok and core.pending_prompt == "a better prompt"
        contents = [e["content"]
                    for e in core.memory.get_chat_history_unformatted()]
        assert "original prompt" not in contents

    def test_an_unchanged_empty_or_aborted_edit_runs_nothing(self):
        for content, fragment in (("original prompt\n", ""), ("   \n", "empty"),
                                  (None, "")):
            core = self._core()
            ok, msg, *_ = self._edit(core, content)
            assert ok is False, content
            assert fragment in msg
            assert core.pending_prompt is None
        ok, msg, *_ = self._edit(_core(), "anything\n")
        assert ok is False and "no previous prompt" in msg
        assert _run(self._core(), "/edit extra")[0] is False


class TestLastPromptCommand(unittest.TestCase):
    def test_prints_the_last_user_message(self):
        core = _core(("user", "first"), ("assistant", "a"), ("user", "second"))
        ok, _msg, content, *_ = _run(core, "/last-prompt")
        assert ok and content == "second"
        assert _run(_core(), "/last-prompt")[0] is False


class TestEditInEditor(unittest.TestCase):
    """The scratch file is written, handed to $EDITOR, cleaned up."""

    def test_writes_scratch_file_reads_it_back_and_removes_it(self):
        from agent import commands

        seen = {}

        def fake_run(cmd):
            path = cmd[1]
            seen["path"] = path
            with open(path, encoding="utf-8") as handle:
                seen["before"] = handle.read()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("edited text\n")
            return mock.Mock(returncode=0)

        with mock.patch.object(subprocess, "run", fake_run):
            assert commands._edit_in_editor("original", None) == "edited text\n"
        assert seen["before"] == "original\n"
        assert not os.path.exists(seen["path"]), "scratch file left behind"

        with mock.patch.object(subprocess, "run",
                               return_value=mock.Mock(returncode=1)):
            assert commands._edit_in_editor("original", None) is None


class TestTakePendingPrompt(unittest.TestCase):
    """The frontend reads the prompt a command left behind, exactly once."""

    @staticmethod
    def _agent(**attrs):
        from agent.agent import Agent

        agent = Agent.__new__(Agent)
        for key, value in attrs.items():
            setattr(agent, key, value)
        return agent

    def test_returns_once_and_tolerates_a_stub_core(self):
        agent = self._agent(core=mock.Mock(pending_prompt="re-run me"))
        assert agent._take_pending_prompt() == "re-run me"
        assert agent._take_pending_prompt() is None
        assert self._agent(core=mock.Mock(spec=[]))._take_pending_prompt() is None

    def test_the_client_slot_wins_in_remote_mode(self):
        """The daemon owns the core, so the reply carries the prompt instead."""
        agent = self._agent(core=mock.Mock(pending_prompt="stale"),
                            _client_pending_prompt="from the server")
        assert agent._take_pending_prompt() == "from the server"
        assert agent._client_pending_prompt is None


if __name__ == "__main__":
    unittest.main()
