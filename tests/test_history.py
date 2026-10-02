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
        self.saved = 0

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
            removed = len(self._exchanges)
            self._exchanges = []
            return removed
        starts = [i for i, e in enumerate(self._exchanges)
                  if e.get("role") == "user"]
        if len(starts) <= n:
            removed = len(self._exchanges)
            self._exchanges = []
            return removed
        cutoff = starts[-n]
        removed = len(self._exchanges) - cutoff
        self._exchanges = self._exchanges[:cutoff]
        return removed


def _exchanges(*specs):
    """Build entries from (role, content) pairs."""
    return [{"role": role, "content": content} for role, content in specs]


class TestPromptHistoryFiltering(unittest.TestCase):
    """Slash commands stay on disk but out of the navigable history."""

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "history.txt")

    def _history(self) -> PromptHistory:
        return PromptHistory(self.path)

    def test_commands_are_filtered_out(self):
        h = self._history()
        for line in ("/quit", "plot the data", "/config", "another prompt"):
            h.store_string(line)
        # prompt_toolkit's contract: newest first.
        self.assertEqual(
            self._history().load_history_strings(),
            ["another prompt", "plot the data"],
        )

    def test_commands_are_still_written_to_disk(self):
        self._history().store_string("/quit")
        with open(self.path, encoding="utf-8") as handle:
            self.assertIn("/quit", handle.read())

    def test_multiline_prompt_with_slash_line_is_kept(self):
        """A prompt that merely *contains* a slash line is ordinary text."""
        h = self._history()
        h.store_string("explain this:\n/quit means exit")
        self.assertEqual(
            self._history().load_history_strings(),
            ["explain this:\n/quit means exit"],
        )

    def test_indented_command_is_filtered(self):
        self._history().store_string("   /quit")
        self.assertEqual(self._history().load_history_strings(), [])

    def test_paths_are_kept(self):
        """A bare path is not a command and is a very common thing to re-run."""
        self._history().store_string("agent/core.py")
        self.assertEqual(
            self._history().load_history_strings(), ["agent/core.py"])

    def test_empty_history(self):
        self.assertEqual(self._history().load_history_strings(), [])

    def test_is_a_prompt_toolkit_history(self):
        """It must be usable wherever prompt_toolkit expects a History."""
        from prompt_toolkit.history import History

        self.assertIsInstance(self._history(), History)


class TestUndoCommand(unittest.TestCase):
    """`/undo` drops whole exchanges from the tail."""

    def setUp(self) -> None:
        self.core = cast(Any, mock.Mock())
        self.core.memory = _FakeMemory(_exchanges(
            ("user", "first"),
            ("assistant", "answer one"),
            ("user", "second"),
            ("assistant", "answer two"),
            ("tool_call", "{}"),
            ("tool_result", "42"),
        ))
        self.core.pending_prompt = None

    def _run(self, command="/undo", *params):
        from agent.commands import registry

        return registry.run_command(self.core, " ".join([command, *params]))

    def test_drops_the_last_whole_exchange(self):
        ok, msg, *_ = self._run()
        self.assertTrue(ok)
        roles = [e["role"] for e in self.core.memory.get_chat_history_unformatted()]
        self.assertEqual(roles, ["user", "assistant"])

    def test_never_leaves_an_orphaned_tool_result(self):
        """Tool calls/results belong to their exchange, so they go together."""
        self._run()
        roles = [e["role"] for e in self.core.memory.get_chat_history_unformatted()]
        self.assertNotIn("tool_result", roles)

    def test_counts_entries_not_exchanges(self):
        ok, msg, *_ = self._run()
        self.assertTrue(ok)
        # user + assistant + tool_call + tool_result
        self.assertIn("4", msg)

    def test_multiple_exchanges(self):
        self._run("/undo", "2")
        roles = [e["role"] for e in self.core.memory.get_chat_history_unformatted()]
        self.assertEqual(roles, [])

    def test_more_than_available_clears_everything(self):
        ok, *_ = self._run("/undo", "99")
        self.assertTrue(ok)
        self.assertEqual(
            self.core.memory.get_chat_history_unformatted(), [])

    def test_bad_parameter(self):
        ok, msg, *_ = self._run("/undo", "many")
        self.assertFalse(ok)
        self.assertIn("integer", msg)

    def test_nothing_to_undo(self):
        self.core.memory = _FakeMemory([])
        ok, msg, *_ = self._run()
        self.assertFalse(ok)
        self.assertIn("nothing to undo", msg)

    def test_does_not_set_a_pending_prompt(self):
        """`/undo` only removes context; it does not re-run anything."""
        self._run()
        self.assertIsNone(self.core.pending_prompt)


class TestRetryCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.core = cast(Any, mock.Mock())
        self.core.memory = _FakeMemory(_exchanges(
            ("user", "plot it"),
            ("assistant", "here"),
            ("user", "wrong thing"),
            ("assistant", "bad answer"),
        ))
        self.core.pending_prompt = None

    def _run(self, *params):
        from agent.commands import registry

        return registry.run_command(self.core, " ".join(["/retry", *params]))

    def test_re_runs_the_last_prompt(self):
        ok, *_ = self._run()
        self.assertTrue(ok)
        self.assertEqual(self.core.pending_prompt, "wrong thing")

    def test_keeps_the_exchange_by_default(self):
        self._run()
        self.assertEqual(len(self.core.memory.get_chat_history_unformatted()), 4)

    def test_drop_removes_the_old_exchange_first(self):
        """The re-run must see the context the original did."""
        ok, *_ = self._run("--drop")
        self.assertTrue(ok)
        contents = [e["content"] for e in
                    self.core.memory.get_chat_history_unformatted()]
        self.assertEqual(contents, ["plot it", "here"])

    def test_unknown_option(self):
        ok, msg, *_ = self._run("--whatever")
        self.assertFalse(ok)
        self.assertIn("--whatever", msg)

    def test_no_previous_prompt(self):
        self.core.memory = _FakeMemory([])
        ok, msg, *_ = self._run()
        self.assertFalse(ok)
        self.assertIn("no previous prompt", msg)


class TestEditCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.core = cast(Any, mock.Mock())
        self.core.memory = _FakeMemory(_exchanges(
            ("user", "original prompt"),
            ("assistant", "answer"),
        ))
        self.core.pending_prompt = None

    def _run(self, editor_result=0, content=None):
        from agent import commands

        edited = "original prompt\n" if content is None else content
        with mock.patch.object(commands, "_edit_in_editor",
                               return_value=edited), \
             mock.patch.object(subprocess, "run",
                               return_value=mock.Mock(returncode=editor_result)):
            return commands.registry.run_command(self.core, "/edit")

    def test_edited_prompt_replaces_the_old_one(self):
        ok, msg, *_ = self._run(content="a better prompt\n")
        self.assertTrue(ok)
        self.assertEqual(self.core.pending_prompt, "a better prompt")
        contents = [e["content"] for e in
                    self.core.memory.get_chat_history_unformatted()]
        self.assertNotIn("original prompt", contents)

    def test_unchanged_prompt_does_not_rerun(self):
        ok, msg, *_ = self._run(content="original prompt\n")
        self.assertFalse(ok)
        self.assertIsNone(self.core.pending_prompt)

    def test_cancelled_editor_does_not_rerun(self):
        from agent import commands

        with mock.patch.object(commands, "_edit_in_editor", return_value=None):
            ok, msg, *_ = commands.registry.run_command(self.core, "/edit")
        self.assertFalse(ok)
        self.assertIsNone(self.core.pending_prompt)

    def test_empty_edit_does_not_rerun(self):
        ok, msg, *_ = self._run(content="   \n")
        self.assertFalse(ok)
        self.assertIn("empty", msg)

    def test_no_previous_prompt(self):
        self.core.memory = _FakeMemory([])
        ok, msg, *_ = self._run()
        self.assertFalse(ok)
        self.assertIn("no previous prompt", msg)

    def test_rejects_parameters(self):
        from agent import commands

        ok, msg, *_ = commands.registry.run_command(self.core, "/edit extra")
        self.assertFalse(ok)


class TestLastPromptCommand(unittest.TestCase):
    def test_prints_the_last_user_message(self):
        from agent.commands import registry

        core = cast(Any, mock.Mock())
        core.memory = _FakeMemory(_exchanges(
            ("user", "first"),
            ("assistant", "a"),
            ("user", "second"),
        ))
        ok, msg, content, markdown, *_ = registry.run_command(core, "/last-prompt")
        self.assertTrue(ok)
        self.assertEqual(content, "second")

    def test_empty_session(self):
        from agent.commands import registry

        core = cast(Any, mock.Mock())
        core.memory = _FakeMemory([])
        ok, msg, *_ = registry.run_command(core, "/last-prompt")
        self.assertFalse(ok)


class TestEditInEditor(unittest.TestCase):
    """The scratch file is written, handed to $EDITOR, cleaned up."""

    def test_writes_scratch_file_and_reads_back(self):
        from agent import commands

        written = {}

        class _Result:
            returncode = 0

        def fake_run(cmd):
            path = cmd[1]
            written["path"] = path
            with open(path, encoding="utf-8") as handle:
                written["before"] = handle.read()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("edited text\n")
            return _Result()

        with mock.patch.object(subprocess, "run", fake_run):
            result = commands._edit_in_editor("original", None)

        self.assertEqual(result, "edited text\n")
        self.assertEqual(written["before"], "original\n")
        self.assertFalse(os.path.exists(written["path"]), "scratch file left behind")

    def test_nonzero_exit_returns_none(self):
        from agent import commands

        with mock.patch.object(subprocess, "run",
                               return_value=mock.Mock(returncode=1)):
            self.assertIsNone(commands._edit_in_editor("original", None))


class TestTakePendingPrompt(unittest.TestCase):
    """The frontend reads the prompt a command left behind, exactly once."""

    def test_returns_and_clears(self):
        from agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.core = cast(Any, mock.Mock(pending_prompt="re-run me"))
        self.assertEqual(agent._take_pending_prompt(), "re-run me")
        self.assertIsNone(agent._take_pending_prompt())

    def test_none_when_empty(self):
        from agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.core = cast(Any, mock.Mock(pending_prompt=None))
        self.assertIsNone(agent._take_pending_prompt())

    def test_core_without_the_attribute(self):
        """A stub Core (remote mode) must not explode."""
        from agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.core = cast(Any, mock.Mock(spec=[]))
        self.assertIsNone(agent._take_pending_prompt())


if __name__ == "__main__":
    unittest.main()