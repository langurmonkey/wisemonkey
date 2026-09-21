"""Tests for user-invoked shell commands (the `!` prefix)."""

import unittest
from typing import Any, cast
from unittest.mock import MagicMock

from tests.conftest import BaseTest

from agent.shellcmd import append_to_memory, run_shell_command


class FakeOutput:
    def __init__(self):
        self.printed = []
        self.errors = []

    def print(self, text, **kw):
        self.printed.append(text)

    def err(self, text, **kw):
        self.errors.append(text)

    def ok(self, text, **kw):
        self.printed.append(text)


class TestRunShellCommand(BaseTest):
    def setUp(self):
        super().setUp()
        self.out = cast(Any, FakeOutput())

    def test_simple_command(self):
        result = run_shell_command("echo hello", self.out)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["success"])
        self.assertIn("hello", result["stdout"])
        self.assertIn("hello", "\n".join(self.out.printed))

    def test_stderr_reported(self):
        result = run_shell_command("echo oops >&2", self.out)
        self.assertIn("oops", result["stderr"])
        self.assertTrue(any("oops" in line for line in self.out.printed))

    def test_nonzero_exit_code(self):
        result = run_shell_command("exit 3", self.out)
        self.assertEqual(result["exit_code"], 3)
        self.assertFalse(result["success"])
        self.assertTrue(any("3" in e for e in self.out.errors))

    def test_timeout(self):
        result = run_shell_command("sleep 5", self.out, timeout=1)
        self.assertIn("error", result)
        self.assertIn("timed out", result["error"])

    def test_chat_content_built(self):
        result = run_shell_command("echo hi", self.out)
        content = result["_chat_content"]
        self.assertIn("$ echo hi", content)
        self.assertIn("hi", content)
        self.assertIn("exit code: 0", content)

    def test_invalid_command(self):
        result = run_shell_command("definitely_not_a_command_xyz", self.out)
        self.assertEqual(result["exit_code"], 127)


class TestAppendToMemory(BaseTest):
    def _make_core(self):
        core = MagicMock()
        return core

    def test_appends_exchange(self):
        core = self._make_core()
        result = {"_chat_content": "$ echo hi\nhi\nexit code: 0"}
        append_to_memory(core, "echo hi", result)
        core.memory.add_chat_exchange.assert_called_once()
        args = core.memory.add_chat_exchange.call_args[0]
        self.assertEqual(args[1], "user")
        self.assertIn("$ echo hi", args[2])

    def test_skips_when_no_content(self):
        core = self._make_core()
        append_to_memory(core, "echo hi", {"stdout": "hi"})
        core.memory.add_chat_exchange.assert_not_called()


if __name__ == "__main__":
    unittest.main()
