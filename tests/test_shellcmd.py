"""Tests for user-invoked shell commands (the `!` prefix)."""

from typing import Any, cast
from unittest.mock import MagicMock

from tests.conftest import BaseTest

from agent.shellcmd import append_to_memory, run_shell_command


class _FakeOutput:
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
        self.out = cast(Any, _FakeOutput())

    def test_output_exit_code_and_chat_content(self):
        result = run_shell_command("echo hello", self.out)
        assert (result["exit_code"], result["success"]) == (0, True)
        assert "hello" in result["stdout"]
        assert "hello" in "\n".join(self.out.printed)
        assert "$ echo hello" in result["_chat_content"]
        assert "exit code: 0" in result["_chat_content"]

        result = run_shell_command("echo oops >&2; exit 3", self.out)
        assert result["exit_code"] == 3 and result["success"] is False
        assert any("oops" in line for line in self.out.printed)
        assert any("3" in line for line in self.out.errors)

    def test_timeout_and_a_missing_binary(self):
        result = run_shell_command("sleep 5", self.out, timeout=1)
        assert "timed out" in result["error"]
        assert run_shell_command("definitely_not_a_command_xyz", self.out)[
            "exit_code"
        ] == 127


class TestAppendToMemory(BaseTest):
    def test_appends_the_exchange_and_skips_when_there_is_no_content(self):
        core = MagicMock()
        append_to_memory(core, "echo hi", {"stdout": "hi"})
        core.memory.add_chat_exchange.assert_not_called()

        append_to_memory(core, "echo hi", {"_chat_content": "$ echo hi\nhi\nexit code: 0"})
        role, content = core.memory.add_chat_exchange.call_args[0][1:3]
        assert role == "user" and "$ echo hi" in content
