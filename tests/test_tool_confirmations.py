"""Tests for the confirmation policy of the shell and network tools.

Two rules are pinned here:

1. ``run_command`` asks before running a command that references a file or
   directory outside the session working directory, on top of the
   pre-existing "matches a dangerous pattern" rule.
2. ``fetch_url`` and ``web_search`` always ask before a request leaves the
   machine.

A declined prompt has to actually stop the work, so the tests assert both
that the user was asked (and about what) and that nothing ran when they said
no. The prompts themselves are side effects, hence the recording adapter.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tests.conftest import BaseTest
from agent.output import OutputAdapter, set_output
from agent.tools import discover_tools, get_registry
from tools.terminal import (
    _candidate_paths,
    _confirmation_reasons,
    _outside_session_paths,
)


class _RecordingOutput:
    """Output adapter stub that records prompts and answers them."""

    def __init__(self, answer: bool = False):
        self.answer = answer
        self.confirmations: list[str] = []
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        if args:
            self.lines.append(str(args[0]))

    def err(self, *args, **kwargs):
        pass

    def ok(self, *args, **kwargs):
        pass

    def newline(self, *args, **kwargs):
        pass

    def ask_confirm(self, message, default=False):
        self.confirmations.append(message)
        return self.answer


class _Config:
    """Config stub that answers only for ``agent.unsafe``."""

    def __init__(self, unsafe: bool = False):
        self.unsafe = unsafe

    def get(self, key, default=None):
        if key == "agent.unsafe":
            return self.unsafe
        return default


class _FakeResponse:
    """Minimal ``urlopen`` result: a context manager with headers and a body."""

    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestPathDetection(BaseTest):
    """The static analysis that decides which tokens name a filesystem path."""

    def test_flags_an_absolute_path(self):
        assert _candidate_paths("cat /etc/passwd") == ["/etc/passwd"]

    def test_ignores_a_bare_word(self):
        assert _candidate_paths("ls -la") == []

    def test_ignores_a_url(self):
        # URLs carry slashes but are not filesystem paths.
        assert _candidate_paths("curl https://example.com/a/b") == []

    def test_ignores_an_unexpandable_variable(self):
        # ``$HOME/x`` may or may not be inside the session; guessing is worse
        # than staying quiet.
        assert _candidate_paths('cat "$HOME/secret"') == []

    def test_reads_the_value_half_of_an_assignment(self):
        assert _candidate_paths("cmd --out=/tmp/x") == ["/tmp/x"]

    def test_inside_paths_are_not_reported(self):
        local = Path.cwd() / "setup.py"
        assert _outside_session_paths(f"cat {local}") == []

    def test_outside_paths_are_reported(self):
        assert _outside_session_paths("cat /etc/passwd") == ["/etc/passwd"]

    def test_parent_relative_path_is_outside(self):
        assert _outside_session_paths("ls ../") == ["../"]

    def test_home_is_outside_a_project_directory(self):
        # The session is the project directory, which is not ``$HOME``.
        assert _outside_session_paths("ls ~") == ["~"]

    def test_a_plain_local_command_has_no_reasons(self):
        assert _confirmation_reasons("ls -la", 30) == []

    def test_dangerous_patterns_still_report(self):
        reasons = _confirmation_reasons("rm -rf build", 30)
        assert any("dangerous" in r for r in reasons)

    def test_outside_path_is_a_reason_naming_the_path(self):
        reasons = _confirmation_reasons("ls /etc", 30)
        assert len(reasons) == 1
        assert "/etc" in reasons[0]


class TestRunCommandPathConfirmation(BaseTest):
    """``run_command`` prompts for paths outside the session directory."""

    def setUp(self):
        super().setUp()
        discover_tools()
        self.handler = get_registry()["run_command"]["handler"]
        # ``discover_tools()`` loads every tool file under its bare stem
        # (``terminal``), so the handler's globals are a *different* module
        # object from the ``tools.terminal`` imported above. Patch through
        # ``__globals__`` to hit the copy the handler actually reads.
        self.globals = self.handler.__globals__

    def _run(self, command, answer=False, unsafe=False, **args):
        output = _RecordingOutput(answer=answer)
        set_output(cast(OutputAdapter, output))
        self.addCleanup(set_output, None)
        with patch.dict(self.globals, {"get_config": lambda: _Config(unsafe)}):
            with patch("subprocess.run") as run:
                run.return_value = SimpleNamespace(
                    stdout="ok", stderr="", returncode=0
                )
                result = self.handler({"command": command, **args})
        return output, result, run

    def test_local_command_runs_without_a_prompt(self):
        output, result, run = self._run("ls -la")
        assert output.confirmations == []
        assert run.call_count == 1
        assert result["success"] is True

    def test_a_relative_path_inside_the_session_does_not_prompt(self):
        # ``sub/file.txt`` resolves under the cwd, so it is not an outside
        # path even though it carries a separator.
        output, _, run = self._run("cat sub/file.txt")
        assert output.confirmations == []
        assert run.call_count == 1

    def test_absolute_path_outside_the_session_prompts(self):
        output, _, run = self._run("cat /etc/hostname")
        assert len(output.confirmations) == 1
        assert run.call_count == 0

    def test_parent_path_prompts(self):
        output, _, run = self._run("ls ../")
        assert len(output.confirmations) == 1
        assert run.call_count == 0

    def test_redirection_outside_the_session_prompts(self):
        output, _, run = self._run("echo hi > ../out.txt")
        assert len(output.confirmations) == 1
        assert run.call_count == 0

    def test_home_path_prompts(self):
        output, _, run = self._run("ls ~")
        assert len(output.confirmations) == 1
        assert run.call_count == 0

    def test_declined_prompt_returns_user_cancelled(self):
        _, result, run = self._run("cat /etc/hostname")
        assert result["user_cancelled"] is True
        assert "error" in result
        assert run.call_count == 0

    def test_confirmed_prompt_runs_the_command(self):
        output, result, run = self._run("cat /etc/hostname", answer=True)
        assert len(output.confirmations) == 1
        assert run.call_count == 1
        assert result["stdout"] == "ok"

    def test_unsafe_mode_skips_the_prompt(self):
        output, result, run = self._run("cat /etc/hostname", unsafe=True)
        assert output.confirmations == []
        assert run.call_count == 1
        assert result["success"] is True

    def test_skip_confirmation_argument_still_wins(self):
        output, _, run = self._run("cat /etc/hostname", _skip_confirmation=True)
        assert output.confirmations == []
        assert run.call_count == 1


class TestNetworkConfirmation(BaseTest):
    """``fetch_url`` and ``web_search`` ask before they touch the network."""

    def setUp(self):
        super().setUp()
        discover_tools()
        self.fetch = get_registry()["fetch_url"]["handler"]
        self.search = get_registry()["web_search"]["handler"]
        # Same module-identity caveat as the terminal tests: the handler reads
        # the ``network`` copy loaded by ``discover_tools()``.
        self.globals = self.fetch.__globals__

    def _output(self, answer: bool):
        output = _RecordingOutput(answer=answer)
        set_output(cast(OutputAdapter, output))
        self.addCleanup(set_output, None)
        return output

    def _patch_config(self, unsafe=False):
        return patch.dict(self.globals, {"get_config": lambda: _Config(unsafe)})

    def test_fetch_url_prompts_and_shows_the_url(self):
        output = self._output(answer=False)
        with self._patch_config():
            with patch("urllib.request.urlopen") as urlopen:
                self.fetch({"url": "https://example.com/"})
        assert len(output.confirmations) == 1
        assert any("https://example.com/" in line for line in output.lines)
        assert urlopen.call_count == 0

    def test_fetch_url_cancelled_returns_error(self):
        self._output(answer=False)
        with self._patch_config():
            result = self.fetch({"url": "https://example.com/"})
        assert result["user_cancelled"] is True
        assert result["url"] == "https://example.com/"

    def test_fetch_url_confirmed_fetches(self):
        self._output(answer=True)
        response = _FakeResponse(b"<html><body>hi</body></html>")
        with self._patch_config():
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                result = self.fetch({"url": "https://example.com/"})
        assert urlopen.call_count == 1
        assert "hi" in result

    def test_fetch_url_unsafe_mode_skips_the_prompt(self):
        output = self._output(answer=False)
        response = _FakeResponse(b"<html>hi</html>")
        with self._patch_config(unsafe=True):
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                self.fetch({"url": "https://example.com/"})
        assert output.confirmations == []
        assert urlopen.call_count == 1

    def test_web_search_prompts_and_shows_the_query(self):
        output = self._output(answer=False)
        with self._patch_config():
            with patch("urllib.request.urlopen") as urlopen:
                self.search({"query": "wisemonkey cli"})
        assert len(output.confirmations) == 1
        assert any("wisemonkey cli" in line for line in output.lines)
        assert urlopen.call_count == 0

    def test_web_search_cancelled_returns_error(self):
        self._output(answer=False)
        with self._patch_config():
            result = self.search({"query": "wisemonkey cli"})
        assert result["user_cancelled"] is True
        assert result["query"] == "wisemonkey cli"

    def test_web_search_confirmed_parses_results(self):
        self._output(answer=True)
        page = (
            '<a class="result__a" href="https://example.com/a">Example A</a>'
            '<a class="result__snippet">First snippet</a>'
            '<a class="result__a" href="https://example.com/b">Example B</a>'
            '<a class="result__snippet">Second snippet</a>'
        )
        response = _FakeResponse(page.encode())
        with self._patch_config():
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                result = self.search({"query": "example"})
        assert urlopen.call_count == 1
        assert result["query"] == "example"
        assert [r["title"] for r in result["results"]] == ["Example A", "Example B"]

    def test_web_search_missing_query_does_not_prompt(self):
        output = self._output(answer=False)
        with self._patch_config():
            result = self.search({})
        assert result == {"error": "No query provided"}
        assert output.confirmations == []

    def test_safe_paths_do_not_prompt_in_the_other_direction(self):
        """A URL is not a path: fetching one is the network rule, not the path rule."""
        assert _outside_session_paths("curl https://example.com/a/b") == []


if __name__ == "__main__":
    import unittest

    unittest.main()
