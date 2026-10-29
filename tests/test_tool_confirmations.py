"""Tests for the confirmation policy of the shell and network tools.

Two rules are pinned here:

1. ``run_command`` asks before running a command that references a file or
   directory outside the session working directory, on top of the
   pre-existing "matches a dangerous pattern" rule.
2. ``fetch_url`` and ``web_search`` always ask before a request leaves the
   machine.
3. The session is the working directory plus two open directories of its own
   session directory (``$XDG_DATA_HOME/wisemonkey/sessions/<name>``):
   ``pasted/`` and ``tmp/``. The rest of the session directory -- plans, notes,
   profile, chat history, the vector store -- is agent state reached through
   its own tool, so a path to it still asks; a sibling session's directory
   asks too. The roots are derived from the session rather than hard-coded, so
   the boundary stays one rule instead of a growing list of trusted locations.
4. The file tools (``read_file``, ``write_file``, ``patch_file``, ``list_dir``,
   ``find_files``, ``search_content``) ask on the same boundary: reaching a
   path outside the session is an explicit decision whichever tool names it,
   and going through a shell was only ever the older way to reach one.

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
from tools.files import contractuser
from tools.terminal import (
    _candidate_paths,
    _confirmation_reasons,
    _is_within_session,
    _outside_session_paths,
    _tool_env,
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
        # The session's scratch directory, as the handler will see it. Every
        # run gets one, so the tests exercise the real environment path
        # instead of whatever the host happens to have.
        self.session_temp = self._tmpdir / "session-tmp"

    def _run(self, command, answer=False, unsafe=False, **args):
        output = _RecordingOutput(answer=answer)
        set_output(cast(OutputAdapter, output))
        self.addCleanup(set_output, None)
        with patch.dict(
            self.globals,
            {
                "get_config": lambda: _Config(unsafe),
                "_session_temp_dir": lambda: self.session_temp,
            },
        ):
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

    def test_the_session_temp_dir_runs_without_a_prompt(self):
        output, result, run = self._run(f"cat {self.session_temp}/scratch.txt")
        assert output.confirmations == []
        assert run.call_count == 1
        assert result["success"] is True

    def test_the_subprocess_gets_the_session_temp_dir_as_TMPDIR(self):
        _, _, run = self._run("python plot.py")
        assert run.call_args.kwargs["env"]["TMPDIR"] == str(self.session_temp)

    def test_the_shared_tmp_still_prompts(self):
        """The carve-out is the session's own directory, not ``/tmp`` itself:
        that one is world-readable and full of other programs' state."""
        output, _, run = self._run("cat /tmp/other-process.state")
        assert len(output.confirmations) == 1
        assert run.call_count == 0


class TestSessionTempDir(BaseTest):
    """Tool subprocesses run with a private ``TMPDIR`` inside the session.

    Keeping the scratch directory *inside* the session is what lets a command
    use a temp directory without prompting for it -- and without landing in
    the shared, world-readable ``/tmp`` that the boundary check exists to ask
    about in the first place.
    """

    def setUp(self):
        super().setUp()
        self.temp = self._tmpdir / "session-tmp"

    def _patch_temp(self):
        return patch("tools.terminal._session_temp_dir", return_value=self.temp)

    def _patch_session_dir(self, path):
        return patch("tools.terminal._session_pasted_dir", return_value=path)

    def test_tool_env_points_TMPDIR_at_the_session_temp_dir(self):
        with self._patch_temp():
            env = _tool_env()
        assert env["TMPDIR"] == str(self.temp)

    def test_tool_env_creates_the_directory_privately(self):
        with self._patch_temp():
            _tool_env()
        assert self.temp.is_dir()
        assert (self.temp.stat().st_mode & 0o777) == 0o700

    def test_tool_env_carries_the_rest_of_the_environment_over(self):
        """``env=`` replaces the child's environment, so it must not be a
        fresh dict: PATH and the API keys have to survive."""
        with patch.dict("os.environ", {"WM_API_KEY": "secret"}, clear=True):
            with self._patch_temp():
                env = _tool_env()
        assert env["WM_API_KEY"] == "secret"

    def test_tool_env_leaves_an_inherited_temp_dir_alone_without_a_session(self):
        with patch.dict("os.environ", {"TMPDIR": "/inherited/tmp"}, clear=True):
            with patch("tools.terminal._session_temp_dir", return_value=None):
                env = _tool_env()
        assert env["TMPDIR"] == "/inherited/tmp"

    def test_an_unwritable_scratch_directory_does_not_block_the_command(self):
        """A missing scratch directory is never a reason to refuse to run."""
        with patch.dict("os.environ", {"TMPDIR": "/inherited/tmp"}, clear=True):
            with self._patch_temp():
                with patch("pathlib.Path.mkdir", side_effect=OSError("read-only")):
                    env = _tool_env()
        assert env["TMPDIR"] == "/inherited/tmp"

    def test_the_session_temp_dir_is_inside_the_session(self):
        with self._patch_temp():
            assert _is_within_session(self.temp)
            assert _is_within_session(self.temp / "run-1" / "out.txt")

    def test_the_shared_tmp_is_not_inside_the_session(self):
        with self._patch_temp():
            assert not _is_within_session("/tmp")
            assert not _is_within_session("/tmp/wisemonkey-scratch.txt")

    def test_the_session_pasted_dir_is_inside_the_session(self):
        """Pasted files are the user's own text, so they need no prompt.

        They live under the session directory, not the working directory, so
        without this root the agent could not read back a long prompt it had
        spilled to disk.
        """
        pasted = self._tmpdir / "pasted"
        with self._patch_session_dir(pasted):
            assert _is_within_session(pasted)
            assert _is_within_session(pasted / "paste_20260101_120000_000000.md")

    def test_the_rest_of_the_session_directory_still_prompts(self):
        """Plans, notes and chat history are agent state, not open files.

        Each has a tool of its own, so reaching the file by path is reaching
        around it -- and the model has no business rewriting the history the
        agent is reading from.
        """
        pasted = self._tmpdir / "pasted"
        with self._patch_session_dir(pasted):
            assert not _is_within_session(self._tmpdir)
            assert not _is_within_session(self._tmpdir / "plans" / "PLAN.md")
            assert not _is_within_session(self._tmpdir / "notes.json")
            assert not _is_within_session(self._tmpdir / "chat_history.json")
            assert not _is_within_session(self._tmpdir / "vectorstore")

    def test_a_sibling_session_directory_is_outside(self):
        pasted = self._tmpdir / "pasted"
        with self._patch_session_dir(pasted):
            sibling = self._tmpdir.parent / "another-session"
            assert not _is_within_session(sibling)
            assert not _is_within_session(sibling / "pasted" / "paste_x.md")

    def test_no_session_bound_trusts_only_the_working_directory(self):
        """A harness with no session must not accidentally trust a default one."""
        with self._patch_session_dir(None):
            assert not _is_within_session(self._tmpdir)

    def test_writing_to_the_session_temp_dir_is_not_an_outside_path(self):
        with self._patch_temp():
            assert _outside_session_paths(f"python plot.py --out {self.temp}/p.png") == []

    def test_writing_to_the_shared_tmp_is_still_an_outside_path(self):
        assert _outside_session_paths("python plot.py --out /tmp/p.png") == ["/tmp/p.png"]


class TestFindFilesConfirmation(BaseTest):
    """``find_files`` asks on the same rule, from the same helper."""

    def setUp(self):
        super().setUp()
        discover_tools()
        self.find = get_registry()["find_files"]["handler"]
        self.output = _RecordingOutput(answer=False)
        set_output(cast(OutputAdapter, self.output))
        self.addCleanup(set_output, None)
        self.temp = self._tmpdir / "session-tmp"

    def _find(self, root):
        with patch("tools.terminal._session_temp_dir", return_value=self.temp):
            return self.find({"root": str(root), "pattern": "*.txt"})

    def test_the_session_temp_dir_does_not_prompt(self):
        self.temp.mkdir(parents=True, exist_ok=True)
        result = self._find(self.temp)
        assert self.output.confirmations == []
        assert "error" not in result

    def test_a_relative_subdirectory_of_the_working_directory_does_not_prompt(self):
        result = self._find("tests")
        assert self.output.confirmations == []
        assert "error" not in result

    def test_a_directory_outside_the_session_prompts(self):
        elsewhere = self._tmpdir / "elsewhere"
        elsewhere.mkdir()
        result = self._find(elsewhere)
        assert len(self.output.confirmations) == 1
        assert result["user_cancelled"] is True


class TestFileToolConfirmation(BaseTest):
    """The file tools ask on the same boundary the shell tools use.

    A path outside the session is an explicit decision whichever tool names
    it: ``run_command`` was the first tool to reach one, not the only one.
    ``agent.unsafe`` silences these prompts like every other.
    """

    def setUp(self):
        super().setUp()
        discover_tools()
        self.handlers = {
            name: get_registry()[name]["handler"]
            for name in (
                "read_file",
                "write_file",
                "patch_file",
                "list_dir",
                "search_content",
            )
        }
        # The file tools all come out of one module, so one globals dict
        # carries their ``get_config``.
        self.globals = self.handlers["read_file"].__globals__
        self.temp = self._tmpdir / "session-tmp"
        self.outside = self._write_file("outside.txt", "outside content\n")

    def _call(self, tool, args, answer=False, unsafe=False, pasted_dir=None):
        output = _RecordingOutput(answer=answer)
        set_output(cast(OutputAdapter, output))
        self.addCleanup(set_output, None)
        with patch.dict(self.globals, {"get_config": lambda: _Config(unsafe)}):
            with patch("tools.terminal._session_temp_dir", return_value=self.temp):
                with patch("tools.terminal._session_pasted_dir", return_value=pasted_dir):
                    result = self.handlers[tool](args)
        return output, result

    def test_a_pasted_file_does_not_prompt(self):
        """Pasted text is the user's own writing, not agent state.

        It lives under the session directory, so without this root the agent
        could not read back a long prompt it had spilled to disk.
        """
        session = self._tmpdir / "session"
        pasted = session / "pasted"
        pasted.mkdir(parents=True)
        paste = pasted / "paste_20260101_120000_000000.md"
        paste.write_text("a long prompt\n")
        output, result = self._call(
            "read_file", {"path": str(paste)}, pasted_dir=pasted
        )
        assert output.confirmations == []
        assert result["content"] == "a long prompt\n"

    def test_a_plan_file_still_prompts(self):
        """Plans are read through ``/plan-read``; the path is around that."""
        session = self._tmpdir / "session"
        (session / "plans").mkdir(parents=True)
        plan = session / "plans" / "PLAN.md"
        plan.write_text("# Plan\n")
        output, result = self._call(
            "read_file", {"path": str(plan)}, pasted_dir=session / "pasted"
        )
        assert len(output.confirmations) == 1
        assert result["user_cancelled"] is True

    def test_another_sessions_pasted_dir_still_prompts(self):
        session = self._tmpdir / "session"
        other = self._tmpdir / "other-session"
        (other / "pasted").mkdir(parents=True)
        paste = other / "pasted" / "paste_x.md"
        paste.write_text("theirs\n")
        output, result = self._call(
            "read_file", {"path": str(paste)}, pasted_dir=session / "pasted"
        )
        assert len(output.confirmations) == 1
        assert result["user_cancelled"] is True

    def test_a_file_in_the_working_directory_does_not_prompt(self):
        output, result = self._call("read_file", {"path": "tests/conftest.py"})
        assert output.confirmations == []
        assert "content" in result

    def test_a_file_in_the_session_temp_dir_does_not_prompt(self):
        self.temp.mkdir(parents=True, exist_ok=True)
        (self.temp / "scratch.txt").write_text("scratch\n")
        output, result = self._call(
            "read_file", {"path": str(self.temp / "scratch.txt")}
        )
        assert output.confirmations == []
        assert result["content"] == "scratch\n"

    def test_reading_outside_the_session_prompts_and_names_the_file(self):
        output, result = self._call("read_file", {"path": str(self.outside)})
        assert len(output.confirmations) == 1
        # The prompt shows the *contracted* path (``contractuser``, the same
        # shortening the REPL use), so the home prefix reads as
        # ``~``. Comparing against the raw absolute path only matched while the
        # temp directory happened to sit outside $HOME -- and ``mkdtemp``
        # honours $TMPDIR, which the agent points at its own scratch dir inside
        # the home tree. Compare the form the user actually sees.
        assert any(contractuser(str(self.outside)) in line for line in output.lines)
        assert result["user_cancelled"] is True

    def test_a_confirmed_read_returns_the_content(self):
        output, result = self._call(
            "read_file", {"path": str(self.outside)}, answer=True
        )
        assert len(output.confirmations) == 1
        assert result["content"] == "outside content\n"

    def test_a_missing_file_reports_missing_rather_than_asking(self):
        """Existence is checked first: asking about a typo would be noise,
        and the answer would not change the outcome."""
        output, result = self._call("read_file", {"path": str(self._tmpdir / "nope.txt")})
        assert output.confirmations == []
        assert "does not exist" in result["error"]

    def test_writing_outside_the_session_prompts_and_writes_nothing(self):
        target = self._tmpdir / "new.txt"
        output, result = self._call("write_file", {"path": str(target), "content": "x"})
        assert len(output.confirmations) == 1
        assert result["user_cancelled"] is True
        assert not target.exists()

    def test_a_declined_write_creates_no_parent_directories(self):
        target = self._tmpdir / "nested" / "new.txt"
        result = self._call("write_file", {"path": str(target), "content": "x"})[1]
        assert result["user_cancelled"] is True
        assert not target.parent.exists()

    def test_a_confirmed_write_lands(self):
        target = self._tmpdir / "new.txt"
        self._call("write_file", {"path": str(target), "content": "x"}, answer=True)
        assert target.read_text() == "x"

    def test_patching_outside_the_session_prompts_and_leaves_the_file_alone(self):
        output, result = self._call(
            "patch_file",
            {
                "path": str(self.outside),
                "old_string": "outside content",
                "new_string": "inside content",
            },
        )
        assert len(output.confirmations) == 1
        assert result["user_cancelled"] is True
        assert self.outside.read_text() == "outside content\n"

    def test_a_confirmed_patch_lands(self):
        self._call(
            "patch_file",
            {
                "path": str(self.outside),
                "old_string": "outside content",
                "new_string": "inside content",
            },
            answer=True,
        )
        assert self.outside.read_text() == "inside content\n"

    def test_listing_outside_the_session_prompts(self):
        output, result = self._call("list_dir", {"path": str(self._tmpdir)})
        assert len(output.confirmations) == 1
        assert result["user_cancelled"] is True

    def test_searching_outside_the_session_prompts(self):
        output, result = self._call(
            "search_content", {"root": str(self._tmpdir), "query": "outside"}
        )
        assert len(output.confirmations) == 1
        assert result["user_cancelled"] is True

    def test_searching_the_session_temp_dir_does_not_prompt(self):
        self.temp.mkdir(parents=True, exist_ok=True)
        (self.temp / "scratch.txt").write_text("needle\n")
        output, result = self._call(
            "search_content", {"root": str(self.temp), "query": "needle"}
        )
        assert output.confirmations == []
        assert result["count"] == 1

    def test_the_session_directory_around_the_scratch_dir_is_still_outside(self):
        """The scratch directory is trusted; the session directory holding it
        (chat history, notes) is not."""
        self.temp.mkdir(parents=True, exist_ok=True)
        notes = self._write_file("notes.json", "{}")
        output, result = self._call("read_file", {"path": str(notes)})
        assert len(output.confirmations) == 1
        assert result["user_cancelled"] is True

    def test_unsafe_mode_skips_the_file_prompts(self):
        output, result = self._call(
            "read_file", {"path": str(self.outside)}, unsafe=True
        )
        assert output.confirmations == []
        assert result["content"] == "outside content\n"


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
