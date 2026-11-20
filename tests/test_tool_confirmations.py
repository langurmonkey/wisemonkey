"""Tests for the confirmation policy of the shell, network and file tools."""

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
        return self.unsafe if key == "agent.unsafe" else default


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

    def test_only_real_paths_are_candidates(self):
        for command, expected in (
            ("cat /etc/passwd", ["/etc/passwd"]),
            ("ls -la", []),                           # bare words
            ("curl https://example.com/a/b", []),     # a URL
            ('cat "$HOME/secret"', []),               # unexpandable
            ("cmd --out=/tmp/x", ["/tmp/x"]),         # assignment
        ):
            self.assertEqual(_candidate_paths(command), expected, command)

    def test_inside_the_session_is_silent_and_outside_is_not(self):
        self.assertEqual(_outside_session_paths(f"cat {Path.cwd() / 'setup.py'}"), [])
        self.assertEqual(_outside_session_paths("ls -la"), [])
        for command, token in (("cat /etc/passwd", "/etc/passwd"),
                               ("ls ../", "../"), ("ls ~", "~")):
            self.assertEqual(_outside_session_paths(command), [token], command)
        # Over-flagging is deliberate: reading /etc prompts like writing there.
        self.assertEqual(_confirmation_reasons("ls -la", 30), [])
        self.assertTrue(any("dangerous" in r
                            for r in _confirmation_reasons("rm -rf build", 30)))
        reasons = _confirmation_reasons("ls /etc", 30)
        self.assertEqual(len(reasons), 1)
        self.assertIn("/etc", reasons[0])


class TestRunCommandPathConfirmation(BaseTest):
    """``run_command`` prompts for paths outside the session directory."""

    def setUp(self):
        super().setUp()
        discover_tools()
        self.handler = get_registry()["run_command"]["handler"]
        # ``discover_tools()`` loads each tool file under its bare stem, so the
        # handler's globals are a different module object: patch __globals__.
        self.globals = self.handler.__globals__
        self.session_temp = self._tmpdir / "session-tmp"

    def _run(self, command, answer=False, unsafe=False, **args):
        output = _RecordingOutput(answer=answer)
        set_output(cast(OutputAdapter, output))
        self.addCleanup(set_output, None)
        with patch.dict(self.globals, {
            "get_config": lambda: _Config(unsafe),
            "_session_temp_dir": lambda: self.session_temp,
        }), patch("subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout="ok", stderr="", returncode=0)
            result = self.handler({"command": command, **args})
        return output, result, run

    def test_commands_inside_the_session_run_without_a_prompt(self):
        for command in ("ls -la", "cat sub/file.txt",
                        f"cat {self.session_temp}/scratch.txt"):
            output, result, run = self._run(command)
            self.assertEqual(output.confirmations, [], command)
            self.assertEqual(run.call_count, 1, command)
            self.assertTrue(result["success"], command)

    def test_commands_reaching_outside_the_session_are_refused_on_a_decline(self):
        for command in ("cat /etc/hostname", "ls ../", "echo hi > ../out.txt",
                        "ls ~", "cat /tmp/other-process.state"):
            output, result, run = self._run(command)
            self.assertEqual(len(output.confirmations), 1, command)
            self.assertEqual(run.call_count, 0, command)
            self.assertTrue(result["user_cancelled"], command)

    def test_confirmation_unsafe_and_the_internal_flag_all_run_it(self):
        for kwargs, prompts in (({"answer": True}, 1), ({"unsafe": True}, 0),
                                ({"_skip_confirmation": True}, 0)):
            output, result, run = self._run("cat /etc/hostname", **kwargs)
            self.assertEqual(len(output.confirmations), prompts, kwargs)
            self.assertEqual(run.call_count, 1, kwargs)
            self.assertEqual(result["stdout"], "ok", kwargs)
        # The child gets a TMPDIR inside the session, not the shared /tmp.
        _, _, run = self._run("python plot.py")
        self.assertEqual(run.call_args.kwargs["env"]["TMPDIR"], str(self.session_temp))


class TestSessionTempDir(BaseTest):
    """Tool subprocesses run with a private ``TMPDIR`` inside the session."""

    _UNSET = object()

    def setUp(self):
        super().setUp()
        self.temp = self._tmpdir / "session-tmp"

    def _env(self, temp=_UNSET, environ=None):
        with patch.dict("os.environ", environ or {}, clear=True), \
                patch("tools.terminal._session_temp_dir",
                      return_value=self.temp if temp is self._UNSET else temp):
            return _tool_env()

    def test_tool_env_points_tmpdir_at_a_private_scratch_directory(self):
        env = self._env(environ={"WM_API_KEY": "secret"})
        self.assertEqual(env["TMPDIR"], str(self.temp))
        self.assertEqual(env["WM_API_KEY"], "secret")  # env= replaces the child's
        self.assertEqual(self.temp.stat().st_mode & 0o777, 0o700)
        # No session bound, or a scratch dir that cannot be created: inherit.
        inherited = {"TMPDIR": "/inherited/tmp"}
        self.assertEqual(self._env(temp=None, environ=inherited)["TMPDIR"],
                         "/inherited/tmp")
        with patch("pathlib.Path.mkdir", side_effect=OSError("read-only")):
            self.assertEqual(self._env(environ=inherited)["TMPDIR"], "/inherited/tmp")

    def test_the_session_scratch_and_pasted_dirs_are_inside(self):
        pasted = self._tmpdir / "pasted"
        with patch("tools.terminal._session_temp_dir", return_value=self.temp), \
                patch("tools.terminal._session_pasted_dir", return_value=pasted):
            self.assertTrue(_is_within_session(self.temp))
            self.assertTrue(_is_within_session(self.temp / "run-1" / "out.txt"))
            self.assertTrue(_is_within_session(pasted / "paste_20260101.md"))
            # Writing into the scratch space is not an outside path.
            self.assertEqual(_outside_session_paths(
                f"python plot.py --out {self.temp}/p.png"), [])
        self.assertEqual(_outside_session_paths("python plot.py --out /tmp/p.png"),
                         ["/tmp/p.png"])

    def test_agent_state_and_other_sessions_are_outside(self):
        """Plans, notes and history are reached through their own tools."""
        pasted = self._tmpdir / "pasted"
        sibling = self._tmpdir.parent / "another-session"
        with patch("tools.terminal._session_temp_dir", return_value=self.temp), \
                patch("tools.terminal._session_pasted_dir", return_value=pasted):
            for path in (self._tmpdir, self._tmpdir / "plans" / "PLAN.md",
                         self._tmpdir / "notes.json",
                         self._tmpdir / "chat_history.json",
                         "/tmp", sibling, sibling / "pasted" / "paste_x.md"):
                self.assertFalse(_is_within_session(path), path)
        # A harness with no session bound must not trust a default one.
        with patch("tools.terminal._session_pasted_dir", return_value=None):
            self.assertFalse(_is_within_session(self._tmpdir))


class TestFindFilesConfirmation(BaseTest):
    """``find_files`` asks on the same rule, from the same helper."""

    def test_inside_is_silent_and_outside_prompts(self):
        discover_tools()
        find = get_registry()["find_files"]["handler"]
        output = _RecordingOutput(answer=False)
        set_output(cast(OutputAdapter, output))
        self.addCleanup(set_output, None)
        temp = self._tmpdir / "session-tmp"
        temp.mkdir(parents=True, exist_ok=True)
        with patch("tools.terminal._session_temp_dir", return_value=temp):
            for root in (temp, "tests"):
                result = find({"root": str(root), "pattern": "*.txt"})
                self.assertEqual(output.confirmations, [], root)
                self.assertNotIn("error", result, root)
            elsewhere = self._tmpdir / "elsewhere"
            elsewhere.mkdir()
            result = find({"root": str(elsewhere), "pattern": "*.txt"})
        self.assertEqual(len(output.confirmations), 1)
        self.assertTrue(result["user_cancelled"])


class TestFileToolConfirmation(BaseTest):
    """The file tools ask on the same boundary the shell tools use."""

    def setUp(self):
        super().setUp()
        discover_tools()
        self.handlers = {
            name: get_registry()[name]["handler"]
            for name in ("read_file", "write_file", "patch_file", "list_dir",
                         "search_content")
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
        with patch.dict(self.globals, {"get_config": lambda: _Config(unsafe)}), \
                patch("tools.terminal._session_temp_dir", return_value=self.temp), \
                patch("tools.terminal._session_pasted_dir", return_value=pasted_dir):
            result = self.handlers[tool](args)
        return output, result

    def test_the_users_own_paste_is_trusted_and_agent_state_is_not(self):
        """Pasted text is the user's writing; plans and history are agent state."""
        session = self._tmpdir / "session"
        pasted = session / "pasted"
        (session / "plans").mkdir(parents=True)
        pasted.mkdir()
        paste = pasted / "paste_20260101_120000_000000.md"
        paste.write_text("a long prompt\n")
        output, result = self._call("read_file", {"path": str(paste)}, pasted_dir=pasted)
        self.assertEqual(output.confirmations, [])
        self.assertEqual(result["content"], "a long prompt\n")

        plan = session / "plans" / "PLAN.md"
        plan.write_text("# Plan\n")
        other = self._tmpdir / "other-session" / "pasted"
        other.mkdir(parents=True)
        theirs = other / "paste_x.md"
        theirs.write_text("theirs\n")
        for path in (plan, theirs):
            output, result = self._call("read_file", {"path": str(path)},
                                        pasted_dir=pasted)
            self.assertEqual(len(output.confirmations), 1, path)
            self.assertTrue(result["user_cancelled"], path)

    def test_files_inside_the_session_do_not_prompt(self):
        self.temp.mkdir(parents=True, exist_ok=True)
        (self.temp / "scratch.txt").write_text("scratch\n")
        for path in ("tests/conftest.py", str(self.temp / "scratch.txt")):
            output, result = self._call("read_file", {"path": path})
            self.assertEqual(output.confirmations, [], path)
            self.assertIn("content", result, path)

    def test_reading_outside_names_the_file_and_a_confirm_returns_it(self):
        output, result = self._call("read_file", {"path": str(self.outside)})
        self.assertEqual(len(output.confirmations), 1)
        # Compare the *contracted* path the prompt shows (~), not the raw one:
        # mkdtemp honours TMPDIR, which points inside the home tree.
        self.assertTrue(any(contractuser(str(self.outside)) in line
                            for line in output.lines))
        self.assertTrue(result["user_cancelled"])

        output, result = self._call("read_file", {"path": str(self.outside)},
                                    answer=True)
        self.assertEqual(result["content"], "outside content\n")
        # unsafe mode skips the prompt on the same boundary.
        output, result = self._call("read_file", {"path": str(self.outside)},
                                    unsafe=True)
        self.assertEqual(output.confirmations, [])
        self.assertEqual(result["content"], "outside content\n")

    def test_a_missing_file_reports_missing_rather_than_asking(self):
        """Existence is checked first: the answer could not change the outcome."""
        output, result = self._call("read_file",
                                    {"path": str(self._tmpdir / "nope.txt")})
        self.assertEqual(output.confirmations, [])
        self.assertIn("does not exist", result["error"])

    def test_a_declined_write_leaves_no_file_and_no_directory(self):
        target = self._tmpdir / "nested" / "new.txt"
        output, result = self._call("write_file",
                                    {"path": str(target), "content": "x"})
        self.assertEqual(len(output.confirmations), 1)
        self.assertTrue(result["user_cancelled"])
        self.assertFalse(target.parent.exists())
        self._call("write_file", {"path": str(target), "content": "x"}, answer=True)
        self.assertEqual(target.read_text(), "x")

    def test_patch_prompts_and_a_decline_leaves_the_file_alone(self):
        patch_args = {"path": str(self.outside),
                      "old_string": "outside content",
                      "new_string": "inside content"}
        output, result = self._call("patch_file", patch_args)
        self.assertEqual(len(output.confirmations), 1)
        self.assertTrue(result["user_cancelled"])
        self.assertEqual(self.outside.read_text(), "outside content\n")
        self._call("patch_file", patch_args, answer=True)
        self.assertEqual(self.outside.read_text(), "inside content\n")

    def test_listing_and_searching_ask_unless_the_root_is_trusted(self):
        for tool, args in (("list_dir", {"path": str(self._tmpdir)}),
                           ("search_content", {"root": str(self._tmpdir),
                                               "query": "outside"})):
            output, result = self._call(tool, args)
            self.assertEqual(len(output.confirmations), 1, tool)
            self.assertTrue(result["user_cancelled"], tool)

        self.temp.mkdir(parents=True, exist_ok=True)
        (self.temp / "scratch.txt").write_text("needle\n")
        output, result = self._call("search_content",
                                    {"root": str(self.temp), "query": "needle"})
        self.assertEqual(output.confirmations, [])
        self.assertEqual(result["count"], 1)


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

    def _call(self, handler, args, answer=False, unsafe=False, response=None):
        output = _RecordingOutput(answer=answer)
        set_output(cast(OutputAdapter, output))
        self.addCleanup(set_output, None)
        with patch.dict(self.globals, {"get_config": lambda: _Config(unsafe)}), \
                patch("urllib.request.urlopen",
                      return_value=response or _FakeResponse(b"<html>hi</html>")) as urlopen:
            result = handler(args)
        return output, result, urlopen

    def test_fetch_url_prompts_showing_the_url_and_a_decline_stops_it(self):
        output, result, urlopen = self._call(
            self.fetch, {"url": "https://example.com/"})
        self.assertEqual(len(output.confirmations), 1)
        self.assertTrue(any("https://example.com/" in line for line in output.lines))
        self.assertEqual(urlopen.call_count, 0)
        self.assertTrue(result["user_cancelled"])
        self.assertEqual(result["url"], "https://example.com/")

        # Confirmed or unsafe both fetch.
        for kwargs in ({"answer": True}, {"unsafe": True}):
            _output, result, urlopen = self._call(
                self.fetch, {"url": "https://example.com/"}, **kwargs)
            self.assertEqual(urlopen.call_count, 1, kwargs)
            self.assertIn("hi", result, kwargs)

    def test_web_search_prompts_declines_and_parses_on_a_confirm(self):
        output, result, urlopen = self._call(self.search, {"query": "wisemonkey cli"})
        self.assertEqual(len(output.confirmations), 1)
        self.assertTrue(any("wisemonkey cli" in line for line in output.lines))
        self.assertEqual(urlopen.call_count, 0)
        self.assertTrue(result["user_cancelled"])

        page = (
            '<a class="result__a" href="https://example.com/a">Example A</a>'
            '<a class="result__snippet">First snippet</a>'
            '<a class="result__a" href="https://example.com/b">Example B</a>'
            '<a class="result__snippet">Second snippet</a>'
        )
        _output, result, urlopen = self._call(
            self.search, {"query": "example"}, answer=True,
            response=_FakeResponse(page.encode()))
        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(result["query"], "example")
        self.assertEqual([r["title"] for r in result["results"]],
                         ["Example A", "Example B"])

    def test_a_missing_query_does_not_prompt(self):
        output, result, _urlopen = self._call(self.search, {})
        self.assertEqual(result, {"error": "No query provided"})
        self.assertEqual(output.confirmations, [])
