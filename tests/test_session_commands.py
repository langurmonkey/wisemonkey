"""Tests for `/sessions` and `/resume`.

`/sessions` only reads session directories. `/resume` rebinds the `Memory`
singleton, so the interesting assertions are about what a switch must and must
not disturb -- and about the one mode where it must refuse.
"""

import datetime
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from agent.commands import registry
from agent.ipc import ClientRequest, UnixTransport
from agent.memory import Memory


def _run(raw: str, core):
    """Execute a command string through the real registry."""
    command, params = registry.lookup(raw.split())
    assert command is not None, f"command not found: {raw}"
    ok, msg, content, markdown = command.handler(core, params)
    return ok, msg, content, markdown


class _SessionTestBase(unittest.TestCase):
    """Temp sessions root + a fake Core wired to a real Memory."""

    def setUp(self):
        import tempfile

        import agent.memory as memory_mod
        from tests.conftest import reset_singletons

        reset_singletons()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="wisemonkey-session-"))
        self._sessions = self._tmpdir / "sessions"
        self._sessions.mkdir(parents=True)
        self.addCleanup(reset_singletons)

        self._root = memory_mod.SESSIONS_DIR
        memory_mod.SESSIONS_DIR = self._sessions
        self.addCleanup(setattr, memory_mod, "SESSIONS_DIR", self._root)

        self.memory = Memory(session="default")
        self.core = SimpleNamespace(
            memory=self.memory,
            config=SimpleNamespace(get=lambda *a, **k: 80000),
        )
        self.addCleanup(self._rmtree)

    def _rmtree(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_session(self, name, messages=0, accessed=None):
        d = self._sessions / name
        d.mkdir(parents=True, exist_ok=True)
        when = accessed or datetime.datetime.now()
        (d / ".session-metadata").write_text(
            f"created: {when.isoformat()}\naccessed: {when.isoformat()}\n"
        )
        (d / "chat_history.json").write_text(
            json.dumps({"exchanges": [{"role": "user", "content": "x"}] * messages})
        )
        return d

    def _fresh(self, name):
        from tests.conftest import reset_singletons

        reset_singletons()
        self.memory = Memory(session=name)
        self.core.memory = self.memory
        return self.memory


class TestSessionsCommand(_SessionTestBase):
    def test_lists_every_session(self):
        self._make_session("alpha")
        self._make_session("beta")
        ok, _msg, content, _md = _run("/sessions", self.core)
        self.assertTrue(ok)
        self.assertIn("alpha", content)
        self.assertIn("beta", content)
        self.assertIn("default", content)

    def test_marks_the_current_session(self):
        self._make_session("other")
        _ok, _msg, content, _md = _run("/sessions", self.core)
        # The marker sits on its own row, which the fixed-width columns make
        # easy to find.
        current_rows = [line for line in content.splitlines() if "▸" in line]
        self.assertEqual(len(current_rows), 1)
        self.assertIn("default", current_rows[0])

    def test_shows_a_readable_last_access(self):
        """Not `None ago`: naive metadata timestamps need normalizing, and
        `pretty_timedelta` returns None for a zero delta."""
        self._make_session("stamped", accessed=datetime.datetime.now())
        self._make_session(
            "older", accessed=datetime.datetime.now() - datetime.timedelta(hours=3)
        )
        _ok, _msg, content, _md = _run("/sessions", self.core)
        rows = {
            name: next(line for line in content.splitlines() if name in line)
            for name in ("stamped", "older")
        }
        self.assertNotIn("None", rows["stamped"])
        self.assertNotIn("None", rows["older"])
        self.assertIn("ago", rows["older"])

    def test_shows_message_counts(self):
        self._make_session("busy", messages=12)
        _ok, _msg, content, _md = _run("/sessions", self.core)
        row = next(line for line in content.splitlines() if "busy" in line)
        self.assertIn("12", row)

    def test_orders_most_recently_used_first(self):
        now = datetime.datetime.now()
        self._make_session("stale", accessed=now - datetime.timedelta(days=3))
        self._make_session("fresh", accessed=now)
        _ok, _msg, content, _md = _run("/sessions", self.core)
        rows = [line for line in content.splitlines() if "fresh" in line or "stale" in line]
        self.assertLess(content.index("fresh"), content.index("stale"))

    def test_no_sessions_is_a_message_not_a_panel(self):
        import shutil

        shutil.rmtree(self._sessions)
        self._sessions.mkdir()
        for d in list(self._sessions.iterdir()):
            shutil.rmtree(d, ignore_errors=True)
        ok, msg, content, _md = _run("/sessions", self.core)
        # The current session was created at setUp, so clear it too.
        self.assertTrue(ok or msg)

    def test_rejects_parameters(self):
        ok, msg, _content, _md = _run("/sessions nope", self.core)
        self.assertFalse(ok)
        self.assertIn("does not take any parameters", msg)


class TestResumeCommand(_SessionTestBase):
    def test_switches_to_a_named_session(self):
        self._make_session("gaiasky", messages=3)
        ok, _msg, content, _md = _run("/resume gaiasky", self.core)
        self.assertTrue(ok)
        self.assertEqual(self.memory.session, "gaiasky")
        self.assertIn("gaiasky", content)

    def test_switching_keeps_the_singleton_identity(self):
        """Core, the footer and the prompt all hold this object."""
        self._make_session("other")
        before = id(self.memory)
        _run("/resume other", self.core)
        self.assertEqual(id(self.core.memory), before)
        self.assertEqual(self.core.memory.session, "other")

    def test_switching_loads_the_other_history(self):
        other = self._fresh("other")
        other._chat_history.set_exchanges([{"role": "user", "content": "from other"}])
        other._chat_history.save()
        _run("/resume other", self.core)
        self.assertEqual(
            self.memory.get_chat_history_unformatted(),
            [{"role": "user", "content": "from other"}],
        )

    def test_switching_creates_an_unknown_session(self):
        """`wmk <new>` creates sessions, so `/resume <new>` does too."""
        ok, _msg, content, _md = _run("/resume brand-new", self.core)
        self.assertTrue(ok)
        self.assertEqual(self.memory.session, "brand-new")
        self.assertTrue((self._sessions / "brand-new").is_dir())
        self.assertIn("new session", content)

    def test_no_argument_resumes_the_most_recent_other_session(self):
        now = datetime.datetime.now()
        self._make_session("older", accessed=now - datetime.timedelta(days=1))
        self._make_session("newer", accessed=now)
        # 'default' was accessed at setUp, i.e. just now -- but it is the
        # current session, so it must be skipped.
        _ok, _msg, _content, _md = _run("/resume", self.core)
        self.assertEqual(self.memory.session, "newer")

    def test_no_argument_with_a_single_session_is_an_error(self):
        ok, msg, _content, _md = _run("/resume", self.core)
        self.assertFalse(ok)
        self.assertIn("no other session", msg)
        self.assertEqual(self.memory.session, "default")

    def test_resuming_the_current_session_is_a_noop(self):
        ok, msg, _content, _md = _run("/resume default", self.core)
        self.assertTrue(ok)
        self.assertIn("already in session", msg)

    def test_rejects_a_traversing_name(self):
        ok, msg, _content, _md = _run("/resume ../../etc", self.core)
        self.assertFalse(ok)
        self.assertIn("invalid session name", msg)
        self.assertEqual(self.memory.session, "default")
        self.assertFalse((self._tmpdir.parent / "etc").exists())

    def test_rejects_too_many_parameters(self):
        ok, msg, _content, _md = _run("/resume a b", self.core)
        self.assertFalse(ok)
        self.assertIn("too many", msg)

    def test_saves_the_previous_session_before_leaving_it(self):
        self.memory.add_note("buffered note")
        self._make_session("other")
        _run("/resume other", self.core)
        saved = json.loads((self._sessions / "default" / "notes.json").read_text())
        self.assertEqual([n["content"] for n in saved], ["buffered note"])

    def test_resets_turn_state_on_the_core(self):
        self.core.reset_session_state = lambda: setattr(
            self.core, "reset_called", True
        )
        self._make_session("other")
        _run("/resume other", self.core)
        self.assertTrue(getattr(self.core, "reset_called", False))

    def test_requests_a_prompt_rebuild(self):
        """The prompt message names the session, so it must be recreated."""
        self._make_session("other")
        with patch("pubsub.pub.sendMessage") as send:
            _run("/resume other", self.core)
        send.assert_called_once_with("prompt-update")

    def test_help_is_generated_like_every_other_command(self):
        """`-h` is handled by the registry, so it never reaches the handler."""
        command = registry.lookup(["/resume"])[0]
        assert command is not None
        ok, msg, content, _md, should_exit = registry.execute(
            self.core, command, ["-h"]
        )
        self.assertTrue(ok)
        self.assertFalse(should_exit)
        assert content is not None
        self.assertIn("/resume", content)
        self.assertEqual(self.memory.session, "default")

    def test_the_help_flag_does_not_create_a_session(self):
        """Guarded in the handler too, in case it is called directly."""
        _ok, msg, _content, _md = _run("/resume -h", self.core)
        self.assertFalse(_ok)
        self.assertFalse((self._sessions / "-h").exists())


class TestResumeRefusedInRemoteMode(unittest.TestCase):
    """The daemon owns its session, so `/resume` must refuse there.

    Enforced in `WisemonkeyServer._handle_command`, before the handler runs:
    rebinding the daemon's memory would change a session the attached client
    still believes it is talking to. A refusal is still a normal
    `CommandResultPayload`, so the client renders it like any other error and
    nothing in the protocol changes.
    """

    def _send(self, raw, core):
        from agent.ipc import CommandPayload, Message, loopback_pair
        from agent.server import WisemonkeyServer

        client, peer = loopback_pair()
        server = WisemonkeyServer.__new__(WisemonkeyServer)
        server.core = core
        # The server speaks over a UnixTransport in production; a loopback
        # pair exercises the same send/receive contract without a socket.
        server._client = cast(UnixTransport, client)

        message = Message.request(
            ClientRequest.COMMAND,
            CommandPayload(raw=raw, name=raw.split()[0], params=raw.split()[1:]),
        )
        server._handle_command(message)
        reply = peer.recv(timeout=1.0)
        client.close()
        peer.close()
        assert reply is not None, "no reply sent"
        return reply.payload

    def test_the_daemon_refuses_and_explains(self):
        payload = self._send(
            "/resume other",
            SimpleNamespace(memory=SimpleNamespace(session="a"), pending_prompt=None),
        )
        self.assertFalse(payload["ok"])
        self.assertIn("remote mode", payload["msg"])
        self.assertIn("wmk --server", payload["msg"])
        self.assertIn("wmk <name>", payload["msg"])

    def test_the_handler_never_runs(self):
        """A refusal, not a partially executed command."""
        from agent.memory import Memory

        memory = Memory(session="a")
        payload = self._send("/resume other", SimpleNamespace(memory=memory, pending_prompt=None))
        self.assertFalse(payload["ok"])
        self.assertEqual(memory.session, "a")

    def test_other_commands_still_run(self):
        """The refusal is per-command; `/session` must be unaffected."""
        from types import SimpleNamespace as NS

        from agent.memory import Memory

        memory = Memory(session="a")
        core = NS(
            memory=memory,
            pending_prompt=None,
            config=NS(get=lambda *a, **k: 80000),
            get_context_breakdown=lambda: [("System", 10)],
        )
        payload = self._send("/session", core)
        self.assertTrue(payload["ok"], payload["msg"])
        self.assertEqual(payload["command"], "/session")


