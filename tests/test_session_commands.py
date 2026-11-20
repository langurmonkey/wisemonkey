"""Tests for `/sessions` and `/resume`."""

import datetime
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import agent.memory as memory_mod
from agent.commands import registry
from agent.ipc import ClientRequest, UnixTransport
from agent.memory import Memory
from tests.conftest import reset_singletons


def _run(raw: str, core):
    """Execute a command string through the real registry."""
    command, params = registry.lookup(raw.split())
    assert command is not None, f"command not found: {raw}"
    return command.handler(core, params)


class _SessionTestBase(unittest.TestCase):
    """Temp sessions root + a fake Core wired to a real Memory."""

    def setUp(self):
        reset_singletons()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="wisemonkey-session-"))
        self._sessions = self._tmpdir / "sessions"
        self._sessions.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self._tmpdir, True)
        self.addCleanup(reset_singletons)

        self._root = memory_mod.SESSIONS_DIR
        memory_mod.SESSIONS_DIR = self._sessions
        self.addCleanup(setattr, memory_mod, "SESSIONS_DIR", self._root)

        self.memory = Memory(session="default")
        self.core = SimpleNamespace(
            memory=self.memory,
            config=SimpleNamespace(get=lambda *a, **k: 80000),
        )

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

    def _row(self, content, name):
        return next(line for line in content.splitlines() if name in line)


class TestSessionsCommand(_SessionTestBase):
    def test_lists_newest_first_with_counts_a_current_marker_and_no_parameters(self):
        now = datetime.datetime.now()
        self._make_session("stale", accessed=now - datetime.timedelta(days=3))
        self._make_session("fresh", messages=12, accessed=now)
        ok, _msg, content, _md = _run("/sessions", self.core)
        assert ok
        assert content.index("fresh") < content.index("stale")
        assert "12" in self._row(content, "fresh")
        # Exactly one session is marked current, and it is this one.
        current = [line for line in content.splitlines() if "▸" in line]
        assert len(current) == 1 and "default" in current[0]
        # Not "None ago": naive timestamps need normalizing, and
        # `pretty_timedelta` returns None for a zero delta.
        for name in ("fresh", "stale"):
            assert "None" not in self._row(content, name), name
        assert "ago" in self._row(content, "stale")

        ok, msg, _c, _m = _run("/sessions nope", self.core)
        assert ok is False and "does not take any parameters" in msg


class TestResumeCommand(_SessionTestBase):
    def test_switches_to_a_named_session_and_loads_its_history(self):
        self._make_session("gaiasky", messages=3)
        reset_singletons()
        other = Memory(session="gaiasky")
        other._chat_history.set_exchanges([{"role": "user", "content": "from other"}])
        other._chat_history.save()

        ok, _msg, content, _md = _run("/resume gaiasky", self.core)
        assert ok and "gaiasky" in content
        assert self.memory.session == "gaiasky"
        assert self.memory.get_chat_history_unformatted() == [
            {"role": "user", "content": "from other"}
        ]

    def test_switching_keeps_the_singleton_and_flushes_the_old_session(self):
        """Core, the footer and the prompt all hold this object."""
        identity = id(self.memory)
        self.memory.add_note("buffered note")
        self._make_session("other")
        self.core.reset_session_state = lambda: setattr(self.core, "reset_called", True)
        with patch("pubsub.pub.sendMessage") as send:
            _run("/resume other", self.core)
        assert id(self.core.memory) == identity
        assert self.core.reset_called is True
        saved = json.loads((self._sessions / "default" / "notes.json").read_text())
        assert [n["content"] for n in saved] == ["buffered note"]
        # The prompt names the session, so it must be rebuilt.
        send.assert_called_once_with("prompt-update")

    def test_an_unknown_name_creates_it_and_the_current_one_is_a_noop(self):
        """`wmk <new>` creates sessions, so `/resume <new>` does too."""
        ok, _msg, content, _md = _run("/resume brand-new", self.core)
        assert ok and self.memory.session == "brand-new"
        assert (self._sessions / "brand-new").is_dir()
        assert "new session" in content
        ok, msg, _c, _m = _run("/resume brand-new", self.core)
        assert ok and "already in session" in msg

    def test_no_argument_resumes_the_most_recent_other_session_or_errors(self):
        ok, msg, _c, _m = _run("/resume", self.core)
        assert ok is False and "no other session" in msg
        now = datetime.datetime.now()
        self._make_session("older", accessed=now - datetime.timedelta(days=1))
        self._make_session("newer", accessed=now)
        # 'default' was accessed most recently of all, but it is current.
        assert _run("/resume", self.core)[0] is True
        assert self.memory.session == "newer"

    def test_a_traversing_name_or_extra_parameters_are_refused(self):
        for raw, fragment in (("/resume ../../etc", "invalid session name"),
                              ("/resume a b", "too many")):
            ok, msg, _c, _m = _run(raw, self.core)
            assert ok is False and fragment in msg, raw
        assert self.memory.session == "default"
        assert not (self._tmpdir.parent / "etc").exists()


class TestResumeRefusedInRemoteMode(unittest.TestCase):
    """The daemon owns its session, so `/resume` must refuse there."""

    def setUp(self):
        reset_singletons()
        self.addCleanup(reset_singletons)

    def _send(self, raw, core):
        from agent.ipc import CommandPayload, Message, loopback_pair
        from agent.server import WisemonkeyServer

        client, peer = loopback_pair()
        server = WisemonkeyServer.__new__(WisemonkeyServer)
        server.core = core
        # The server speaks over a UnixTransport in production; a loopback pair
        # exercises the same contract without a socket.
        server._client = cast(UnixTransport, client)
        server._handle_command(Message.request(
            ClientRequest.COMMAND,
            CommandPayload(raw=raw, name=raw.split()[0], params=raw.split()[1:]),
        ))
        reply = peer.recv(timeout=1.0)
        client.close()
        peer.close()
        assert reply is not None, "no reply sent"
        return reply.payload

    def test_resume_is_refused_while_other_commands_still_run(self):
        memory = Memory(session="a")
        payload = self._send("/resume other",
                             SimpleNamespace(memory=memory, pending_prompt=None))
        assert payload["ok"] is False
        assert "remote mode" in payload["msg"] and "wmk --server" in payload["msg"]
        assert memory.session == "a"

        core = SimpleNamespace(
            memory=Memory(session="a"),
            pending_prompt=None,
            config=SimpleNamespace(get=lambda *a, **k: 80000),
            get_context_breakdown=lambda: [("System", 10)],
        )
        payload = self._send("/session", core)
        assert payload["ok"] is True, payload["msg"]
        assert payload["command"] == "/session"
