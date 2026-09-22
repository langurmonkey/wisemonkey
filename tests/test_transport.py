"""Tests for the Unix domain socket transport and server/client (phase 2)."""

import socket
import tempfile
import unittest
from pathlib import Path

from agent.ipc import (
    ClientRequest,
    Event,
    Message,
    PromptPayload,
    ProtocolError,
    ReplyPayload,
    ServerRequest,
    TransportClosed,
    UnixTransport,
    payload_as,
    socket_path,
)


class TestSocketPathReal(unittest.TestCase):
    def test_path_under_xdg_runtime_dir(self, monkeypatch=None):
        import os

        old = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["XDG_RUNTIME_DIR"] = tempfile.mkdtemp()
        try:
            p = socket_path("mysession")
            self.assertTrue(str(p).endswith("wisemonkey/mysession.sock"))
            self.assertEqual(p.parent.stat().st_mode & 0o777, 0o700)
        finally:
            if old is None:
                del os.environ["XDG_RUNTIME_DIR"]
            else:
                os.environ["XDG_RUNTIME_DIR"] = old

    def test_path_fallback_without_xdg(self):
        import os

        old = os.environ.pop("XDG_RUNTIME_DIR", None)
        try:
            p = socket_path("mysession")
            self.assertIn("wisemonkey-", str(p))
            self.assertTrue(str(p).endswith("mysession.sock"))
        finally:
            if old is not None:
                os.environ["XDG_RUNTIME_DIR"] = old


class TestUnixTransport(unittest.TestCase):
    """Round-trip tests over real AF_UNIX sockets."""

    def setUp(self):
        self.a, self.b = socket.socketpair(socket.AF_UNIX)
        self.ta = UnixTransport(self.a)
        self.tb = UnixTransport(self.b)

    def tearDown(self):
        self.ta.close()
        self.tb.close()

    def test_round_trip(self):
        self.tb.send(Message.request("prompt", PromptPayload(text="hello")))
        msg = self.ta.recv(timeout=1)
        assert msg is not None
        self.assertEqual(msg.name, "prompt")
        self.assertEqual(payload_as(PromptPayload, msg).text, "hello")

    def test_event_round_trip(self):
        self.ta.send(Message.event(Event.CONTENT, {"text": "stream"}))
        msg = self.tb.recv(timeout=1)
        assert msg is not None
        self.assertEqual(msg.name, Event.CONTENT)

    def test_recv_timeout_returns_none(self):
        self.assertIsNone(self.ta.recv(timeout=0.1))

    def test_multiple_messages_framed(self):
        for i in range(5):
            self.tb.send(Message.event(Event.CONTENT, {"text": str(i)}))
        msgs: list = []
        for _ in range(5):
            m = self.ta.recv(timeout=1)
            assert m is not None
            msgs.append(m)
        got = [payload_as(PromptPayload, m) for m in msgs]
        self.assertEqual([g.text for g in got], [str(i) for i in range(5)])

    def test_close_raises_on_recv(self):
        self.tb.close()
        with self.assertRaises(TransportClosed):
            self.ta.recv(timeout=1)

    def test_send_after_close(self):
        self.ta.close()
        with self.assertRaises(TransportClosed):
            self.ta.send(Message.event(Event.CONTENT))

    def test_peer_disconnect_raises(self):
        self.tb.close()
        # Give the peer a moment to see the EOF
        import time

        time.sleep(0.05)
        with self.assertRaises(TransportClosed):
            self.ta.recv(timeout=1)

    def test_reply_round_trip(self):
        req = Message.request("confirm", {"message": "ok?"})
        self.tb.send(req)
        got = self.ta.recv(timeout=1)
        assert got is not None
        self.ta.send(Message.response(got.id, ReplyPayload(value=True)))
        reply = self.tb.recv(timeout=1)
        assert reply is not None
        self.assertTrue(reply.is_reply)
        self.assertEqual(reply.reply_to, req.id)
        self.assertTrue(payload_as(ReplyPayload, reply).value)


if __name__ == "__main__":
    unittest.main()