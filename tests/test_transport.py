"""Tests for the Unix domain socket transport and its socket path."""

import os
import socket
import tempfile
import time
import unittest

from agent.ipc import (
    Event,
    Message,
    PromptPayload,
    ReplyPayload,
    TransportClosed,
    UnixTransport,
    payload_as,
    socket_path,
)


class TestSocketPath(unittest.TestCase):
    def test_under_xdg_runtime_dir_privately(self):
        old = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["XDG_RUNTIME_DIR"] = tempfile.mkdtemp()
        try:
            path = socket_path("mysession")
            assert str(path).endswith("wisemonkey/mysession.sock")
            assert path.parent.stat().st_mode & 0o777 == 0o700
        finally:
            if old is not None:
                os.environ["XDG_RUNTIME_DIR"] = old
            else:
                del os.environ["XDG_RUNTIME_DIR"]

    def test_falls_back_without_xdg(self):
        old = os.environ.pop("XDG_RUNTIME_DIR", None)
        try:
            path = socket_path("mysession")
            assert "wisemonkey-" in str(path)
            assert str(path).endswith("mysession.sock")
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
        # A transport that saw the EOF marks itself closed and skips close(),
        # so drop the raw sockets too.
        self.a.close()
        self.b.close()

    def test_framing_round_trip_and_replies(self):
        for i in range(5):
            self.tb.send(Message.event(Event.CONTENT, {"text": str(i)}))
        received = [self.ta.recv(timeout=1) for _ in range(5)]
        assert all(msg is not None for msg in received)
        got = [payload_as(PromptPayload, msg) for msg in received if msg]
        assert [g.text for g in got] == [str(i) for i in range(5)]

        request = Message.request("prompt", PromptPayload(text="hello"))
        self.tb.send(request)
        arrived = self.ta.recv(timeout=1)
        assert arrived is not None
        self.ta.send(Message.response(arrived.id, ReplyPayload(value=True)))
        reply = self.tb.recv(timeout=1)
        assert reply is not None
        assert reply.is_reply and reply.reply_to == request.id
        assert payload_as(ReplyPayload, reply).value is True

    def test_a_closed_transport_refuses_io_and_wakes_the_peer(self):
        self.ta.close()
        with self.assertRaises(TransportClosed):
            self.ta.send(Message.event(Event.CONTENT))
        # The peer sees the EOF rather than blocking forever.
        time.sleep(0.05)
        with self.assertRaises(TransportClosed):
            self.tb.recv(timeout=1)
