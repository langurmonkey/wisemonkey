"""Tests for mid-turn injection on the server side.

The daemon owns the turn, so it owns the queue of steering lines: the client
enqueues them, the running turn drains them at a tool seam, and delivery is
confirmed with an ``injected`` event. The client's own copy is only dropped on
that event, which is what makes a line that never gets delivered safe.
"""

import threading
import unittest
from typing import Any

from agent.ipc import (
    ClientRequest,
    InjectPayload,
    InjectWhen,
    Message,
)
from agent.server import WisemonkeyServer


def _message(payload: InjectPayload) -> Any:
    msg = Message.request(ClientRequest.INJECT, payload)
    msg.id = "req-1"
    return msg


class _FakeServer(WisemonkeyServer):
    """Only the injection machinery, plus a recorded ``_send``."""

    def __init__(self) -> None:
        self._injections = []
        self._injection_lock = threading.Lock()
        self.sent: list[Message] = []

    def _send(self, message) -> None:  # type: ignore[override]
        self.sent.append(message)


def make_server() -> _FakeServer:
    return _FakeServer()


class TestInjectHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.server = make_server()

    def test_line_is_queued(self):
        self.server._handle_inject(_message(InjectPayload(
            text="use pandas", when=InjectWhen.AFTER_TOOL)))
        self.assertEqual(self.server._injections, ["use pandas"])

    def test_reply_is_ok(self):
        self.server._handle_inject(_message(InjectPayload(
            text="hello", when=InjectWhen.AFTER_TOOL)))
        reply = self.server.sent[-1]
        self.assertEqual(reply.kind, "response")
        self.assertEqual(reply.reply_to, "req-1")

    def test_several_lines_queue_in_order(self):
        for text in ("one", "two", "three"):
            self.server._handle_inject(_message(InjectPayload(
                text=text, when=InjectWhen.AFTER_TOOL)))
        self.assertEqual(self.server.sent[0].kind, "response")
        self.assertEqual(
            self.server._take_injections(), ["one", "two", "three"])

    def test_empty_text_is_rejected(self):
        self.server._handle_inject(_message(InjectPayload(
            text="   ", when=InjectWhen.AFTER_TOOL)))
        self.assertEqual(self.server._injections, [])
        self.assertEqual(self.server.sent[-1].kind, "error")

    def test_between_turns_is_refused(self):
        """The client runs such a line itself once the turn returns."""
        self.server._handle_inject(_message(InjectPayload(
            text="later", when=InjectWhen.BETWEEN_TURNS)))
        self.assertEqual(self.server._injections, [])
        self.assertEqual(self.server.sent[-1].kind, "error")
        self.assertIn("between_turns", self.server.sent[-1].payload["message"])

    def test_interrupt_is_refused(self):
        """Not implemented: cancelling the stream is a much larger UX change."""
        self.server._handle_inject(_message(InjectPayload(
            text="stop", when=InjectWhen.INTERRUPT)))
        self.assertEqual(self.server._injections, [])
        self.assertEqual(self.server.sent[-1].kind, "error")

    def test_unknown_timing_is_refused(self):
        self.server._handle_inject(_message(InjectPayload(
            text="soon", when="whenever")))
        self.assertEqual(self.server._injections, [])
        self.assertEqual(self.server.sent[-1].kind, "error")


class TestTakeInjections(unittest.TestCase):
    def setUp(self) -> None:
        self.server = make_server()

    def test_drains_everything(self):
        self.server._injections = ["a", "b"]
        self.assertEqual(self.server._take_injections(), ["a", "b"])
        self.assertEqual(self.server._injections, [])

    def test_empty_queue(self):
        self.assertEqual(self.server._take_injections(), [])

    def test_queue_is_emptied(self):
        """A second drain must not re-deliver the same lines."""
        self.server._injections = ["a"]
        self.assertEqual(self.server._take_injections(), ["a"])
        self.assertEqual(self.server._take_injections(), [])

    def test_concurrent_takes_do_not_lose_lines(self):
        """The client's reader thread and the turn thread touch this queue."""
        original = [f"line-{i}" for i in range(50)]
        self.server._injections = list(original)
        got: list[str] = []
        got_lock = threading.Lock()

        def drain() -> None:
            mine = self.server._take_injections()
            with got_lock:
                got.extend(mine)

        threads = [threading.Thread(target=drain) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(got), sorted(original))


if __name__ == "__main__":
    unittest.main()