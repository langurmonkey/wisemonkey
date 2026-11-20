"""Tests for mid-turn injection on the server side."""

import threading
import unittest
from typing import Any

from agent.ipc import ClientRequest, InjectPayload, InjectWhen, Message
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


class TestInjectHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _FakeServer()

    def _inject(self, text, when=InjectWhen.AFTER_TOOL):
        self.server._handle_inject(_message(InjectPayload(text=text, when=when)))
        return self.server.sent[-1]

    def test_an_after_tool_line_is_queued_and_acknowledged_in_order(self):
        for text in ("one", "two"):
            reply = self._inject(text)
            assert reply.kind == "response" and reply.reply_to == "req-1"
        assert self.server._take_injections() == ["one", "two"]

    def test_every_other_timing_is_refused(self):
        """The client runs a between-turns line itself; interrupt is unbuilt."""
        for text, when in (("   ", InjectWhen.AFTER_TOOL),
                           ("later", InjectWhen.BETWEEN_TURNS),
                           ("stop", InjectWhen.INTERRUPT),
                           ("soon", "whenever")):
            assert self._inject(text, when).kind == "error", (text, when)
            assert self.server._injections == [], (text, when)
        assert "between_turns" in self._inject(
            "later", InjectWhen.BETWEEN_TURNS
        ).payload["message"]


class TestTakeInjections(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _FakeServer()

    def test_a_drain_is_complete_under_concurrency_and_never_repeats(self):
        expected = [f"line-{i}" for i in range(50)]
        self.server._injections = list(expected)
        got: list[str] = []
        lock = threading.Lock()

        def drain() -> None:
            mine = self.server._take_injections()
            with lock:
                got.extend(mine)

        threads = [threading.Thread(target=drain) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(got) == sorted(expected)
        assert self.server._take_injections() == []


if __name__ == "__main__":
    unittest.main()
