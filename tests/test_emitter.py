"""Tests for agent/emitter.py — turn event emitter over the IPC protocol."""

import unittest

from agent.emitter import NullEmitter, TurnEmitter
from agent.ipc import (
    ContentPayload,
    ErrorPayload,
    Event,
    InjectedPayload,
    InjectWhen,
    ReasoningPayload,
    StageKind,
    StagePayload,
    ToolCallPayload,
    ToolResultPayload,
    TurnEndPayload,
    TurnStartPayload,
    loopback_pair,
    payload_as,
)


def _connected_emitter(**kwargs):
    """Build an emitter wired to one end of a loopback pair."""
    a, b = loopback_pair()
    return TurnEmitter(transport=a, **kwargs), b


def _recv(peer, cls, timeout=0.1):
    return payload_as(cls, peer.recv(timeout=timeout))


class TestCallbacks(unittest.TestCase):
    """Every core callback produces the matching event with the right payload."""

    def test_the_stage_accepts_enum_strings_and_carries_the_turn_id(self):
        from agent.core import Stage

        for given, expected in (
            (Stage.START, StageKind.START),
            ("stop", StageKind.STOP),
            ("PROCESS", StageKind.PROCESS),
        ):
            emitter, peer = _connected_emitter(turn_id="T1")
            emitter.prompt(given)
            msg = peer.recv(timeout=0.1)
            assert msg.is_event and msg.name == Event.STAGE and msg.turn_id == "T1"
            payload = payload_as(StagePayload, msg)
            assert (payload.name, payload.stage) == ("prompt", expected), given

    def test_reasoning_and_content(self):
        emitter, peer = _connected_emitter()
        emitter.reasoning("process", "aha", True)
        payload = _recv(peer, ReasoningPayload)
        assert (payload.text, payload.visible, payload.stage) == (
            "aha", True, StageKind.PROCESS
        )
        emitter.content("hi")
        assert _recv(peer, ContentPayload).text == "hi"
        # None becomes "", and the emitter's own visibility flag wins.
        emitter, peer = _connected_emitter(reasoning_visible=False)
        emitter.reasoning("start", None, True)
        payload = _recv(peer, ReasoningPayload)
        assert (payload.text, payload.visible) == ("", False)

    def test_tool_call_and_result(self):
        emitter, peer = _connected_emitter()
        emitter.tool_call("read_file", '{"path": "x"}')
        emitter.tool_call("next", {"a": 1})  # dict args are stringified
        first = _recv(peer, ToolCallPayload)
        second = _recv(peer, ToolCallPayload)
        assert (first.name, first.arguments, first.index) == (
            "read_file", '{"path": "x"}', 0
        )
        assert second.index == 1 and "a" in second.arguments

        emitter.tool_result("c1", "read_file", "data", duration=0.5)
        payload = _recv(peer, ToolResultPayload)
        assert (payload.id, payload.content, payload.duration) == ("c1", "data", 0.5)
        assert payload.is_error is False

        emitter.tool_result("c2", "screenshot", "[image]", is_error=True,
                            image_base64="QUJD", mime_type="image/png")
        payload = _recv(peer, ToolResultPayload)
        assert payload.is_error is True
        assert (payload.image_base64, payload.mime_type) == ("QUJD", "image/png")


class TestLifecycleEvents(unittest.TestCase):
    def test_turn_start_and_end_carry_the_outcome(self):
        emitter, peer = _connected_emitter()
        emitter.turn_start("do it")
        start = _recv(peer, TurnStartPayload)
        assert start.prompt == "do it" and start.turn_id == emitter.turn_id

        emitter.turn_end("done", total_tokens=10, n_tools=2, gen_time=1.5)
        end = _recv(peer, TurnEndPayload)
        assert (end.response, end.total_tokens, end.n_tools, end.gen_time) == (
            "done", 10, 2, 1.5
        )
        assert (end.cancelled, end.incomplete, end.incomplete_reason) == (
            False, False, ""
        )

        # Both flags travel: a remote frontend must see a fragment as one.
        emitter, peer = _connected_emitter()
        emitter.turn_end("", cancelled=True)
        assert _recv(peer, TurnEndPayload).cancelled is True
        emitter, peer = _connected_emitter()
        emitter.turn_end("half a sent", incomplete=True,
                         incomplete_reason="stream_ended")
        end = _recv(peer, TurnEndPayload)
        assert (end.incomplete, end.incomplete_reason) == (True, "stream_ended")

    def test_status_keeps_known_fields_and_drops_the_rest(self):
        emitter, peer = _connected_emitter()
        emitter.status(busy=True, memory_chars=100, nonsense=True)
        msg = peer.recv(timeout=0.1)
        assert msg.name == Event.STATUS
        assert msg.payload["busy"] is True and msg.payload["memory_chars"] == 100
        assert "nonsense" not in msg.payload


class TestCancellation(unittest.TestCase):
    """Cancellation is observable state, not a raised exception."""

    def test_cancel_is_polled_and_the_legacy_raise_fires_once(self):
        from agent.core import TurnCancelled

        emitter = TurnEmitter()
        assert emitter.poll() is False and emitter.is_cancelled is False
        emitter.cancel("timeout")
        assert emitter.poll() is True and emitter.cancel_reason == "timeout"

        # Without a transport it records state and does not raise.
        TurnEmitter().cancelled(KeyboardInterrupt())

        emitter, peer = _connected_emitter(raise_on_cancel=True)
        for _ in range(2):
            with self.assertRaises(TurnCancelled):
                emitter.cancelled(KeyboardInterrupt())
        msg = peer.recv(timeout=0.1)
        assert msg.name == Event.CANCELLED and msg.payload["reason"] == "user"
        assert peer.recv(timeout=0.1) is None

    def test_reset_allows_a_new_turn(self):
        emitter, peer = _connected_emitter(turn_id="old")
        emitter.cancelled(KeyboardInterrupt())
        peer.recv(timeout=0.1)
        emitter.reset()
        assert emitter.poll() is False and emitter.turn_id != "old"
        emitter.cancelled(KeyboardInterrupt())
        assert peer.recv(timeout=0.1).name == Event.CANCELLED
        # An explicit id is kept, which is how a retried turn reuses its own.
        emitter.reset("fixed")
        assert emitter.turn_id == "fixed"

    def test_error_emits_then_raises(self):
        emitter, peer = _connected_emitter()
        with self.assertRaises(RuntimeError):
            emitter.error(ValueError("boom"))
        payload = _recv(peer, ErrorPayload)
        assert payload.message == "boom" and payload.recoverable is False


class TestNoTransport(unittest.TestCase):
    def test_every_method_is_safe_without_a_transport_or_with_echo_off(self):
        emitter = TurnEmitter()
        emitter.turn_start("x")
        emitter.prompt("start")
        emitter.reasoning("process", "x", True)
        emitter.content("x")
        emitter.tool_call("t", "{}")
        emitter.tool_result("c", "t", "r")
        emitter.status(busy=True)
        emitter.injected("nobody is listening")
        emitter.turn_end("x")

        emitter, peer = _connected_emitter(echo=False)
        emitter.content("hidden")
        assert peer.recv(timeout=0.05) is None

        null = NullEmitter()
        null.content("ignored")
        null.cancel()
        assert null.poll() is True


class TestTurnScoping(unittest.TestCase):
    def test_events_carry_the_turn_id_and_generated_ids_are_unique(self):
        emitter, peer = _connected_emitter(turn_id="T1")
        emitter.content("a")
        emitter.tool_call("t", "{}")
        emitter.turn_end("a")
        for _ in range(3):
            assert peer.recv(timeout=0.1).turn_id == "T1"
        assert len({TurnEmitter().turn_id for _ in range(50)}) == 50
        assert TurnEmitter().elapsed() >= 0

    def test_injected_confirms_delivery_with_its_timing(self):
        """The client drops its own copy on this event."""
        emitter, peer = _connected_emitter()
        emitter.injected("use pandas", when=InjectWhen.AFTER_TOOL)
        payload = _recv(peer, InjectedPayload, timeout=1.0)
        assert (payload.text, payload.when, payload.role) == (
            "use pandas", InjectWhen.AFTER_TOOL, "user"
        )
        assert payload.turn_id == emitter.turn_id
        emitter.injected("later")
        assert _recv(peer, InjectedPayload, timeout=1.0).when == InjectWhen.BETWEEN_TURNS


if __name__ == "__main__":
    unittest.main()
