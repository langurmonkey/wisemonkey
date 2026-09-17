"""Tests for agent/emitter.py — turn event emitter over the IPC protocol."""

import unittest

from agent.emitter import NullEmitter, TurnEmitter
from agent.ipc import (
    ContentPayload,
    ErrorPayload,
    Event,
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


class _Sink:
    """Collects (name, payload) pairs instead of emitting on a transport."""

    def __init__(self):
        self.events = []

    def emit_hook(self, message):
        self.events.append(message)

    def names(self):
        return [e.name for e in self.events]

    def last(self, name):
        for e in reversed(self.events):
            if e.name == name:
                return e
        return None


def _connected_emitter(**kwargs):
    """Build an emitter wired to one end of a loopback pair.

    Returns (emitter, peer) where *peer* receives everything the emitter emits.
    """
    a, b = loopback_pair()
    emitter = TurnEmitter(transport=a, **kwargs)
    return emitter, b


class TestStageNormalisation(unittest.TestCase):
    """The emitter must accept the core's Stage enum, strings, or anything."""

    def test_accepts_core_stage_enum(self):
        from agent.core import Stage

        emitter, peer = _connected_emitter()
        emitter.prompt(Stage.START)
        msg = peer.recv(timeout=0.1)
        payload = payload_as(StagePayload, msg)
        assert payload.name == "prompt"
        assert payload.stage == StageKind.START

    def test_accepts_plain_string(self):
        emitter, peer = _connected_emitter()
        emitter.prompt("stop")
        payload = payload_as(StagePayload, peer.recv(timeout=0.1))
        assert payload.stage == StageKind.STOP

    def test_accepts_uppercase_string(self):
        emitter, peer = _connected_emitter()
        emitter.prompt("PROCESS")
        payload = payload_as(StagePayload, peer.recv(timeout=0.1))
        assert payload.stage == StageKind.PROCESS


class TestCallbacks(unittest.TestCase):
    """Every core callback produces the matching event with the right payload."""

    def test_prompt(self):
        emitter, peer = _connected_emitter()
        emitter.prompt("start")
        msg = peer.recv(timeout=0.1)
        assert msg.is_event
        assert msg.name == Event.STAGE
        assert msg.turn_id == emitter.turn_id

    def test_reasoning_with_content(self):
        emitter, peer = _connected_emitter()
        emitter.reasoning("process", "aha", True)
        payload = payload_as(ReasoningPayload, peer.recv(timeout=0.1))
        assert payload.text == "aha"
        assert payload.visible is True

    def test_reasoning_none_content_becomes_empty(self):
        emitter, peer = _connected_emitter()
        emitter.reasoning("start", None, False)
        payload = payload_as(ReasoningPayload, peer.recv(timeout=0.1))
        assert payload.text == ""
        assert payload.visible is False
        assert payload.stage == StageKind.START

    def test_reasoning_stage_ride_along(self):
        emitter, peer = _connected_emitter()
        emitter.reasoning("stop", "", True)
        payload = payload_as(ReasoningPayload, peer.recv(timeout=0.1))
        assert payload.stage == StageKind.STOP

    def test_reasoning_visible_override(self):
        emitter, peer = _connected_emitter(reasoning_visible=False)
        emitter.reasoning("process", "x", True)
        payload = payload_as(ReasoningPayload, peer.recv(timeout=0.1))
        assert payload.visible is False

    def test_content(self):
        emitter, peer = _connected_emitter()
        emitter.content("hi")
        payload = payload_as(ContentPayload, peer.recv(timeout=0.1))
        assert payload.text == "hi"

    def test_tool_call(self):
        emitter, peer = _connected_emitter()
        emitter.tool_call("read_file", '{"path": "x"}')
        payload = payload_as(ToolCallPayload, peer.recv(timeout=0.1))
        assert payload.name == "read_file"
        assert payload.arguments == '{"path": "x"}'
        assert payload.index == 0

    def test_tool_call_index_increments(self):
        emitter, peer = _connected_emitter()
        emitter.tool_call("a", "{}")
        emitter.tool_call("b", "{}")
        first = payload_as(ToolCallPayload, peer.recv(timeout=0.1))
        second = payload_as(ToolCallPayload, peer.recv(timeout=0.1))
        assert first.index == 0
        assert second.index == 1

    def test_tool_call_dict_args_are_stringified(self):
        emitter, peer = _connected_emitter()
        emitter.tool_call("t", {"a": 1})
        payload = payload_as(ToolCallPayload, peer.recv(timeout=0.1))
        assert "a" in payload.arguments

    def test_tool_result(self):
        emitter, peer = _connected_emitter()
        emitter.tool_result("c1", "read_file", "data", duration=0.5)
        payload = payload_as(ToolResultPayload, peer.recv(timeout=0.1))
        assert payload.id == "c1"
        assert payload.name == "read_file"
        assert payload.content == "data"
        assert payload.duration == 0.5
        assert payload.is_error is False

    def test_tool_result_error(self):
        emitter, peer = _connected_emitter()
        emitter.tool_result("c1", "run_command", "boom", is_error=True)
        payload = payload_as(ToolResultPayload, peer.recv(timeout=0.1))
        assert payload.is_error is True

    def test_tool_result_image(self):
        emitter, peer = _connected_emitter()
        emitter.tool_result(
            "c2", "screenshot", "[image]", image_base64="QUJD", mime_type="image/png"
        )
        payload = payload_as(ToolResultPayload, peer.recv(timeout=0.1))
        assert payload.image_base64 == "QUJD"
        assert payload.mime_type == "image/png"


class TestLifecycleEvents(unittest.TestCase):
    def test_turn_start(self):
        emitter, peer = _connected_emitter()
        emitter.turn_start("do it")
        payload = payload_as(TurnStartPayload, peer.recv(timeout=0.1))
        assert payload.prompt == "do it"
        assert payload.turn_id == emitter.turn_id

    def test_turn_end_success(self):
        emitter, peer = _connected_emitter()
        emitter.turn_end("done", total_tokens=10, n_tools=2, gen_time=1.5)
        payload = payload_as(TurnEndPayload, peer.recv(timeout=0.1))
        assert payload.response == "done"
        assert payload.total_tokens == 10
        assert payload.n_tools == 2
        assert payload.gen_time == 1.5
        assert payload.cancelled is False

    def test_turn_end_cancelled(self):
        emitter, peer = _connected_emitter()
        emitter.turn_end("", cancelled=True)
        payload = payload_as(TurnEndPayload, peer.recv(timeout=0.1))
        assert payload.cancelled is True

    def test_status(self):
        emitter, peer = _connected_emitter()
        emitter.status(busy=True, memory_chars=100, memory_max=200)
        msg = peer.recv(timeout=0.1)
        assert msg.name == Event.STATUS
        assert msg.payload["busy"] is True
        assert msg.payload["memory_chars"] == 100

    def test_status_ignores_unknown_fields(self):
        emitter, peer = _connected_emitter()
        emitter.status(nonsense=True)
        msg = peer.recv(timeout=0.1)
        assert "nonsense" not in msg.payload


class TestCancellation(unittest.TestCase):
    """Cancellation is observable state, not a raised exception."""

    def test_poll_false_initially(self):
        emitter = TurnEmitter()
        assert emitter.poll() is False
        assert emitter.is_cancelled is False

    def test_cancel_sets_poll(self):
        emitter = TurnEmitter()
        emitter.cancel()
        assert emitter.poll() is True
        assert emitter.is_cancelled is True

    def test_cancel_records_reason(self):
        emitter = TurnEmitter()
        emitter.cancel("timeout")
        assert emitter.cancel_reason == "timeout"

    def test_cancelled_callback_does_not_raise_by_default(self):
        emitter = TurnEmitter()
        emitter.cancelled(KeyboardInterrupt())  # must not raise
        assert emitter.is_cancelled is True

    def test_cancelled_callback_emits_event(self):
        emitter, peer = _connected_emitter()
        emitter.cancelled(KeyboardInterrupt())
        msg = peer.recv(timeout=0.1)
        assert msg.name == Event.CANCELLED
        assert msg.payload["reason"] == "user"

    def test_cancelled_callback_raises_when_asked(self):
        from agent.core import TurnCancelled

        emitter = TurnEmitter(raise_on_cancel=True)
        with self.assertRaises(TurnCancelled):
            emitter.cancelled(KeyboardInterrupt())

    def test_reset_clears_cancellation(self):
        emitter = TurnEmitter()
        emitter.cancel()
        old_id = emitter.turn_id
        emitter.reset()
        assert emitter.poll() is False
        assert emitter.turn_id != old_id

    def test_reset_can_preserve_turn_id(self):
        emitter = TurnEmitter()
        emitter.reset("fixed")
        assert emitter.turn_id == "fixed"


class TestError(unittest.TestCase):
    def test_error_emits_and_raises(self):
        emitter, peer = _connected_emitter()
        with self.assertRaises(RuntimeError):
            emitter.error(ValueError("inner"), "outer")
        msg = peer.recv(timeout=0.1)
        assert msg.name == Event.ERROR
        payload = payload_as(ErrorPayload, msg)
        assert payload.message == "outer"
        assert payload.recoverable is False

    def test_error_without_message_uses_exception(self):
        emitter, peer = _connected_emitter()
        with self.assertRaises(RuntimeError):
            emitter.error(ValueError("boom"))
        payload = payload_as(ErrorPayload, peer.recv(timeout=0.1))
        assert payload.message == "boom"


class TestNoTransport(unittest.TestCase):
    def test_all_methods_are_safe_without_transport(self):
        emitter = TurnEmitter()
        emitter.turn_start("x")
        emitter.prompt("start")
        emitter.reasoning("process", "x", True)
        emitter.content("x")
        emitter.tool_call("t", "{}")
        emitter.tool_result("c", "t", "r")
        emitter.status(busy=True)
        emitter.turn_end("x")

    def test_echo_false_suppresses_emission(self):
        emitter, peer = _connected_emitter(echo=False)
        emitter.content("hidden")
        assert peer.recv(timeout=0.05) is None

    def test_null_emitter_discards_but_tracks_cancel(self):
        emitter = NullEmitter()
        emitter.content("ignored")
        emitter.cancel()
        assert emitter.poll() is True


class TestTurnScoping(unittest.TestCase):
    def test_events_carry_turn_id(self):
        emitter, peer = _connected_emitter(turn_id="T1")
        emitter.content("a")
        emitter.tool_call("t", "{}")
        emitter.turn_end("a")
        for _ in range(3):
            msg = peer.recv(timeout=0.1)
            assert msg.turn_id == "T1"

    def test_generated_turn_ids_are_unique(self):
        ids = {TurnEmitter().turn_id for _ in range(50)}
        assert len(ids) == 50

    def test_elapsed_is_non_negative(self):
        emitter = TurnEmitter()
        assert emitter.elapsed() >= 0


if __name__ == "__main__":
    unittest.main()
