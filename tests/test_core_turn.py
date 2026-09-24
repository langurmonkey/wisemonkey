"""Tests for the phase-1 core turn changes: TurnResult, cancellation state,
tool result reporting, and the emitter integration."""

import unittest

from typing import Any, cast
from unittest.mock import patch

from agent.core import Core, Stage, TurnCancelled, TurnResult
from agent.emitter import TurnEmitter
from agent.ipc import Event, ToolResultPayload, loopback_pair, payload_as


class _Config:
    """Minimal config stand-in."""

    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


class _Memory:
    """Records chat exchanges without touching the filesystem."""

    def __init__(self):
        self.exchanges = []
        self.saved = 0

    def add_chat_exchange(self, core, role, content, **extra):
        self.exchanges.append({"role": role, "content": content, **extra})

    def save(self):
        self.saved += 1

    def get_chat_history_unformatted(self):
        return list(self.exchanges)

    def get_chat_stats(self):
        return (0, 100, 0.0)


def make_core(llm_replies, config=None, memory=None):
    """Build a Core without running __init__ (no router, no MCP)."""
    core = cast(Any, Core.__new__(Core))
    core.config = config or _Config({"agent.max_turns": 5})
    core.memory = memory or _Memory()
    core._pending_image = None
    core._turn_cancelled = False
    core.response_buffer = ""
    core.thinking = False
    core.generating = False
    core._build_system_prompt = lambda: "system"
    replies = list(llm_replies)

    def _send_to_llm(prompt_callback=None, reasoning_callback=None,
                     content_callback=None, cancel_callback=None,
                     error_callback=None, poll=None):
        core.response_buffer = ""
        if poll and poll():
            # Mimic the real stream handler: cancellation is recorded as state.
            core._turn_cancelled = True
            if cancel_callback:
                cancel_callback(KeyboardInterrupt())
            return ({"text": "", "tool_calls": None}, 0, 0.0, None)
        reply = replies.pop(0)
        core.response_buffer = reply.get("text", "")
        if content_callback:
            content_callback(reply.get("text", ""))
        return (reply, reply.get("tokens", 1), 0.1, reply.get("stream_error"))

    core._send_to_llm = _send_to_llm
    return core


class TestTurnResult(unittest.TestCase):
    """TurnResult must stay tuple-compatible for existing callers."""

    def test_unpacks_as_four_tuple(self):
        response, tokens, tools, gen = TurnResult("hi", 3, 1, 0.5)
        assert (response, tokens, tools, gen) == ("hi", 3, 1, 0.5)

    def test_len_and_index(self):
        r = TurnResult("hi", 3, 1, 0.5)
        assert len(r) == 4
        assert r[0] == "hi"
        assert r[3] == 0.5

    def test_defaults(self):
        r = TurnResult()
        assert r.response == ""
        assert r.cancelled is False
        assert r.error == ""

    def test_cancelled_flag(self):
        r = TurnResult(response="[Cancelled]", cancelled=True)
        assert r.cancelled is True


class TestRunTurnReturnValues(unittest.TestCase):
    def test_final_response(self):
        core = make_core([{"text": "answer", "tokens": 4}])
        result = core.run_turn("hello")
        assert result.response == "answer"
        assert result.total_tokens == 4
        assert result.cancelled is False
        assert core.memory.saved == 1

    def test_unpacks_like_a_tuple(self):
        core = make_core([{"text": "answer"}])
        response, tokens, tools, gen = core.run_turn("hello")
        assert response == "answer"

    def test_max_turns_returns_turn_result(self):
        core = make_core(
            [{"text": "narrate", "tool_calls": [{"id": "1", "type": "function",
              "function": {"name": "t", "arguments": "{}"}}]}] * 6,
            config=_Config({"agent.max_turns": 2}),
        )
        with patch("agent.core.execute_tool", return_value="ok"):
            result = core.run_turn("go")
        assert "maximum number of turns" in result.response
        assert isinstance(result, TurnResult)

    def test_max_tool_calls_stops_execution(self):
        """Hitting the tool-call cap must skip execution and let the model answer."""
        tool_call = {"id": "1", "type": "function",
                     "function": {"name": "t", "arguments": "{}"}}
        replies = [
            {"text": "narrate", "tool_calls": [tool_call]},
            {"text": "narrate2", "tool_calls": [tool_call]},
            {"text": "final answer"},
        ]
        core = make_core(replies, config=_Config({"agent.max_tool_calls": 1}))
        with patch("agent.core.execute_tool", return_value="ok") as ex:
            result = core.run_turn("go")
        assert result.response == "final answer"
        assert result.n_tools == 1  # only the first batch executed
        ex.assert_called_once()
        # The limit note was injected into the conversation and history
        roles = [e["role"] for e in core.memory.exchanges]
        assert "user" in roles
        assert any(
            "Tool call limit reached" in e["content"]
            for e in core.memory.exchanges if e["role"] == "user"
        )

    def test_max_tool_calls_zero_is_unlimited(self):
        """Default (0) must not restrict tool calls."""
        tool_call = {"id": "1", "type": "function",
                     "function": {"name": "t", "arguments": "{}"}}
        replies = [
            {"text": "narrate", "tool_calls": [tool_call]},
            {"text": "answer"},
        ]
        core = make_core(replies, config=_Config({"agent.max_tool_calls": 0}))
        with patch("agent.core.execute_tool", return_value="ok") as ex:
            result = core.run_turn("go")
        assert result.response == "answer"
        assert result.n_tools == 1
        ex.assert_called_once()

    def test_max_tool_calls_partial_batch_blocked(self):
        """A batch that would exceed the cap is blocked entirely (no partial execution)."""
        tool_call = {"id": "1", "type": "function",
                     "function": {"name": "t", "arguments": "{}"}}
        replies = [
            {"text": "narrate", "tool_calls": [tool_call]},
            {"text": "narrate2", "tool_calls": [tool_call, tool_call]},
            {"text": "final"},
        ]
        core = make_core(replies, config=_Config({"agent.max_tool_calls": 2}))
        with patch("agent.core.execute_tool", return_value="ok") as ex:
            result = core.run_turn("go")
        assert result.response == "final"
        assert result.n_tools == 1  # second batch of 2 would exceed cap of 2
        ex.assert_called_once()


class TestCancellation(unittest.TestCase):
    """Poll-driven cancellation must not persist a partial answer."""

    def test_poll_cancels_turn(self):
        core = make_core([{"text": "never"}])
        emitter = TurnEmitter()
        emitter.cancel()
        result = core.run_turn("hello", poll=emitter.poll)
        assert result.cancelled is True

    def test_cancelled_turn_is_not_persisted_as_answer(self):
        core = make_core([{"text": "never"}])
        emitter = TurnEmitter()
        emitter.cancel()
        core.run_turn("hello", poll=emitter.poll)
        roles = [e["role"] for e in core.memory.exchanges]
        # The user prompt is recorded, but no assistant answer.
        assert "assistant" not in roles

    def test_cancel_flag_is_reset_between_turns(self):
        core = make_core([{"text": "first"}, {"text": "second"}])
        emitter = TurnEmitter()
        emitter.cancel()
        # The cancelled turn never consumes a reply.
        core.run_turn("one", poll=emitter.poll)
        emitter.reset()
        result = core.run_turn("two", poll=emitter.poll)
        assert result.cancelled is False
        assert result.response == "first"

    def test_raise_on_cancel_emitter_is_handled(self):
        """A raise_on_cancel emitter still yields a clean TurnResult."""
        core = make_core([{"text": "never"}])
        emitter = TurnEmitter(raise_on_cancel=True)
        emitter.cancel()
        result = core.run_turn("hello", cancel_callback=emitter.cancelled,
                               poll=emitter.poll)
        assert result.cancelled is True

    def test_cancel_callback_no_longer_required_for_cancellation(self):
        """The legacy cancel_callback hook is optional."""
        core = make_core([{"text": "never"}])
        emitter = TurnEmitter()
        emitter.cancel()
        result = core.run_turn("hello", poll=emitter.poll)
        assert result.cancelled is True
        assert result.response == "[Cancelled]"

    def test_stream_error_still_propagates(self):
        core = make_core([{"text": "partial", "stream_error": ValueError("bad sse")}])
        with self.assertRaises(ValueError):
            core.run_turn("hello")


class TestToolCallbacks(unittest.TestCase):
    def _core_with_one_tool(self):
        reply = {
            "text": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "x"}'}}],
        }
        return make_core([reply, {"text": "done"}])

    def test_tool_result_callback_receives_result(self):
        core = self._core_with_one_tool()
        seen = []
        with patch("agent.core.execute_tool", return_value={"content": "data"}):
            core.run_turn("go", tool_result_callback=lambda *a: seen.append(a))
        assert len(seen) == 1
        tool_id, name, content, is_error, duration = seen[0]
        assert tool_id == "c1"
        assert name == "read_file"
        assert "data" in content
        assert is_error is False
        assert duration >= 0

    def test_tool_result_callback_marks_errors(self):
        core = self._core_with_one_tool()
        seen = []
        with patch("agent.core.execute_tool", return_value={"error": "nope"}):
            core.run_turn("go", tool_result_callback=lambda *a: seen.append(a))
        assert seen[0][3] is True

    def test_tool_result_callback_handles_images(self):
        core = self._core_with_one_tool()
        seen = []
        image_result = {"text": "[image]", "image_base64": "QUJD", "mime_type": "image/png"}
        with patch("agent.core.execute_tool", return_value=image_result):
            core.run_turn("go", tool_result_callback=lambda *a: seen.append(a))
        tool_id, name, content, is_error, duration, b64, mime = seen[0]
        assert b64 == "QUJD"
        assert mime == "image/png"

    def test_tool_result_callback_optional(self):
        core = self._core_with_one_tool()
        with patch("agent.core.execute_tool", return_value="ok"):
            result = core.run_turn("go")
        assert result.response == "done"


class TestEmitterIntegration(unittest.TestCase):
    """The emitter can drive a whole turn through a loopback transport."""

    def test_full_turn_event_stream(self):
        core = make_core([{"text": "hello there"}])
        a, peer = loopback_pair()
        emitter = TurnEmitter(transport=a)

        emitter.turn_start("hi")
        result = core.run_turn(
            "hi",
            prompt_callback=emitter.prompt,
            content_callback=emitter.content,
            error_callback=emitter.error,
            poll=emitter.poll,
        )
        emitter.turn_end(result.response, result.total_tokens,
                         result.n_tools, result.gen_time, cancelled=result.cancelled)

        names = []
        while True:
            msg = peer.recv(timeout=0.05)
            if msg is None:
                break
            names.append(msg.name)

        assert names[0] == Event.TURN_START
        assert Event.CONTENT in names
        assert names[-1] == Event.TURN_END

    def test_emitter_error_callback_raises_through_run_turn(self):
        core = make_core([{"text": "x"}])
        emitter = TurnEmitter()

        def _send_to_llm(prompt_callback=None, reasoning_callback=None,
                         content_callback=None, cancel_callback=None,
                         error_callback=None, poll=None):
            assert error_callback is not None
            error_callback(ValueError("boom"), "endpoint down")
            raise AssertionError("unreachable")

        core._send_to_llm = _send_to_llm
        with self.assertRaises(RuntimeError):
            core.run_turn("hi", error_callback=emitter.error)
    def test_tool_events_end_to_end(self):
        reply = {
            "text": "",
            "tool_calls": [{"id": "c9", "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"}}],
        }
        core = make_core([reply, {"text": "final"}])
        a, peer = loopback_pair()
        emitter = TurnEmitter(transport=a)

        with patch("agent.core.execute_tool", return_value="filedata"):
            core.run_turn(
                "go",
                tool_callback=emitter.tool_call,
                tool_result_callback=emitter.tool_result,
            )

        events = []
        while True:
            msg = peer.recv(timeout=0.05)
            if msg is None:
                break
            events.append(msg)

        names = [e.name for e in events]
        assert Event.TOOL_CALL in names
        assert Event.TOOL_RESULT in names

        result_msg = next(e for e in events if e.name == Event.TOOL_RESULT)
        payload = payload_as(ToolResultPayload, result_msg)
        assert payload.id == "c9"
        assert payload.content == "filedata"


class TestStageEnumUnchanged(unittest.TestCase):
    """The core's Stage enum keeps its meaning for legacy callers."""

    def test_values(self):
        assert Stage.START.value == 0
        assert Stage.PROCESS.value == 1
        assert Stage.STOP.value == 2

    def test_turn_cancelled_is_an_exception(self):
        assert issubclass(TurnCancelled, Exception)


if __name__ == "__main__":
    unittest.main()
