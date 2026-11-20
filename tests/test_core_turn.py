"""Tests for the core turn: TurnResult, cancellation, tool reporting, injection."""

import json
import unittest
from typing import Any, cast
from unittest.mock import patch

from agent.core import Core, TurnResult
from agent.emitter import TurnEmitter
from agent.ipc import Event, ToolResultPayload, loopback_pair, payload_as


class _Config:
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
            return ({"text": "", "tool_calls": None}, 0, 0.0, None, 0)
        reply = replies.pop(0)
        core.response_buffer = reply.get("text", "")
        if content_callback:
            content_callback(reply.get("text", ""))
        return (reply, reply.get("tokens", 1), 0.1, reply.get("stream_error"),
                reply.get("prompt_tokens", 0))

    core._send_to_llm = _send_to_llm
    return core


def _tool_call(call_id="1", name="t", args="{}"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args}}


class TestTurnResult(unittest.TestCase):
    def test_tuple_compatibility_and_defaults(self):
        result = TurnResult("hi", 3, 1, 0.5)
        assert (result.response, result.total_tokens, result.n_tools, result.gen_time) == (
            "hi", 3, 1, 0.5
        )
        assert len(result) == 4 and result[0] == "hi" and result[3] == 0.5
        empty = TurnResult()
        assert empty.response == "" and empty.cancelled is False and empty.error == ""
        assert TurnResult(cancelled=True).cancelled is True


class TestRunTurn(unittest.TestCase):
    def test_final_response(self):
        core = make_core([{"text": "answer", "tokens": 4}])
        result = core.run_turn("hello")
        assert result.response == "answer" and result.total_tokens == 4
        assert result.cancelled is False and core.memory.saved == 1

    def test_max_turns_stops_the_loop(self):
        core = make_core(
            [{"text": "narrate", "tool_calls": [_tool_call()]}] * 6,
            config=_Config({"agent.max_turns": 2}),
        )
        with patch("agent.core.execute_tool", return_value="ok"):
            assert "maximum number of turns" in core.run_turn("go").response

    def test_the_tool_cap_blocks_a_batch_rather_than_splitting_it(self):
        """Hitting the cap skips the whole batch and lets the model answer."""
        single = [{"text": "narrate", "tool_calls": [_tool_call()]}] * 2
        for cap, batch, expected in (
            (1, None, 1),
            (2, [_tool_call(), _tool_call()], 1),  # 1 + 2 > 2: no partial run
            (0, None, 2),                          # zero is unlimited
        ):
            replies = list(single[:1])
            replies += [{"text": "narrate2", "tool_calls": batch or [_tool_call()]}]
            replies.append({"text": "final answer"})
            core = make_core(replies, config=_Config({"agent.max_tool_calls": cap}))
            with patch("agent.core.execute_tool", return_value="ok") as execute:
                result = core.run_turn("go")
            assert result.response == "final answer", cap
            assert result.n_tools == expected, cap
            assert execute.call_count == expected, cap
            # The limit note reaches the conversation and the history.
            noted = any(
                "Tool call limit reached" in e["content"]
                for e in core.memory.exchanges if e["role"] == "user"
            )
            assert noted is bool(cap), cap


class TestCancellation(unittest.TestCase):
    """Poll-driven cancellation must not persist a partial answer."""

    def test_a_cancelled_turn_is_flagged_and_persists_no_answer(self):
        for raise_on_cancel in (False, True):
            core = make_core([{"text": "never"}])
            emitter = TurnEmitter(raise_on_cancel=raise_on_cancel)
            emitter.cancel()
            result = core.run_turn(
                "hello", cancel_callback=emitter.cancelled, poll=emitter.poll
            )
            assert result.cancelled is True
            assert result.response == "[Cancelled]"
            assert "assistant" not in [e["role"] for e in core.memory.exchanges]

    def test_the_cancel_flag_does_not_leak_into_the_next_turn(self):
        core = make_core([{"text": "first"}, {"text": "second"}])
        emitter = TurnEmitter()
        emitter.cancel()
        core.run_turn("one", poll=emitter.poll)
        emitter.reset()
        result = core.run_turn("two", poll=emitter.poll)
        assert result.cancelled is False and result.response == "first"

    def test_errors_and_a_finished_batch(self):
        # A stream error propagates on its own channel.
        core = make_core([{"text": "partial", "stream_error": ValueError("bad sse")}])
        with self.assertRaises(ValueError):
            core.run_turn("hello")

        # Ctrl+C while a tool runs cancels the turn, and persists nothing:
        # a dangling tool result would be malformed in the prompt.
        core = make_core([{"text": "narrate", "tool_calls": [_tool_call(name="slow")]}])
        with patch("agent.core.execute_tool", side_effect=KeyboardInterrupt):
            assert core.run_turn("go").cancelled is True
        assert core.memory.saved == 0

        core = make_core([{"text": "narrate", "tool_calls": [_tool_call()]},
                          {"text": "answer"}])
        with patch("agent.core.execute_tool", return_value="fine"):
            core.run_turn("go")
        assert core.memory.saved == 1


class TestCancellationIsAnnounced(unittest.TestCase):
    def test_both_cancellation_paths_announce_once(self):
        for via_poll in (True, False):
            emitter = TurnEmitter()
            seen = []
            core = make_core([{"text": "narrate", "tool_calls": [_tool_call(name="slow")]}])
            with patch.object(emitter, "emit",
                              side_effect=lambda name, payload=None: seen.append(name)):
                if via_poll:
                    emitter.cancel()
                    result = core.run_turn("go", cancel_callback=emitter.cancelled,
                                           poll=emitter.poll)
                else:
                    with patch("agent.core.execute_tool", side_effect=KeyboardInterrupt):
                        result = core.run_turn("go", cancel_callback=emitter.cancelled)
            assert result.cancelled is True
            assert seen.count(Event.CANCELLED) == 1


class TestToolCallbacks(unittest.TestCase):
    def test_the_callback_gets_the_result_the_error_flag_and_any_image(self):
        for result, is_error in (({"content": "data"}, False), ({"error": "nope"}, True)):
            core = make_core([
                {"text": "", "tool_calls": [_tool_call("c1", "read_file", '{"path": "x"}')]},
                {"text": "done"},
            ])
            seen = []
            with patch("agent.core.execute_tool", return_value=result):
                core.run_turn("go", tool_result_callback=lambda *a: seen.append(a))
            tool_id, name, content, err, duration = seen[0]
            assert (tool_id, name, err) == ("c1", "read_file", is_error)
            assert ("data" in content) is not is_error
            assert duration >= 0

        core = make_core([
            {"text": "", "tool_calls": [_tool_call("c1", "read_file", '{"path": "x"}')]},
            {"text": "done"},
        ])
        seen = []
        image = {"text": "[image]", "image_base64": "QUJD", "mime_type": "image/png"}
        with patch("agent.core.execute_tool", return_value=image):
            core.run_turn("go", tool_result_callback=lambda *a: seen.append(a))
        assert seen[0][5:] == ("QUJD", "image/png")


class TestMalformedToolArguments(unittest.TestCase):
    """A model can emit a truncated argument string (a bare ``{`` is common)."""

    def test_a_parse_error_is_a_tool_error_not_a_dead_turn(self):
        core = make_core([
            {"text": "", "tool_calls": [_tool_call("c1", "run_command", "{")]},
            {"text": "done"},
        ])
        calls = []
        seen = []
        with patch("agent.core.execute_tool",
                   side_effect=lambda name, args: calls.append(args) or "ok"):
            result = core.run_turn(
                "go", tool_result_callback=lambda *a: seen.append(a)
            )
        assert result.response == "done"
        assert result.error == "" and result.cancelled is False
        assert calls == []  # the handler never ran
        results = [e for e in core.memory.exchanges if e["role"] == "tool_result"]
        assert "not valid JSON" in results[0]["content"]
        assert results[0]["tool_call_id"] == "c1"
        # The call itself is still recorded, so history stays well-formed.
        assert len([e for e in core.memory.exchanges if e["role"] == "tool_call"]) == 1
        assert (seen[0][0], seen[0][1], seen[0][3]) == ("c1", "run_command", True)


class TestReasoningContentRoundTrip(unittest.TestCase):
    """DeepSeek's thinking mode rejects a tool-call message with no
    ``reasoning_content``, so the round's reasoning must ride back on it."""

    def _core(self, *thinkings):
        replies = [
            {"text": "", "thinking": thinking,
             "tool_calls": [_tool_call("c1", "read_file")]}
            for thinking in thinkings
        ]
        replies.append({"text": "done"})
        core = make_core(replies)
        fake_send = core._send_to_llm
        self.sent = []

        def _send(*a, **k):
            # Mirror _stream_handler: reset, then accumulate the round's reasoning.
            core.thinking_buffer = ""
            self.sent.append([dict(m) for m in core.messages])
            result = fake_send(*a, **k)
            core.thinking_buffer = result[0].get("thinking", "")
            if core.thinking_buffer:
                core._thinking_seen = True
            return result

        core._send_to_llm = _send
        return core

    @staticmethod
    def _assistants(messages):
        return [m for m in messages if m.get("role") == "assistant"]

    def test_the_reasoning_rides_back_and_otherwise_stays_out(self):
        core = self._core("I should read the file first.")
        with patch("agent.core.execute_tool", return_value="ok"):
            core.run_turn("go")
        assert self._assistants(core.messages)[0]["reasoning_content"] == (
            "I should read the file first."
        )
        # A provider without thinking mode gets no field at all.
        core = self._core("")
        with patch("agent.core.execute_tool", return_value="ok"):
            core.run_turn("go")
        for request in self.sent:
            for msg in self._assistants(request):
                assert "reasoning_content" not in msg, msg

    def test_every_assistant_message_carries_the_field_without_overwriting(self):
        core = self._core("keep me", "", "second thoughts")
        with patch("agent.core.execute_tool", return_value="ok"):
            core.run_turn("go")
        # Round 3 reasons, so round 1's already-shaped message is patched too,
        # and a real value is never replaced by the empty one of round 2.
        assert [m.get("reasoning_content") for m in self._assistants(self.sent[3])] == [
            "keep me", "", "second thoughts",
        ]


class TestEmitterIntegration(unittest.TestCase):
    @staticmethod
    def _drain(peer):
        messages = []
        while (msg := peer.recv(timeout=0.05)) is not None:
            messages.append(msg)
        return messages

    def test_a_whole_turn_streams_its_events_in_order(self):
        core = make_core([
            {"text": "", "tool_calls": [_tool_call("c9", "read_file")]},
            {"text": "final"},
        ])
        a, peer = loopback_pair()
        emitter = TurnEmitter(transport=a)

        emitter.turn_start("go")
        with patch("agent.core.execute_tool", return_value="filedata"):
            result = core.run_turn(
                "go", prompt_callback=emitter.prompt,
                tool_callback=emitter.tool_call,
                tool_result_callback=emitter.tool_result, poll=emitter.poll,
            )
        emitter.turn_end(result.response, result.total_tokens,
                         result.n_tools, result.gen_time, cancelled=result.cancelled)

        events = self._drain(peer)
        names = [e.name for e in events]
        assert names[0] == Event.TURN_START and names[-1] == Event.TURN_END
        assert Event.TOOL_CALL in names and Event.TOOL_RESULT in names
        payload = payload_as(
            ToolResultPayload, next(e for e in events if e.name == Event.TOOL_RESULT)
        )
        assert (payload.id, payload.content) == ("c9", "filedata")


class TestSteerInjection(unittest.TestCase):
    """Lines submitted mid-turn reach the model at the next tool seam."""

    @staticmethod
    def _replies(n_tools=2, final="done"):
        out = [{"text": f"narration {i}", "tool_calls": [_tool_call()]}
               for i in range(n_tools)]
        out.append({"text": final})
        return out

    @staticmethod
    def _users(core):
        return [m["content"] for m in core.messages if m["role"] == "user"]

    def test_lines_reach_the_model_and_the_history(self):
        for pending, expected in (
            ([[]], ["go"]),
            ([[None]], ["go"]),
            ([["just a line"]], ["go", "just a line"]),
            ([["first", "second"]], ["go", "first", "second"]),
        ):
            core = make_core(self._replies())
            with patch("agent.core.execute_tool", return_value="ok"):
                core.run_turn("go", inject_callback=lambda: pending.pop(0))
            assert self._users(core) == expected, pending
            # Every delivered line is stored as a user message, like a typed one.
            stored = [e["content"] for e in core.memory.exchanges
                      if e["role"] == "user"]
            assert stored == expected, pending

    def test_the_seam_rules(self):
        # Polled once per tool round, and never before a final answer request.
        core = make_core(self._replies(n_tools=3))
        calls = []
        with patch("agent.core.execute_tool", return_value="ok"):
            core.run_turn("go", inject_callback=lambda: calls.append(1) or [])
        assert len(calls) == 3
        calls.clear()
        make_core([{"text": "answer"}]).run_turn(
            "go", inject_callback=lambda: calls.append(1) or []
        )
        assert calls == []

        # Blank and non-string lines are dropped.
        core = make_core(self._replies())
        pending = [["", "  ", None, 42, "keep"]]
        with patch("agent.core.execute_tool", return_value="ok"):
            core.run_turn("go", inject_callback=lambda: pending.pop(0))
        assert self._users(core) == ["go", "keep"]

        # A UI failure to collect typed text must not kill a working turn.
        def inject():
            raise RuntimeError("footer exploded")

        core = make_core(self._replies())
        with patch("agent.core.execute_tool", return_value="ok"):
            assert core.run_turn("go", inject_callback=inject).response == "done"


class TestImageToolResults(unittest.TestCase):
    """Images from screenshot/read_image must reach the model as image blocks."""

    def _core(self):
        return make_core([
            {"text": "", "tool_calls": [_tool_call("c1", "fake_img")]},
            {"text": "done"},
        ])

    def test_an_image_becomes_a_block_an_error_is_flagged_and_a_string_stays(self):
        core = self._core()
        payload = {"image_base64": "QUJD", "mime_type": "image/jpeg"}
        with patch("agent.core.execute_tool", return_value=json.dumps(payload)):
            core.run_turn("look")
        content = [m for m in core.messages if m["role"] == "tool"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"

        seen = []
        with patch("agent.core.execute_tool", return_value=json.dumps({"error": "boom"})):
            self._core().run_turn("go", tool_result_callback=lambda *a: seen.append(a))
        assert seen[0][3] is True

        core = self._core()
        with patch("agent.core.execute_tool", return_value="ok"):
            core.run_turn("go")
        assert [m for m in core.messages if m["role"] == "tool"][0]["content"] == "ok"


class _Delta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta=None, finish_reason=None):
        self.delta = _Delta(**(delta or {}))
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices=None, usage=None):
        self.choices = [_Choice(**c) for c in (choices or [])]
        self.usage = usage
        self.model = "test"


class _UsageObject:
    """A CompletionUsage dataclass look-alike."""

    prompt_tokens = 11
    completion_tokens = 4


def _bare_core():
    core = cast(Any, Core.__new__(Core))
    core.config = _Config()
    core._turn_cancelled = False
    core.thinking = False
    core.generating = False
    return core


def _text_chunk(text, finish_reason=None):
    return _Chunk(choices=[{"delta": {"content": text}, "finish_reason": finish_reason}])


class TestStreamCompleteness(unittest.TestCase):
    """A stream that ends without a terminal marker delivered a prefix."""

    def _run(self, *chunks, **kw):
        core = _bare_core()
        core._stream_handler(iter(chunks), **kw)
        return core

    def test_the_terminal_marker_decides(self):
        for reason, incomplete in (("stop", False), ("length", True),
                                   ("tool_calls", False)):
            core = self._run(
                _text_chunk("text"),
                _Chunk(choices=[{"finish_reason": reason}]),
            )
            assert (core.finish_reason, core.stream_incomplete) == (reason, incomplete)
        # No marker ever arrived: what a cut-off stream looks like.
        core = self._run(_text_chunk("half a sent"))
        assert core.finish_reason is None and core.stream_incomplete is True

    def test_an_empty_cancelled_or_errored_stream_is_not_flagged(self):
        assert self._run(_Chunk(choices=[])).stream_incomplete is False
        core = _bare_core()
        core._stream_handler(iter([_text_chunk("partial")]), poll=lambda: True)
        assert core._turn_cancelled and core.stream_incomplete is False

        def _boom():
            yield _text_chunk("text")
            raise ValueError("malformed SSE")

        core = _bare_core()
        _, _, stream_error, _ = core._stream_handler(_boom())
        assert stream_error is not None and core.stream_incomplete is False

    def test_usage_is_read_without_swallowing_a_content_delta(self):
        cases = (
            (_Chunk(choices=[], usage={"prompt_tokens": 7, "completion_tokens": 3}),
             {"prompt_tokens": 7, "completion_tokens": 3}),
            (_Chunk(choices=[], usage=_UsageObject()),
             {"prompt_tokens": 11, "completion_tokens": 4}),
        )
        for chunk, expected in cases:
            core = _bare_core()
            _, _, _, usage = core._stream_handler(iter([_text_chunk("x"), chunk]))
            assert usage == expected
            assert core.response_buffer == "x"
        # A provider may attach usage to its last *content* chunk: keep the text.
        core = self._run(_Chunk(
            choices=[{"delta": {"content": "the tail"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 5, "completion_tokens": 2},
        ))
        assert core.response_buffer == "the tail"


class TestIncompleteTurnResult(unittest.TestCase):
    """The flag reaches the caller and the stored exchange."""

    def _core(self, text, *, incomplete, reason):
        core = make_core([{"text": text}])
        real_send = core._send_to_llm

        def _send(*args, **kwargs):
            result = real_send(*args, **kwargs)
            # Mirror the real _stream_handler, which records this on the core.
            core.stream_incomplete = incomplete
            core.finish_reason = reason
            return result

        core._send_to_llm = _send
        return core

    def test_the_result_and_the_stored_exchange_are_flagged(self):
        for text, incomplete, reason, expected, expected_reason in (
            ("all done.", False, "stop", False, ""),
            ("cut off here", True, "length", True, "length"),
            ("half a sent", True, None, True, "stream_ended"),
        ):
            core = self._core(text, incomplete=incomplete, reason=reason)
            result = core.run_turn("q")
            assert (result.incomplete, result.incomplete_reason) == (
                expected, expected_reason
            ), text
            assert bool(core.memory.exchanges[-1].get("incomplete")) is expected, text

    def test_a_stale_flag_does_not_mark_a_later_turn_and_partials_are_flagged(self):
        core = make_core([{"text": "first"}, {"text": "second"}])
        core.stream_incomplete = True
        assert core.run_turn("q").incomplete is False

        core = make_core([])
        core.messages = []
        core.response_buffer = "streamed half a sen"
        core._persist_partial_turn("q")
        stored = [e for e in core.memory.exchanges if e["role"] == "assistant"]
        assert stored[-1].get("incomplete")


if __name__ == "__main__":
    unittest.main()
