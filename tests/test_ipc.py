"""Tests for agent/ipc.py — message envelope, framing, version, loopback transport."""

import json
import unittest

from agent.ipc import (
    PROTOCOL_VERSION,
    ClientRequest,
    Event,
    InjectPayload,
    InjectWhen,
    Message,
    MessageKind,
    OutputPayload,
    ProtocolError,
    ServerRequest,
    ToolResultPayload,
    Transport,
    TransportClosed,
    check_protocol_version,
    decode_message,
    encode_message,
    loopback_pair,
    payload_as,
)


class TestMessageFactories(unittest.TestCase):
    def test_each_constructor_sets_its_kind_and_names(self):
        event = Message.event(Event.TURN_END, {"response": "hi"}, turn_id="t1")
        assert (event.kind, event.name) == (MessageKind.EVENT, "turn_end")
        assert event.is_event and not event.is_request and not event.is_reply

        request = Message.request(ClientRequest.PROMPT, {"text": "go"})
        assert (request.kind, request.name, request.is_request) == (
            MessageKind.REQUEST, "prompt", True
        )

        server_request = Message.request(ServerRequest.CONFIRM, {"message": "ok?"})
        assert server_request.name == "confirm"

        reply = Message.response("abc", {"value": True})
        assert (reply.kind, reply.reply_to, reply.is_reply) == (
            MessageKind.RESPONSE, "abc", True
        )

        error = Message.error("abc", "boom", kind="timeout")
        assert error.reply_to == "abc"
        assert (error.payload["message"], error.payload["kind"]) == ("boom", "timeout")

        # Ids are unique, a timestamp is set, and a payload may be a dataclass.
        assert len({Message.event(Event.PONG).id for _ in range(50)}) == 50
        msg = Message.event(Event.PONG)
        assert msg.ts > 0 and msg.payload == {}
        assert Message.event(Event.OUTPUT, OutputPayload(text="hi")).payload["text"] == "hi"


class TestSerialization(unittest.TestCase):
    def test_dict_and_json_round_trip(self):
        for original in (
            Message.event(Event.TOOL_RESULT, ToolResultPayload(
                id="c1", name="read_file", content="data"), turn_id="t9"),
            Message.request(ClientRequest.INJECT, InjectPayload(
                text="stop", when=InjectWhen.INTERRUPT)),
        ):
            assert Message.from_dict(original.to_dict()) == original
            line = encode_message(original)
            # One message per line, even when the payload holds newlines.
            assert line.endswith("\n") and line.count("\n") == 1
            assert decode_message(line) == original
        assert json.loads(encode_message(Message.event(Event.STATUS, {"busy": True})))[
            "name"
        ] == "status"

    def test_malformed_frames_raise_protocol_error(self):
        for payload in (
            "   ",
            "{not json}",
            "[1, 2, 3]",
            '{"kind": "nope", "name": "x"}',
            '{"kind": "event"}',
            '{"kind": "event", "name": "x", "payload": 3}',
        ):
            with self.assertRaises(ProtocolError, msg=payload):
                decode_message(payload)

    def test_optional_fields_may_be_absent(self):
        msg = decode_message('{"kind": "event", "name": "pong"}')
        assert (msg.reply_to, msg.turn_id) == ("", "")
        assert msg.id


class TestProtocolVersion(unittest.TestCase):
    def test_only_a_major_mismatch_is_fatal(self):
        major = PROTOCOL_VERSION.split(".")[0]
        check_protocol_version(PROTOCOL_VERSION)
        check_protocol_version(f"{major}.999")  # a newer minor is compatible
        for bad in ("99.0", "abc"):
            with self.assertRaises(ProtocolError, msg=bad):
                check_protocol_version(bad)


class TestPayloadAs(unittest.TestCase):
    def test_reconstructs_ignoring_unknown_keys_and_defaulting_the_rest(self):
        full = Message.request(ClientRequest.INJECT, InjectPayload(text="x", role="system"))
        assert payload_as(InjectPayload, full).role == "system"

        future = Message(kind=MessageKind.EVENT, name="x",
                         payload={"text": "y", "future": 1})
        p = payload_as(InjectPayload, future)
        assert p.text == "y" and p.when == InjectWhen.BETWEEN_TURNS

        assert payload_as(InjectPayload, Message(
            kind=MessageKind.EVENT, name="x", payload={}
        )).text == ""


class TestLoopbackTransport(unittest.TestCase):
    def test_bidirectional_send_and_recv(self):
        a, b = loopback_pair()
        first = Message.event(Event.PONG, {"nonce": "1"})
        second = Message.request(ClientRequest.PING, {"nonce": "2"})
        a.send(first)
        b.send(second)
        assert b.recv() == first and a.recv() == second
        assert isinstance(a, Transport)
        # Nothing queued: a timeout and a zero wait both return None.
        assert a.recv(timeout=0.01) is None and a.recv(timeout=0) is None

    def test_closing_wakes_the_peer_and_refuses_further_io(self):
        a, b = loopback_pair()
        assert a.closed is False
        a.close()
        a.close()  # idempotent
        assert a.closed is True and b.closed is False
        with self.assertRaises(TransportClosed):
            b.recv(timeout=0.1)
        assert b.closed is True
        a2, _ = loopback_pair()
        a2.close()
        for call in (lambda: a2.send(Message.event(Event.PONG)),
                     lambda: a2.recv(timeout=0)):
            with self.assertRaises(TransportClosed):
                call()
