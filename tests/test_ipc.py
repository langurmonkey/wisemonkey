"""Tests for agent/ipc.py — protocol dataclasses, serialization, transports."""

import json
import unittest

from agent.ipc import (
    PROTOCOL_VERSION,
    AskChoicePayload,
    ClientRequest,
    CommandResultPayload,
    Event,
    HandshakePayload,
    InjectPayload,
    InjectWhen,
    LoopbackTransport,
    Message,
    MessageKind,
    OutputFormat,
    OutputLevel,
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
    """Message.event / request / response / error constructors."""

    def test_event(self):
        msg = Message.event(Event.TURN_END, {"response": "hi"}, turn_id="t1")
        assert msg.kind == MessageKind.EVENT
        assert msg.name == "turn_end"
        assert msg.payload == {"response": "hi"}
        assert msg.turn_id == "t1"
        assert msg.is_event and not msg.is_request and not msg.is_reply

    def test_event_accepts_dataclass_payload(self):
        msg = Message.event(Event.OUTPUT, OutputPayload(text="hello"))
        assert msg.payload["text"] == "hello"
        assert msg.payload["format"] == OutputFormat.TEXT

    def test_event_without_payload(self):
        msg = Message.event(Event.PONG)
        assert msg.payload == {}

    def test_request(self):
        msg = Message.request(ClientRequest.PROMPT, {"text": "go"})
        assert msg.kind == MessageKind.REQUEST
        assert msg.name == "prompt"
        assert msg.is_request

    def test_server_request(self):
        msg = Message.request(ServerRequest.CONFIRM, {"message": "ok?"})
        assert msg.name == "confirm"
        assert msg.payload["message"] == "ok?"

    def test_response(self):
        msg = Message.response("abc", {"value": True})
        assert msg.kind == MessageKind.RESPONSE
        assert msg.reply_to == "abc"
        assert msg.is_reply

    def test_error(self):
        msg = Message.error("abc", "boom", kind="timeout")
        assert msg.kind == MessageKind.ERROR
        assert msg.reply_to == "abc"
        assert msg.payload["message"] == "boom"
        assert msg.payload["kind"] == "timeout"

    def test_ids_are_unique(self):
        ids = {Message.event(Event.PONG).id for _ in range(50)}
        assert len(ids) == 50

    def test_timestamp_is_set(self):
        assert Message.event(Event.PONG).ts > 0


class TestSerialization(unittest.TestCase):
    """Round-trips through dict and newline-delimited JSON."""

    def test_dict_round_trip(self):
        original = Message.event(Event.TOOL_RESULT, ToolResultPayload(
            id="c1", name="read_file", content="data",
        ), turn_id="t9")
        restored = Message.from_dict(original.to_dict())
        assert restored == original

    def test_encode_decode_round_trip(self):
        original = Message.request(ClientRequest.INJECT, InjectPayload(
            text="stop", when=InjectWhen.INTERRUPT,
        ))
        line = encode_message(original)
        assert line.endswith("\n")
        assert line.count("\n") == 1
        assert decode_message(line) == original

    def test_encode_is_single_line(self):
        msg = Message.event(Event.OUTPUT, OutputPayload(text="a\nb\nc"))
        assert encode_message(msg).count("\n") == 1

    def test_encode_is_json_serializable(self):
        msg = Message.event(Event.STATUS, {"busy": True})
        assert json.loads(encode_message(msg))["name"] == "status"

    def test_decode_empty_raises(self):
        with self.assertRaises(ProtocolError):
            decode_message("   ")

    def test_decode_bad_json_raises(self):
        with self.assertRaises(ProtocolError):
            decode_message("{not json}")

    def test_decode_non_object_raises(self):
        with self.assertRaises(ProtocolError):
            decode_message("[1, 2, 3]")

    def test_decode_unknown_kind_raises(self):
        with self.assertRaises(ProtocolError):
            decode_message('{"kind": "nope", "name": "x"}')

    def test_decode_missing_name_raises(self):
        with self.assertRaises(ProtocolError):
            decode_message('{"kind": "event"}')

    def test_decode_bad_payload_raises(self):
        with self.assertRaises(ProtocolError):
            decode_message('{"kind": "event", "name": "x", "payload": 3}')

    def test_decode_tolerates_missing_optional_fields(self):
        msg = decode_message('{"kind": "event", "name": "pong"}')
        assert msg.reply_to == ""
        assert msg.turn_id == ""
        assert msg.id


class TestProtocolVersion(unittest.TestCase):
    def test_matching_version_ok(self):
        check_protocol_version(PROTOCOL_VERSION)

    def test_minor_difference_ok(self):
        major = PROTOCOL_VERSION.split(".")[0]
        check_protocol_version(f"{major}.999")

    def test_major_mismatch_raises(self):
        with self.assertRaises(ProtocolError):
            check_protocol_version("99.0")

    def test_malformed_version_raises(self):
        with self.assertRaises(ProtocolError):
            check_protocol_version("abc")


class TestPayloads(unittest.TestCase):
    """Payload dataclasses cover the OutputAdapter and turn surfaces."""

    def test_output_payload_defaults(self):
        p = OutputPayload(text="hi", format=OutputFormat.MARKUP, level=OutputLevel.OK)
        assert p.end == "\n"
        assert p.indent == 0
        assert p.level == OutputLevel.OK

    def test_output_payload_panel(self):
        p = OutputPayload(format=OutputFormat.PANEL, text="# md", title="t", subtitle="s")
        d = json.loads(json.dumps(p.__dict__))
        assert d["format"] == "panel"
        assert d["title"] == "t"

    def test_output_payload_rule(self):
        p = OutputPayload(format=OutputFormat.RULE, style="agent", align="left")
        assert p.style == "agent"
        assert p.align == "left"

    def test_ask_choice_options_are_lists(self):
        p = AskChoicePayload(message="pick", options=[["a", "Alpha"], ["b", "Beta"]])
        assert json.loads(json.dumps(p.__dict__))["options"] == [["a", "Alpha"], ["b", "Beta"]]

    def test_tool_result_image(self):
        p = ToolResultPayload(name="screenshot", image_base64="QUJD", mime_type="image/jpeg")
        assert p.image_base64 == "QUJD"
        assert p.is_error is False

    def test_command_result(self):
        p = CommandResultPayload(command="help", ok=True, should_exit=False)
        assert p.msg == ""
        assert p.markdown == ""

    def test_handshake(self):
        p = HandshakePayload(session="s1", model="m", capabilities=["commands"])
        assert p.protocol_version == PROTOCOL_VERSION
        assert json.loads(json.dumps(p.__dict__))["capabilities"] == ["commands"]

    def test_inject_when_values(self):
        assert InjectWhen.BETWEEN_TURNS == "between_turns"
        assert InjectWhen.AFTER_TOOL == "after_tool"
        assert InjectWhen.INTERRUPT == "interrupt"

    def test_inject_payload_defaults(self):
        p = InjectPayload(text="steer")
        assert p.when == InjectWhen.BETWEEN_TURNS
        assert p.role == "user"


class TestPayloadAs(unittest.TestCase):
    def test_reconstruct(self):
        msg = Message.request(ClientRequest.INJECT, InjectPayload(text="x", role="system"))
        p = payload_as(InjectPayload, msg)
        assert p.text == "x"
        assert p.role == "system"

    def test_unknown_keys_ignored(self):
        msg = Message(kind=MessageKind.EVENT, name="x", payload={"text": "y", "future": 1})
        p = payload_as(InjectPayload, msg)
        assert p.text == "y"

    def test_missing_keys_use_defaults(self):
        msg = Message(kind=MessageKind.EVENT, name="x", payload={})
        p = payload_as(InjectPayload, msg)
        assert p.text == ""
        assert p.when == InjectWhen.BETWEEN_TURNS


class TestLoopbackTransport(unittest.TestCase):
    def test_send_recv_both_directions(self):
        a, b = loopback_pair()
        m1 = Message.event(Event.PONG, {"nonce": "1"})
        m2 = Message.request(ClientRequest.PING, {"nonce": "2"})
        a.send(m1)
        b.send(m2)
        assert b.recv() == m1
        assert a.recv() == m2

    def test_recv_timeout_returns_none(self):
        a, _ = loopback_pair()
        assert a.recv(timeout=0.01) is None

    def test_recv_non_blocking_returns_none(self):
        a, _ = loopback_pair()
        assert a.recv(timeout=0) is None

    def test_close_marks_self_closed(self):
        a, _ = loopback_pair()
        assert a.closed is False
        a.close()
        assert a.closed is True

    def test_close_wakes_peer(self):
        a, b = loopback_pair()
        a.close()
        assert b.closed is False
        with self.assertRaises(TransportClosed):
            b.recv(timeout=0.1)
        assert b.closed is True

    def test_send_after_close_raises(self):
        a, _ = loopback_pair()
        a.close()
        with self.assertRaises(TransportClosed):
            a.send(Message.event(Event.PONG))

    def test_recv_after_close_raises(self):
        a, _ = loopback_pair()
        a.close()
        with self.assertRaises(TransportClosed):
            a.recv(timeout=0)

    def test_double_close_is_idempotent(self):
        a, _ = loopback_pair()
        a.close()
        a.close()
        assert a.closed is True

    def test_satisfies_transport_protocol(self):
        a, _ = loopback_pair()
        assert isinstance(a, Transport)


if __name__ == "__main__":
    unittest.main()