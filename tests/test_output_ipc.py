"""Tests for the IPC-backed output adapter (client/server phase 1)."""

import unittest

import agent.output as out_mod
from agent.ipc import Event, OutputPayload, loopback_pair, payload_as
from agent.output import IpcOutputAdapter, get_output_or_ipc, set_output
from tests.conftest import BaseTest


class TestIpcOutputAdapter(BaseTest):
    def setUp(self):
        super().setUp()
        self.transport, self.peer = loopback_pair()
        self.adapter = IpcOutputAdapter(transport=self.transport)

    def _recv_all(self):
        out = []
        while (msg := self.peer.recv(timeout=0.05)) is not None:
            out.append(msg)
        return out

    def test_print_levels_and_rules_become_output_events(self):
        self.adapter.print("hello")
        self.adapter.err("bad")
        self.adapter.ok("good")
        self.adapter.info("meh")
        self.adapter.rule(title="T", style="status")
        payloads = [payload_as(OutputPayload, m) for m in self._recv_all()]
        assert [p.level for p in payloads[1:4]] == ["err", "ok", "info"]
        assert (payloads[0].text, payloads[0].format, payloads[0].level) == (
            "hello", "text", "normal"
        )
        assert payloads[-1].format == "rule" and payloads[-1].title == "T"

    def test_prompts_take_the_default_and_subprocesses_are_refused(self):
        assert self.adapter.ask_confirm("ok?", default=False) is False
        assert self.adapter.ask_confirm("ok?", default=True) is True
        assert self.adapter.ask_string("name?", default="x") == "x"
        assert self.adapter.ask_float("n?", default=1.5) == 1.5
        assert self.adapter.ask_choice("pick?", [("a", "A"), ("b", "B")]) == "a"
        with self.assertRaises(RuntimeError):
            self.adapter.run_subprocess(["ls"])

    def test_a_dead_transport_does_not_raise_and_the_active_output_wins(self):
        self.transport.close()
        self.adapter.print("no crash")
        out_mod._active_output = None
        assert isinstance(get_output_or_ipc(), IpcOutputAdapter)
        sentinel = IpcOutputAdapter(transport=self.transport)
        set_output(sentinel)
        assert get_output_or_ipc() is sentinel
