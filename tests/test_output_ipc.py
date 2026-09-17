"""Tests for the IPC-backed output adapter (client/server phase 1)."""

import unittest

from tests.conftest import BaseTest

from agent.ipc import Event, OutputPayload, loopback_pair, payload_as
from agent.output import IpcOutputAdapter, get_output_or_ipc, set_output


class TestIpcOutputAdapter(BaseTest):
    def setUp(self):
        super().setUp()
        self.transport, self.peer = loopback_pair()
        self.adapter = IpcOutputAdapter(transport=self.transport)

    def _recv_all(self):
        out = []
        while True:
            msg = self.peer.recv(timeout=0.05)
            if msg is None:
                break
            out.append(msg)
        return out

    def test_print_emits_output_event(self):
        self.adapter.print("hello")
        msgs = self._recv_all()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, Event.OUTPUT)
        payload = payload_as(OutputPayload, msgs[0])
        self.assertEqual(payload.text, "hello")
        self.assertEqual(payload.format, "text")

    def test_levels(self):
        self.adapter.err("bad")
        self.adapter.ok("good")
        self.adapter.info("meh")
        msgs = self._recv_all()
        self.assertEqual([payload_as(OutputPayload, m).level for m in msgs], ["err", "ok", "info"])

    def test_rule(self):
        self.adapter.rule(title="T", style="status")
        (msg,) = self._recv_all()
        rp = payload_as(OutputPayload, msg)
        self.assertEqual(rp.format, "rule")
        self.assertEqual(rp.title, "T")

    def test_non_interactive_confirm_denies(self):
        self.assertFalse(self.adapter.ask_confirm("ok?", default=False))
        self.assertTrue(self.adapter.ask_confirm("ok?", default=True))

    def test_non_interactive_string_returns_default(self):
        self.assertEqual(self.adapter.ask_string("name?", default="x"), "x")

    def test_non_interactive_float_returns_default(self):
        self.assertEqual(self.adapter.ask_float("n?", default=1.5), 1.5)

    def test_non_interactive_choice_returns_first(self):
        result = self.adapter.ask_choice("pick?", [("a", "A"), ("b", "B")])
        self.assertEqual(result, "a")

    def test_subprocess_denied(self):
        with self.assertRaises(RuntimeError):
            self.adapter.run_subprocess(["ls"])

    def test_send_failure_is_swallowed(self):
        self.transport.close()
        self.adapter.print("no crash")  # must not raise

    def test_get_output_or_ipc_fallback(self):
        import agent.output as out_mod

        out_mod._active_output = None
        adapter = get_output_or_ipc()
        self.assertIsInstance(adapter, IpcOutputAdapter)

    def test_get_output_or_ipc_prefers_active(self):
        sentinel = IpcOutputAdapter(transport=self.transport)
        set_output(sentinel)
        self.assertIs(get_output_or_ipc(), sentinel)


if __name__ == "__main__":
    unittest.main()
