"""Tests for agent/keys.py and the prompt's newline bindings.

The interesting property is that a terminal sends one byte (0x0D) for Enter,
Shift+Enter and Ctrl+Enter, so "Shift+Enter inserts a newline" is only
possible when the terminal sends something distinguishable and we teach
prompt_toolkit to parse it.
"""

import io
import unittest
from typing import Any
from unittest import mock

from agent.keys import (
    CSI_U_ENTER,
    PUSH,
    SUPPORTED_TERMS,
    describe_enter_sequence,
    extend_ansi_sequences,
    is_modified_enter,
    terminal_supports_kitty_protocol,
)

# The sequences a real kitty sends for a modified Enter, captured from an
# actual terminal session. They are *not* the bare 2/3/5 the protocol's
# examples show: kitty encodes modifiers as ``1 + bitfield`` and reports the
# num_lock bit, so Shift+Enter arrives as ``CSI 13 ; 130 u``. A table that
# only knows ``13;2u`` makes the terminal print its own escape code into the
# prompt, which is exactly the bug these pin down.
SHIFT_ENTER = "\x1b[13;130u"
CTRL_ENTER = "\x1b[13;133u"
ALT_ENTER = "\x1b[13;131u"


class _Event:
    """Stand-in for a prompt_toolkit KeyPressEvent."""

    def __init__(self, data: str | None) -> None:
        self.data = data


class TestIsModifiedEnter(unittest.TestCase):
    """A plain Enter is a bare CR; a modified one arrives as a sequence."""

    def test_plain_enter_is_not_modified(self):
        self.assertFalse(is_modified_enter(_Event("\r")))

    def test_xterm_modify_other_keys_is_modified(self):
        # Already in prompt_toolkit's table; survives on event.data.
        self.assertTrue(is_modified_enter(_Event("\x1b[27;2;13~")))

    def test_kitty_csi_u_is_modified(self):
        self.assertTrue(is_modified_enter(_Event("\x1b[13;2u")))
        self.assertTrue(is_modified_enter(_Event(SHIFT_ENTER)))

    def test_missing_data_is_treated_as_plain(self):
        self.assertFalse(is_modified_enter(_Event(None)))


class TestDescribeEnterSequence(unittest.TestCase):
    def test_bare_enter(self):
        self.assertEqual(describe_enter_sequence(CSI_U_ENTER), "enter")

    def test_decodes_the_sequences_a_real_terminal_sends(self):
        self.assertEqual(
            describe_enter_sequence(SHIFT_ENTER), "enter+shift+num_lock")
        self.assertEqual(
            describe_enter_sequence(CTRL_ENTER), "enter+ctrl+num_lock")
        self.assertEqual(
            describe_enter_sequence(ALT_ENTER), "enter+alt+num_lock")

    def test_decodes_the_bare_modifier_values(self):
        self.assertEqual(describe_enter_sequence("\x1b[13;2u"), "enter+shift")
        self.assertEqual(describe_enter_sequence("\x1b[13;3u"), "enter+alt")
        self.assertEqual(describe_enter_sequence("\x1b[13;5u"), "enter+ctrl")

    def test_unknown_sequence_is_returned_unchanged(self):
        self.assertEqual(describe_enter_sequence("\x1b[A"), "\x1b[A")
        self.assertEqual(describe_enter_sequence("x"), "x")


class TestExtendAnsiSequences(unittest.TestCase):
    def test_it_maps_the_csi_u_forms(self):
        extend_ansi_sequences()
        from prompt_toolkit.input import ansi_escape_sequences as A
        from prompt_toolkit.keys import Keys

        for sequence in (CSI_U_ENTER, "\x1b[13;2u", "\x1b[13;5u", "\x1b[13;3u"):
            self.assertEqual(A.ANSI_SEQUENCES.get(sequence), Keys.ControlM)

    def test_it_covers_the_sequences_a_real_terminal_sends(self):
        """The whole modifier field, not a hand-picked few.

        A miss is not a graceful degradation: prompt_toolkit types the raw
        escape sequence into the prompt buffer instead.
        """
        extend_ansi_sequences()
        from prompt_toolkit.input import ansi_escape_sequences as A
        from prompt_toolkit.keys import Keys

        for sequence in (SHIFT_ENTER, CTRL_ENTER, ALT_ENTER):
            self.assertEqual(
                A.ANSI_SEQUENCES.get(sequence), Keys.ControlM, repr(sequence))

    def test_it_is_idempotent(self):
        extend_ansi_sequences()
        extend_ansi_sequences()

    def test_the_parser_actually_decodes_them(self):
        extend_ansi_sequences()
        from prompt_toolkit.input.vt100_parser import Vt100Parser

        seen = []
        parser = Vt100Parser(seen.append)
        parser.feed(SHIFT_ENTER)
        parser.flush()
        self.assertEqual(len(seen), 1)
        # The key name is c-m, exactly like a plain Enter: only the raw data
        # distinguishes them, which is why the bindings test event.data.
        self.assertEqual(seen[0].key.value, "c-m")
        self.assertEqual(seen[0].data, SHIFT_ENTER)


class TestTerminalSupport(unittest.TestCase):
    def test_known_terms(self):
        for term in SUPPORTED_TERMS:
            with mock.patch.dict("os.environ", {"TERM": term}, clear=False):
                self.assertTrue(terminal_supports_kitty_protocol(), term)

    def test_unknown_term_is_unsupported(self):
        env = {"TERM": "xterm-256color"}
        env.pop("TERM_PROGRAM", None)
        with mock.patch.dict("os.environ", env, clear=True):
            self.assertFalse(terminal_supports_kitty_protocol())

    def test_term_program_covers_macos_emulators(self):
        env = {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}
        with mock.patch.dict("os.environ", env, clear=True):
            self.assertTrue(terminal_supports_kitty_protocol())

    def test_push_sequence_is_the_disambiguate_flag(self):
        self.assertEqual(PUSH, "\x1b[>1u")


class TestEnableDisable(unittest.TestCase):
    """Enabling must be paired with a pop, or the shell inherits the mode."""

    def _capture(self):
        buf: Any = io.StringIO()
        buf.isatty = lambda: True
        return buf

    def test_no_tty_is_a_quiet_no_op(self):
        from agent import keys

        buf: Any = self._capture()
        buf.isatty = lambda: False
        with mock.patch.object(keys.sys, "stdout", buf):
            self.assertFalse(keys.enable_kitty_keyboard())
        self.assertEqual(buf.getvalue(), "")

    def test_enable_writes_the_push_and_disable_the_pop(self):
        from agent import keys

        buf = self._capture()
        with mock.patch.object(keys.sys, "stdout", buf):
            self.assertTrue(keys.enable_kitty_keyboard())
            keys.disable_kitty_keyboard()
        self.assertEqual(buf.getvalue(), PUSH + keys.POP)


def _prompt_session_bindings():
    """The REPL's Enter bindings, from the one place they are defined."""
    from prompt_toolkit.key_binding import KeyBindings

    from agent.keys import install_enter_bindings

    kb = KeyBindings()
    install_enter_bindings(kb)
    return kb


class TestNewlineBindings(unittest.TestCase):
    """End-to-end through a real PromptSession on a pipe input."""

    def _run(self, keys: str) -> str | None:
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.shortcuts import PromptSession

        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = PromptSession(
                input=pipe,
                output=DummyOutput(),
                key_bindings=_prompt_session_bindings(),
                multiline=True,
            )
            try:
                return session.prompt()
            except EOFError:
                return None

    def test_plain_enter_submits(self):
        self.assertEqual(self._run("hi\r"), "hi")

    def test_shift_enter_inserts_a_newline(self):
        self.assertEqual(self._run("a\x1b[13;2ub\r"), "a\nb")

    def test_shift_enter_with_num_lock_inserts_a_newline(self):
        """The sequence kitty actually sends when num_lock is on."""
        self.assertEqual(self._run(f"a{SHIFT_ENTER}b\r"), "a\nb")

    def test_ctrl_enter_with_num_lock_inserts_a_newline(self):
        self.assertEqual(self._run(f"a{CTRL_ENTER}b\r"), "a\nb")

    def test_alt_enter_csi_u_with_num_lock_inserts_a_newline(self):
        self.assertEqual(self._run(f"a{ALT_ENTER}b\r"), "a\nb")

    def test_ctrl_enter_inserts_a_newline(self):
        self.assertEqual(self._run("a\x1b[13;5ub\r"), "a\nb")

    def test_alt_enter_escape_prefix_inserts_a_newline(self):
        """The legacy ESC + CR encoding, for terminals without the protocol."""
        self.assertEqual(self._run("a\x1b\rb\r"), "a\nb")

    def test_xterm_modify_other_keys_shift_enter(self):
        self.assertEqual(self._run("a\x1b[27;2;13~b\r"), "a\nb")

    def test_ctrl_j_inserts_a_newline(self):
        """Ctrl+J used to submit, because the enter binding also matched it."""
        self.assertEqual(self._run("a\nb\r"), "a\nb")


if __name__ == "__main__":
    unittest.main()