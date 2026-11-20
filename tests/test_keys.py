"""Tests for agent/keys.py and the prompt's newline bindings."""

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
    extended_sequences,
    is_ctrl_c_sequence,
    is_modified_enter,
    terminal_supports_kitty_protocol,
)

# As kitty really sends them: modifier 133 is ctrl + num_lock (1 + 4 + 128).
CTRL_C_CSI_U = "\x1b[99;133u"
SHIFT_ENTER = "\x1b[13;130u"
CTRL_ENTER = "\x1b[13;133u"
ALT_ENTER = "\x1b[13;131u"


class _Event:
    """Stand-in for a prompt_toolkit KeyPressEvent."""

    def __init__(self, data: str | None) -> None:
        self.data = data


def _decoded(sequence: str) -> list[Any]:
    """The key names prompt_toolkit's parser produces for *sequence*."""
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    seen: list[Any] = []
    parser = Vt100Parser(seen.append)
    parser.feed(sequence)
    parser.flush()
    return [getattr(k.key, "value", k.key) for k in seen]


class TestIsModifiedEnter(unittest.TestCase):
    def test_only_a_sequence_counts_as_modified(self):
        """A plain Enter is a bare CR; a modified one arrives as a sequence."""
        self.assertFalse(is_modified_enter(_Event("\r")))
        self.assertFalse(is_modified_enter(_Event(None)))
        for sequence in ("\x1b[27;2;13~", "\x1b[13;2u", SHIFT_ENTER):
            self.assertTrue(is_modified_enter(_Event(sequence)), repr(sequence))


class TestCtrlCSequences(unittest.TestCase):
    def test_ctrl_c_is_mapped_for_every_modifier_but_the_bare_form(self):
        extend_ansi_sequences()
        from prompt_toolkit.input import ansi_escape_sequences as A
        from prompt_toolkit.keys import Keys

        for mods in range(2, 258):
            sequence = f"\x1b[99;{mods}u"
            self.assertTrue(is_ctrl_c_sequence(sequence), sequence)
            self.assertEqual(A.ANSI_SEQUENCES.get(sequence), Keys.ControlC, sequence)
        # A modifier value of 1 means "no modifiers": plain 'c' stays text.
        self.assertNotIn("\x1b[99u", A.ANSI_SEQUENCES)
        for sequence in ("\x1b[99u", "\x03", "c"):
            self.assertFalse(is_ctrl_c_sequence(sequence), repr(sequence))
        self.assertEqual(_decoded(CTRL_C_CSI_U), ["c-c"])


class TestDescribeEnterSequence(unittest.TestCase):
    def test_it_decodes_what_a_real_terminal_sends(self):
        for sequence, expected in (
            (CSI_U_ENTER, "enter"),
            (SHIFT_ENTER, "enter+shift+num_lock"),
            (CTRL_ENTER, "enter+ctrl+num_lock"),
            (ALT_ENTER, "enter+alt+num_lock"),
            ("\x1b[13;2u", "enter+shift"),
            ("\x1b[13;3u", "enter+alt"),
            ("\x1b[13;5u", "enter+ctrl"),
            ("\x1b[A", "\x1b[A"),
            ("x", "x"),
        ):
            self.assertEqual(describe_enter_sequence(sequence), expected, repr(sequence))


class TestExtendAnsiSequences(unittest.TestCase):
    def test_the_csi_u_forms_map_to_c_m_with_their_raw_data(self):
        extend_ansi_sequences()
        from prompt_toolkit.input import ansi_escape_sequences as A
        from prompt_toolkit.keys import Keys

        for sequence in (
            CSI_U_ENTER, "\x1b[13;2u", "\x1b[13;3u", "\x1b[13;5u",
            SHIFT_ENTER, CTRL_ENTER, ALT_ENTER,
        ):
            self.assertEqual(A.ANSI_SEQUENCES.get(sequence), Keys.ControlM, repr(sequence))
        # The key name is c-m, exactly like a plain Enter: only the raw data
        # distinguishes them, which is why the bindings test event.data.
        self.assertEqual(_decoded(SHIFT_ENTER), ["c-m"])


class TestTerminalSupport(unittest.TestCase):
    def test_known_terms_and_term_program(self):
        for term in SUPPORTED_TERMS:
            with mock.patch.dict("os.environ", {"TERM": term}, clear=False):
                self.assertTrue(terminal_supports_kitty_protocol(), term)
        for env, expected in (
            ({"TERM": "xterm-256color"}, False),
            ({"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}, True),
        ):
            with mock.patch.dict("os.environ", env, clear=True):
                self.assertIs(terminal_supports_kitty_protocol(), expected, env)


class TestLegacyKeyEncoding(unittest.TestCase):
    """readline readers need the terminal's *legacy* encodings."""

    def setUp(self) -> None:
        from agent import keys

        self.keys = keys
        self.buf: Any = io.StringIO()
        self.buf.isatty = lambda: True
        patches = [
            mock.patch.object(keys.sys, "stdout", self.buf),
            mock.patch.object(keys, "_protocol_pushed", False),
            mock.patch.object(keys, "_legacy_encoding_depth", 0),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

    def _clear(self) -> None:
        self.buf.truncate(0)
        self.buf.seek(0)

    def test_it_pops_inside_and_pushes_back_after(self):
        self.keys.enable_kitty_keyboard()
        self._clear()
        with self.keys.legacy_key_encoding():
            self.assertFalse(self.keys.protocol_pushed())
        self.assertTrue(self.keys.protocol_pushed())
        self.assertEqual(self.buf.getvalue(), self.keys.POP + self.keys.PUSH)

    def test_nesting_pops_once_and_an_exception_still_restores(self):
        self.keys.enable_kitty_keyboard()
        self._clear()
        with self.assertRaises(RuntimeError):
            with self.keys.legacy_key_encoding():
                with self.keys.legacy_key_encoding():
                    self.assertFalse(self.keys.protocol_pushed())
                # The inner exit must not re-push while the outer still holds.
                self.assertFalse(self.keys.protocol_pushed())
                raise RuntimeError("boom")
        self.assertTrue(self.keys.protocol_pushed())
        self.assertEqual(self.buf.getvalue(), self.keys.POP + self.keys.PUSH)

    def test_nothing_is_sent_when_the_flag_was_never_pushed(self):
        with self.keys.legacy_key_encoding():
            pass
        self.assertEqual(self.buf.getvalue(), "")


class TestEnsureReadline(unittest.TestCase):
    """The builtin ``input()`` has no line editor until readline is imported."""

    def test_it_imports_readline_and_survives_a_missing_one(self):
        from agent import keys

        with mock.patch.object(keys, "_readline_loaded", False):
            self.assertTrue(keys.ensure_readline())
            self.assertTrue(keys.ensure_readline())  # idempotent
        # Windows builds can lack it: plain behaviour beats an ImportError.
        with mock.patch.object(keys, "_readline_loaded", False), mock.patch.dict(
            "sys.modules", {"readline": None}
        ):
            self.assertFalse(keys.ensure_readline())


class TestMarkPromptNonprinting(unittest.TestCase):
    def test_every_escape_family_is_wrapped_and_text_is_untouched(self):
        from agent.keys import mark_prompt_nonprinting

        self.assertEqual(mark_prompt_nonprinting("hello"), "hello")
        for escape in (
            "\x1b[1m", "\x1b[2K", "\x1b[?25l", "\x1b]8;;http://x\x1b\\", "\x1bM"
        ):
            with self.subTest(escape=escape):
                self.assertEqual(
                    mark_prompt_nonprinting(f"a{escape}b"), f"a\001{escape}\002b"
                )


class TestReadlinePrompt(unittest.TestCase):
    """The prompt handed to ``input()`` must be styled but unwrapped."""

    def _console(self, *, terminal: bool):
        from rich.console import Console

        return Console(
            file=io.StringIO(),
            force_terminal=terminal,
            color_system="truecolor" if terminal else None,
            width=40,
        )

    def test_a_terminal_gets_the_styling_with_escapes_marked(self):
        from rich.text import Text

        from agent.keys import readline_prompt

        out = readline_prompt(
            self._console(terminal=True),
            Text.from_markup("[bold]Enter the endpoint URL[/bold]: "),
        )
        self.assertIn("\001\x1b[1m\002Enter the endpoint URL", out)
        self.assertIn("\001\x1b[0m\002: ", out)
        # Rich would wrap at the console width; a newline in readline's prompt
        # is a character and skews its arithmetic permanently.
        out = readline_prompt(self._console(terminal=True), Text("x" * 200))
        self.assertNotIn("\n", out)
        self.assertEqual(out.count("x"), 200)

    def test_a_pipe_gets_plain_text_with_no_markers(self):
        # input() prints a prompt verbatim when readline is not in play.
        from rich.text import Text

        from agent.keys import readline_prompt

        out = readline_prompt(
            self._console(terminal=False),
            Text.from_markup("[bold]Enter the endpoint URL[/bold]: "),
        )
        self.assertEqual(out, "Enter the endpoint URL: ")


class TestEnableDisable(unittest.TestCase):
    """Enabling must be paired with a pop, or the shell inherits the mode."""

    def _capture(self):
        buf: Any = io.StringIO()
        buf.isatty = lambda: True
        return buf

    def test_the_push_is_written_and_paired_with_a_pop(self):
        from agent import keys

        self.assertEqual(PUSH, "\x1b[>1u")  # the disambiguate flag
        buf: Any = self._capture()
        buf.isatty = lambda: False
        with mock.patch.object(keys.sys, "stdout", buf):
            self.assertFalse(keys.enable_kitty_keyboard())
        self.assertEqual(buf.getvalue(), "")

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


class TestFunctionalKeysUnderDisambiguate(unittest.TestCase):
    """Every functional key is re-encoded by the flag, not just Enter/Ctrl+C."""

    def test_the_whole_key_set_decodes(self):
        extend_ansi_sequences()
        for sequence, expected in (
            # 129 is num_lock alone and 130 num_lock+shift: lock state is not
            # a held key, so both mean "no modifier" / "shift".
            ("\x1b[1;129A", ["up"]), ("\x1b[1;130A", ["s-up"]),
            ("\x1b[1;5A", ["c-up"]), ("\x1b[1;129B", ["down"]),
            ("\x1b[1;5B", ["c-down"]), ("\x1b[1;129C", ["right"]),
            ("\x1b[1;2C", ["s-right"]), ("\x1b[1;129D", ["left"]),
            ("\x1b[1;5D", ["c-left"]), ("\x1b[1;129H", ["home"]),
            ("\x1b[1;129F", ["end"]), ("\x1b[3;129~", ["delete"]),
            ("\x1b[5;129~", ["pageup"]),
            # Shift+Backspace arrived as CSI 127;130u and typed itself.
            ("\x1b[127;130u", ["c-h"]), ("\x1b[9;130u", ["s-tab"]),
            # xterm's modifyOtherKeys, with num_lock: 130, not 2.
            ("\x1b[27;130;13~", ["c-m"]),
            # Keypad arrows navigate; keypad digits arrive as text.
            ("\x1b[57419;129u", ["up"]), ("\x1b[57423;129u", ["home"]),
            ("\x1b[57399;129u", ["0"]), ("\x1b[57413;129u", ["+"]),
            ("\x1b[57414;129u", ["c-m"]),
            ("\x1b[57376;129u", ["f13"]), ("\x1b[57386;129u", ["f23"]),
            # Alt+Arrow is not a key prompt_toolkit names: ESC then the arrow.
            ("\x1b[1;3D", ["escape", "left"]), ("\x1b[1;3A", ["escape", "up"]),
        ):
            self.assertEqual(_decoded(sequence), expected, repr(sequence))

    def test_the_whole_modifier_range_never_leaks_text(self):
        """One lock state must not be special-cased at the cost of another."""
        extend_ansi_sequences()
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

        for mods in range(2, 258):
            for sequence in (
                f"\x1b[1;{mods}A", f"\x1b[1;{mods}D", f"\x1b[3;{mods}~",
                f"\x1b[127;{mods}u", f"\x1b[57419;{mods}u",
                f"\x1b[57376;{mods}u", f"\x1b[27;{mods};13~",
            ):
                self.assertIn(sequence, ANSI_SEQUENCES, sequence)
                decoded = _decoded(sequence)
                # The parser must consume the whole sequence, never a fragment:
                # digits and ';' are what a half-consumed escape leaks.
                self.assertTrue(decoded, sequence)
                for key in decoded:
                    self.assertNotIn(key, "0123456789;", (sequence, key))

    def test_the_table_is_extended_not_rewritten_and_skips_the_bare_forms(self):
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys

        before = {k: v for k, v in ANSI_SEQUENCES.items() if isinstance(v, Keys)}
        extend_ansi_sequences()
        for sequence, key in before.items():
            self.assertEqual(ANSI_SEQUENCES.get(sequence), key, repr(sequence))
        # Modifier value 1 means *no modifiers*: the legacy encoding.
        added = extended_sequences()
        for legacy in ("\x1b[99u", "\x1b[13u", "\x1b[1A", "\x1b[1;1A", "\x1b[127u"):
            self.assertNotIn(legacy, added)

    def test_the_arrow_keys_move_a_real_buffer(self):
        """End to end: the cursor moves instead of the sequence appearing."""
        extend_ansi_sequences()
        from prompt_toolkit.buffer import Buffer

        buffer = Buffer()
        buffer.insert_text("hello world")
        buffer.cursor_position = 0
        for sequence, expected in (
            ("\x1b[1;129C", 1), ("\x1b[1;129C", 2), ("\x1b[1;129D", 1),
        ):
            for key in _decoded(sequence):
                if key == "right":
                    buffer.cursor_position += 1
                elif key == "left":
                    buffer.cursor_position -= 1
            self.assertEqual(buffer.cursor_position, expected)
        self.assertEqual(buffer.text, "hello world")


class TestNewlineBindings(unittest.TestCase):
    """End-to-end through a real PromptSession on a pipe input."""

    def _run(self, keys: str, extra_bindings=None) -> str | None:
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.shortcuts import PromptSession

        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = PromptSession(
                input=pipe,
                output=DummyOutput(),
                key_bindings=extra_bindings or _prompt_session_bindings(),
                multiline=True,
            )
            try:
                return session.prompt()
            except EOFError:
                return None

    def test_plain_enter_submits(self):
        self.assertEqual(self._run("hi\r"), "hi")

    def test_every_modified_enter_inserts_a_newline(self):
        """Including Ctrl+J, which the enter binding used to swallow."""
        for sequence in (
            SHIFT_ENTER, CTRL_ENTER, ALT_ENTER, "\x1b[13;2u", "\x1b[13;5u",
            "\x1b\r", "\x1b[27;2;13~", "\n",
        ):
            self.assertEqual(self._run(f"a{sequence}b\r"), "a\nb", repr(sequence))

    def test_ctrl_c_csi_u_reaches_the_c_c_binding(self):
        """The prompt's clear/double-tap-quit binding must still fire."""
        from prompt_toolkit.key_binding import KeyBindings

        fired: list[str] = []
        kb = _prompt_session_bindings()

        @kb.add("c-c")
        def _(event):
            fired.append(event.data or "")
            event.app.exit()

        self._run(f"abc{CTRL_C_CSI_U}", extra_bindings=kb)
        self.assertEqual(fired, [CTRL_C_CSI_U])


if __name__ == "__main__":
    unittest.main()
