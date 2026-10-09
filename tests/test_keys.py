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
    is_ctrl_c_sequence,
    is_modified_enter,
    terminal_supports_kitty_protocol,
)

# Kitty-protocol Ctrl+C, as a real terminal sends it: key code 99 ('c') with
# the modifier field encoding ctrl (4) plus the num_lock bit (128), i.e. 1 +
# 4 + 128 = 133. The kernel never sees these bytes, so nothing raises SIGINT
# unless the application handles the sequence itself.
CTRL_C_CSI_U = "\x1b[99;133u"
CTRL_C_CSI_U_NO_LOCK = "\x1b[99;5u"

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


class TestCtrlCSequences(unittest.TestCase):
    """Ctrl+C must survive the kitty keyboard protocol.

    The disambiguate flag makes the terminal report *every* key in the
    ``CSI u`` form, so Ctrl+C no longer arrives as ``0x03``. If the sequence
    is unknown the terminal prints its own escape code into the prompt, and
    SIGINT is never raised at all -- which is what broke cancel.
    """

    def test_the_sequence_is_recognised(self):
        self.assertTrue(is_ctrl_c_sequence(CTRL_C_CSI_U))
        self.assertTrue(is_ctrl_c_sequence(CTRL_C_CSI_U_NO_LOCK))

    def test_a_plain_byte_is_not_a_ctrl_c_sequence(self):
        self.assertFalse(is_ctrl_c_sequence("\x03"))
        self.assertFalse(is_ctrl_c_sequence("c"))

    def test_it_is_mapped_to_c_c(self):
        extend_ansi_sequences()
        from prompt_toolkit.input import ansi_escape_sequences as A
        from prompt_toolkit.keys import Keys

        for sequence in (CTRL_C_CSI_U, CTRL_C_CSI_U_NO_LOCK, "\x1b[99;6u"):
            self.assertEqual(A.ANSI_SEQUENCES.get(sequence), Keys.ControlC)

    def test_the_whole_modifier_range_is_covered(self):
        extend_ansi_sequences()
        from prompt_toolkit.input import ansi_escape_sequences as A
        from prompt_toolkit.keys import Keys

        for mods in range(2, 258):
            sequence = f"\x1b[99;{mods}u"
            self.assertTrue(is_ctrl_c_sequence(sequence), sequence)
            self.assertEqual(A.ANSI_SEQUENCES.get(sequence), Keys.ControlC, sequence)

    def test_unmodified_c_is_not_mapped(self):
        """A modifier value of 1 means "no modifiers" -- plain 'c' stays text.

        A terminal with disambiguation on sends unmodified keys as plain
        bytes, so ``CSI 99 u`` never occurs. Registering it would turn every
        "c" typed in a steering line into a cancel.
        """
        extend_ansi_sequences()
        from prompt_toolkit.input import ansi_escape_sequences as A

        self.assertNotIn("\x1b[99u", A.ANSI_SEQUENCES)
        self.assertFalse(is_ctrl_c_sequence("\x1b[99u"))


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

    def test_the_parser_also_decodes_ctrl_c(self):
        extend_ansi_sequences()
        from prompt_toolkit.input.vt100_parser import Vt100Parser

        seen = []
        parser = Vt100Parser(seen.append)
        parser.feed(CTRL_C_CSI_U)
        parser.flush()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].key.value, "c-c")
        self.assertEqual(seen[0].data, CTRL_C_CSI_U)


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


class TestLegacyKeyEncoding(unittest.TestCase):
    """readline readers need the terminal's *legacy* encodings.

    ``rich.prompt`` reads through the builtin ``input()``, and readline knows
    only ``CSI D``. With the disambiguate flag pushed, Left arrives as
    ``CSI 1;129D`` and is typed into the line as escape characters instead of
    moving the cursor -- so the flag has to come down for the duration.
    """

    def setUp(self):
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

    def test_it_pops_inside_and_pushes_after(self):
        self.keys.enable_kitty_keyboard()
        self.buf.truncate(0)
        self.buf.seek(0)
        inside: list[bool] = []
        with self.keys.legacy_key_encoding():
            inside.append(self.keys.protocol_pushed())
        self.assertEqual(inside, [False])
        self.assertTrue(self.keys.protocol_pushed())

    def test_the_written_bytes_are_balanced(self):
        self.keys.enable_kitty_keyboard()
        self.buf.truncate(0)
        self.buf.seek(0)
        with self.keys.legacy_key_encoding():
            pass
        # Exactly one POP while reading, one PUSH after.
        self.assertEqual(self.buf.getvalue(), self.keys.POP + self.keys.PUSH)

    def test_nothing_is_sent_when_the_flag_was_never_pushed(self):
        # A terminal that does not support the protocol (or no tty at all)
        # must not be sent a POP it never had a PUSH for.
        with self.keys.legacy_key_encoding():
            pass
        self.assertEqual(self.buf.getvalue(), "")

    def test_the_flag_is_restored_after_an_exception(self):
        self.keys.enable_kitty_keyboard()
        with self.assertRaises(RuntimeError):
            with self.keys.legacy_key_encoding():
                raise RuntimeError("boom")
        self.assertTrue(self.keys.protocol_pushed())

    def test_nesting_pops_only_once(self):
        self.keys.enable_kitty_keyboard()
        self.buf.truncate(0)
        self.buf.seek(0)
        with self.keys.legacy_key_encoding():
            with self.keys.legacy_key_encoding():
                self.assertFalse(self.keys.protocol_pushed())
            # The inner exit must not re-push while the outer still holds.
            self.assertFalse(self.keys.protocol_pushed())
        self.assertEqual(self.buf.getvalue(), self.keys.POP + self.keys.PUSH)
        self.assertTrue(self.keys.protocol_pushed())


class TestEnsureReadline(unittest.TestCase):
    """The builtin ``input()`` has no line editor until readline is imported.

    CPython only wires readline into ``input()`` when the module has been
    imported, and it does not import it for a script. Wisemonkey is a script,
    so ``rich.prompt``'s read was a bare canonical read: the kernel echoed
    every byte, and an arrow key typed ``^[[C`` into the line instead of moving
    the cursor.
    """

    def test_it_imports_readline_and_reports_success(self):
        from agent import keys

        with mock.patch.object(keys, "_readline_loaded", False):
            self.assertTrue(keys.ensure_readline())
            # Idempotent: the second call is a no-op, not a re-import.
            self.assertTrue(keys.ensure_readline())

    def test_a_missing_readline_is_not_fatal(self):
        # Windows builds can lack it; the plain behaviour is preferable to an
        # ImportError taking down the REPL.
        from agent import keys

        with mock.patch.object(keys, "_readline_loaded", False), mock.patch.dict(
            "sys.modules", {"readline": None}
        ):
            self.assertFalse(keys.ensure_readline())


class TestMarkPromptNonprinting(unittest.TestCase):
    """readline counts prompt *characters*; escapes must not count.

    An escape it does not know about is counted as if it were visible, so every
    column readline computes afterwards -- and therefore every redraw -- lands
    in the wrong place.
    """

    def test_sgr_escapes_are_wrapped(self):
        from agent.keys import mark_prompt_nonprinting

        self.assertEqual(
            mark_prompt_nonprinting("\x1b[1mhi\x1b[0m"),
            "\001\x1b[1m\002hi\001\x1b[0m\002",
        )

    def test_plain_text_is_untouched(self):
        from agent.keys import mark_prompt_nonprinting

        self.assertEqual(mark_prompt_nonprinting("hello"), "hello")

    def test_every_escape_family_is_covered(self):
        from agent.keys import mark_prompt_nonprinting

        for escape in ("\x1b[2K", "\x1b[?25l", "\x1b]8;;http://x\x1b\\", "\x1bM"):
            with self.subTest(escape=escape):
                self.assertEqual(
                    mark_prompt_nonprinting(f"a{escape}b"), f"a\001{escape}\002b"
                )


class TestReadlinePrompt(unittest.TestCase):
    """The prompt string handed to ``input()`` must be styled but unwrapped."""

    def _console(self, *, terminal: bool):
        import io

        from rich.console import Console

        return Console(
            file=io.StringIO(),
            force_terminal=terminal,
            color_system="truecolor" if terminal else None,
            width=40,
        )

    def _text(self):
        from rich.text import Text

        return Text.from_markup("[bold]Enter the endpoint URL[/bold]: ")

    def test_a_terminal_gets_the_styling_with_escapes_marked(self):
        from agent.keys import readline_prompt

        out = readline_prompt(self._console(terminal=True), self._text())
        self.assertIn("\001\x1b[1m\002Enter the endpoint URL", out)
        self.assertIn("\001\x1b[0m\002: ", out)

    def test_a_long_prompt_is_not_wrapped_by_rich(self):
        # Rich would break it at the console width; an embedded newline becomes
        # a character in readline's prompt and permanently skews its arithmetic.
        from agent.keys import readline_prompt
        from rich.text import Text

        out = readline_prompt(self._console(terminal=True), Text("x" * 200))
        self.assertNotIn("\n", out)
        self.assertEqual(out.count("x"), 200)

    def test_a_pipe_gets_plain_text_with_no_markers(self):
        # input() prints a prompt verbatim when readline is not in play, so the
        # raw SOH/STX bytes would otherwise reach the output.
        from agent.keys import readline_prompt

        out = readline_prompt(self._console(terminal=False), self._text())
        self.assertEqual(out, "Enter the endpoint URL: ")
        self.assertNotIn("\001", out)
        self.assertNotIn("\002", out)
        self.assertNotIn("\x1b", out)


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


class TestFunctionalKeysUnderDisambiguate(unittest.TestCase):
    """Every functional key is re-encoded by the flag, not just Enter/Ctrl+C.

    With disambiguate on, *all* non-text keys carry a modifier field, so a
    plain arrow arrives as ``CSI 1;129 A`` once num_lock is active instead of
    ``CSI A``. prompt_toolkit's table has no entry with a lock modifier set, so
    each key arrived unknown and its own escape code was typed into the prompt
    -- the reported ``[1;129A[1;129B...`` in the prompt line.
    """

    def _decoded(self, sequence: str) -> list[Any]:
        """The keys prompt_toolkit's parser produces for *sequence*."""
        from prompt_toolkit.input.vt100_parser import Vt100Parser

        seen: list[Any] = []
        parser = Vt100Parser(seen.append)
        parser.feed(sequence)
        parser.flush()
        return [getattr(k.key, "value", k.key) for k in seen]

    def test_the_arrows_move_the_cursor(self):
        extend_ansi_sequences()
        # modifier field -> expected key name, for the Up arrow (CSI ... A).
        # 129 is num_lock alone, 130 num_lock+shift: the lock bit is keyboard
        # *state*, not a key the user is holding, so it does not make the arrow
        # a shifted one.
        for mods, key in (("129", "up"), ("130", "s-up"), ("2", "s-up"),
                          ("5", "c-up"), ("133", "c-up")):
            with self.subTest(mods=mods, key=key):
                self.assertEqual(self._decoded(f"\x1b[1;{mods}A"), [key])
        for mods, key in (("129", "down"), ("5", "c-down")):
            with self.subTest(mods=mods, key=key):
                self.assertEqual(self._decoded(f"\x1b[1;{mods}B"), [key])
        for mods, key in (("129", "right"), ("2", "s-right"), ("5", "c-right")):
            with self.subTest(mods=mods, key=key):
                self.assertEqual(self._decoded(f"\x1b[1;{mods}C"), [key])
        for mods, key in (("129", "left"), ("2", "s-left"), ("5", "c-left")):
            with self.subTest(mods=mods, key=key):
                self.assertEqual(self._decoded(f"\x1b[1;{mods}D"), [key])

    def test_alt_arrow_still_moves_the_cursor(self):
        """``CSI 1;3D`` is Alt+Left, and prompt_toolkit already decodes it.

        The bytes are the same under either encoding, and prompt_toolkit reads
        them as the legacy ``escape`` prefix followed by ``Left`` so that an
        ``escape left`` binding works. That is left alone: the terminal here is
        not using the protocol, and overriding it would break Alt+Left there.
        The upshot is the same either way -- the cursor moves.
        """
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[1;3D"), ["escape", "left"])

    def test_home_end_delete(self):
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[1;129H"), ["home"])
        self.assertEqual(self._decoded("\x1b[1;129F"), ["end"])
        self.assertEqual(self._decoded("\x1b[3;129~"), ["delete"])
        self.assertEqual(self._decoded("\x1b[5;129~"), ["pageup"])

    def test_modified_backspace_deletes(self):
        """Shift+Backspace arrived as ``CSI 127 ; 130 u`` and typed itself.

        There is no ``s-backspace`` key in prompt_toolkit, and none is needed:
        every modifier variant should delete backwards exactly like the
        unmodified key.
        """
        extend_ansi_sequences()
        for mods in ("130", "2", "5", "6", "133"):
            with self.subTest(mods=mods):
                self.assertEqual(self._decoded(f"\x1b[127;{mods}u"), ["c-h"])

    def test_shift_tab_is_back_tab(self):
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[9;130u"), ["s-tab"])

    def test_xterm_modify_other_keys_with_a_lock_modifier(self):
        """xterm reports ``27 ; mods ; 13 ~``; with num_lock it is 130, not 2."""
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[27;130;13~"), ["c-m"])

    def test_the_keypad_navigates_and_types(self):
        """Keypad arrows move the cursor; keypad digits arrive as text."""
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[57419;129u"), ["up"])
        self.assertEqual(self._decoded("\x1b[57420;129u"), ["down"])
        self.assertEqual(self._decoded("\x1b[57423;129u"), ["home"])
        self.assertEqual(self._decoded("\x1b[57399;129u"), ["0"])
        self.assertEqual(self._decoded("\x1b[57413;129u"), ["+"])

    def test_keypad_enter_is_enter(self):
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[57414;129u"), ["c-m"])

    def test_function_keys_above_twelve(self):
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[57376;129u"), ["f13"])
        self.assertEqual(self._decoded("\x1b[57386;129u"), ["f23"])

    def test_a_modifier_nothing_is_named_for_still_works(self):
        """Alt+Arrow is not a key prompt_toolkit has a name for.

        It decodes as the legacy ``escape`` prefix plus the unmodified key,
        which is prompt_toolkit's own mapping -- so an ``escape left`` binding
        still applies and the cursor moves. Our registration deliberately does
        not shadow it: the same bytes mean Alt+Left on a terminal using the
        protocol and on one that is not, and the existing mapping is the right
        one for the second case.
        """
        extend_ansi_sequences()
        self.assertEqual(self._decoded("\x1b[1;3D"), ["escape", "left"])
        self.assertEqual(self._decoded("\x1b[1;3A"), ["escape", "up"])

    def test_the_bare_encodings_are_not_registered(self):
        """Modifier value 1 means *no modifiers*: the legacy encoding.

        Registering ``CSI 99 u`` or ``CSI 13 u`` would turn the letter "c" into
        a cancel and the letter "a" into an arrow. A disambiguating terminal
        never sends them, and prompt_toolkit's own table already covers the
        legacy sequences, so they must stay absent from ours.
        """
        extend_ansi_sequences()
        from agent.keys import extended_sequences

        added = extended_sequences()
        for legacy in ("\x1b[99u", "\x1b[13u", "\x1b[1A", "\x1b[1;1A", "\x1b[127u"):
            self.assertNotIn(legacy, added)

    def test_everything_is_registered_for_the_whole_modifier_range(self):
        """One lock state must not be special-cased at the cost of another.

        caps_lock (64) and num_lock (128) combine, and a terminal can add
        hyper/meta, so the field is open-ended. A miss is not a degraded
        feature -- it is a literal escape code in the prompt, which is what
        this reported for every arrow key.
        """
        extend_ansi_sequences()
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys

        def names(value):
            if isinstance(value, tuple):
                return tuple(names(v) for v in value)
            return getattr(value, "value", value)

        for mods in range(2, 258):
            for sequence in (
                f"\x1b[1;{mods}A",
                f"\x1b[1;{mods}D",
                f"\x1b[3;{mods}~",
                f"\x1b[127;{mods}u",
                f"\x1b[57419;{mods}u",
                f"\x1b[57376;{mods}u",
                f"\x1b[27;{mods};13~",
            ):
                with self.subTest(mods=mods, sequence=sequence):
                    # Registered either by us or already upstream: either way
                    # the parser must consume the whole sequence, never let
                    # part of it reach the buffer as text.
                    self.assertIn(sequence, ANSI_SEQUENCES, sequence)
                    decoded = self._decoded(sequence)
                    self.assertTrue(decoded, sequence)
                    for key in decoded:
                        self.assertNotEqual(key, sequence)
                        # A leftover fragment: digits and ';' are what a
                        # partially consumed escape sequence leaks.
                        self.assertNotIn(key, "0123456789;")

    def test_an_upstream_mapping_is_never_lost(self):
        """The table is extended, not rewritten.

        The legacy Alt spellings (``CSI 1;3D`` -> ``escape``, ``left``) are
        what a terminal *not* using the protocol sends, and an ``escape left``
        binding depends on them. Where upstream and this protocol genuinely
        disagree (``CSI 1;2P`` is Shift+F1 here, F13 under the xterm
        convention) upstream wins: both are function keys nothing binds, so
        nothing is lost, while shadowing would be a real regression.
        """
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys

        before = {
            k: v
            for k, v in ANSI_SEQUENCES.items()
            if isinstance(v, Keys) and not k.startswith("\x1b[1;2")
        }
        extend_ansi_sequences()
        for sequence, key in before.items():
            with self.subTest(sequence=sequence):
                self.assertEqual(ANSI_SEQUENCES.get(sequence), key)

    def test_the_arrow_keys_reach_the_prompt(self):
        """End to end: the cursor moves instead of the sequence appearing."""
        extend_ansi_sequences()
        from prompt_toolkit.buffer import Buffer

        buffer = Buffer()
        buffer.insert_text("hello world")
        buffer.cursor_position = 0
        for sequence, expected in (
            ("\x1b[1;129C", 1),
            ("\x1b[1;129C", 2),
            ("\x1b[1;129D", 1),
        ):
            for key in self._decoded(sequence):
                if key == "right":
                    buffer.cursor_position += 1
                elif key == "left":
                    buffer.cursor_position -= 1
            self.assertEqual(buffer.cursor_position, expected)
        self.assertEqual(buffer.text, "hello world")


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

    def test_ctrl_c_csi_u_reaches_the_c_c_binding(self):
        """The prompt's clear/double-tap-quit binding must still fire.

        With the disambiguate flag on, Ctrl+C is ``CSI 99 ; mods u``. Before
        this was mapped, the sequence was unknown, so the binding never ran
        and the escape code itself was typed into the buffer.
        """
        fired: list[str] = []

        def _run_with_c_c(keys: str) -> list[str]:
            from prompt_toolkit.input import create_pipe_input
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.output import DummyOutput
            from prompt_toolkit.shortcuts import PromptSession

            kb = _prompt_session_bindings()

            @kb.add("c-c")
            def _(event):
                fired.append(event.data or "")
                event.app.exit()

            with create_pipe_input() as pipe:
                pipe.send_text(keys)
                session = PromptSession(
                    input=pipe, output=DummyOutput(), key_bindings=kb, multiline=True
                )
                try:
                    session.prompt()
                except EOFError:
                    pass
            return fired

        fired.clear()
        _run_with_c_c(f"abc{CTRL_C_CSI_U}")
        self.assertEqual(fired, [CTRL_C_CSI_U])


if __name__ == "__main__":
    unittest.main()