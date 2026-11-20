"""Tests for agent/footer.py: the sticky DECSTBM footer."""

import fcntl
import io
import os
import pty
import re
import select
import struct
import sys
import termios
import time
import traceback
import unittest
from typing import IO, cast
from unittest import mock

from agent.footer import (
    FOOTER_LINES,
    SPINNER_FRAMES,
    SPINNER_INTERVAL,
    THINKING_LABEL,
    Footer,
    _CURSOR,
    _LockedWriter,
    block_cursor,
    clip_ansi,
    real_stream,
    steer_window,
    visible_len,
)
from agent.palette import PALETTE

_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"


class FakeTTY(io.StringIO):
    """A StringIO that claims to be a terminal."""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:  # pragma: no cover - os.get_terminal_size fallback
        raise OSError("no fileno")


class FakeNotTTY(FakeTTY):
    """A redirected stream, like piped output."""

    def isatty(self) -> bool:
        return False


def _row(out: str, row: int) -> str:
    """The content the footer last wrote to physical *row* (1-based)."""
    marker = f"\x1b[{row};1H\x1b[2K"
    idx = out.rfind(marker)
    if idx < 0:
        return ""
    rest = out[idx + len(marker):]
    m = re.search(r"\x1b\[\d+;1H|\x1b8", rest)
    return rest[: m.start()] if m else rest


def _run_in_pty(body, rows: int = 24, columns: int = 100) -> bytes:
    """Run *body* in a child on a real pty of the given size, return its output."""
    pid, mfd = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        # Set the size here too: the child may get there first and read a fresh
        # pty as 0x0, which disarms the footer.
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        try:
            body()
        except BaseException:
            # os._exit(0) would swallow the traceback, and the failure would
            # read as "the footer wrote nothing".
            traceback.print_exc()
            raise
        finally:
            os._exit(0)
    fcntl.ioctl(mfd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
    out = b""
    deadline = time.time() + 5
    while time.time() < deadline:
        r, _, _ = select.select([mfd], [], [], 0.2)
        if not r:
            if out:
                break
            continue
        try:
            chunk = os.read(mfd, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    os.waitpid(pid, 0)
    return out


class TestAnsiText(unittest.TestCase):
    def test_visible_len_and_clip_ansi(self):
        self.assertEqual(visible_len("hello"), 5)
        self.assertEqual(visible_len(f"{_BOLD}hello{_RESET}"), 5)
        self.assertEqual(visible_len("\x1b[55;1H\x1b[2K\x1b[0m"), 0)

        styled = f"{_BOLD}hi{_RESET}"
        self.assertEqual(clip_ansi("hello", 10), "hello")
        self.assertEqual(clip_ansi(styled, 10), styled)
        # A zero or negative budget is nothing to draw, not a crash.
        for width in (0, -3):
            self.assertEqual(clip_ansi("hello", width), "")
        self.assertEqual(clip_ansi("abcdefgh", 5), "abcd…")
        self.assertEqual(clip_ansi("abcd", 1), "a")
        # Styling survives and the result is exactly the budget.
        out = clip_ansi(f"{_BOLD}abcdefgh{_RESET}", 5)
        self.assertTrue(out.startswith(_BOLD) and out.endswith(_RESET))
        self.assertEqual(visible_len(out), 5)


class TestSteerWindow(unittest.TestCase):
    """Clipping the steering row without losing the caret."""

    def test_the_caret_never_falls_outside_the_window(self):
        """The invariant, swept over every position on a long line."""
        line = "".join(str(i % 10) for i in range(120))
        for avail in (4, 6, 20, 41, 80):
            for caret in range(len(line) + 1):
                text, index = steer_window(line, caret, avail)
                self.assertLessEqual(
                    visible_len(text) + 1, avail, f"avail={avail} caret={caret}"
                )
                self.assertGreaterEqual(index, 0, caret)
                self.assertLessEqual(index, len(text), (avail, caret, text))
                # What the caret sits on is what it was pointing at.
                if index < len(text) and not text[index:].startswith("…"):
                    self.assertEqual(text[index], line[caret], (avail, caret))

    def test_what_survives_is_the_text_before_the_caret(self):
        line = "a" * 50 + "CARET" + "b" * 50
        text, caret = steer_window(line, 53, 20)
        self.assertTrue(text.startswith("…") and text.endswith("…"))
        self.assertIn("CAR", text[:caret])
        self.assertNotIn("b", text)  # what a tail-anchored clip would keep
        # At the end of the line, the tail is what survives.
        text, caret = steer_window("x" * 200, 200, 20)
        self.assertTrue(text.startswith("…"))
        self.assertEqual(caret, len(text))
        self.assertLessEqual(visible_len(text) + 1, 20)
        # At the start, the head.
        text, caret = steer_window("y" * 200, 0, 20)
        self.assertFalse(text.startswith("…"), "nothing was cut at the front")
        self.assertTrue(text.endswith("…"))
        self.assertEqual((caret, visible_len(text) + 1), (0, 20))

    def test_degenerate_cases(self):
        self.assertEqual(steer_window("use pandas", 3, 40), ("use pandas", 3))
        self.assertEqual(steer_window("", 0, 0), ("", 0))
        self.assertEqual(steer_window("", 0, 20), ("", 0))
        # A stale index (the line shrank under it) must not slice wildly.
        self.assertEqual(steer_window("abc", 99, 20), ("abc", 3))


class TestBlockCursor(unittest.TestCase):
    """The caret cell: one column, painted over what it points at."""

    def test_the_cell_is_one_column_and_keeps_the_character_inside_it(self):
        sgr = PALETTE.footer_sgr()
        for under in ("", "a", " ", "€", "rest"):
            self.assertEqual(visible_len(block_cursor(under)), 1, repr(under))
        cell = block_cursor("q")
        self.assertIn("q", cell)
        self.assertIn(sgr["cursor_over"], cell)
        # Otherwise reverse video would leak into the rest of the row.
        self.assertTrue(cell.endswith(sgr["reset"]))
        # Past the end there is no character, so the cell is a filled block.
        self.assertEqual(block_cursor(""), _CURSOR)
        self.assertIn(sgr["cursor"], block_cursor(""))


class TestLockedWriter(unittest.TestCase):
    def test_it_proxies_and_a_broken_debug_log_never_breaks_the_write(self):
        import tempfile

        buffer = io.StringIO()
        writer = _LockedWriter(buffer)
        writer.write("abc")
        writer.writelines(["d", "e"])
        self.assertEqual(buffer.getvalue(), "abcde")
        self.assertTrue(_LockedWriter(FakeTTY()).isatty())
        self.assertFalse(writer.isatty())

        with tempfile.NamedTemporaryFile("r+", delete=False) as tmp:
            path = tmp.name
        try:
            _LockedWriter(buffer, path).write("\x1b[1m")
            with open(path) as log:
                self.assertIn("\\x1b", log.read())  # repr'd bytes
        finally:
            os.unlink(path)
        buffer = io.StringIO()
        _LockedWriter(buffer, "/nonexistent/dir/x.log").write("still written")
        self.assertEqual(buffer.getvalue(), "still written")


class _FooterCase(unittest.TestCase):
    """A footer on a fake tty, with the size mocked for the test's lifetime."""

    SIZE = (80, 24)

    def setUp(self) -> None:
        self.saved = sys.stdout
        self.tty = FakeTTY()
        sys.stdout = self.tty
        # Must outlive start(): stop() and update_status() re-read the size.
        # Tests that resize nest their own patch over this one, which wins.
        patcher = mock.patch.object(Footer, "_read_terminal_size", return_value=self.SIZE)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.footer = Footer()

    def tearDown(self) -> None:
        sys.stdout = self.saved

    def _clear(self) -> None:
        self.tty.seek(0)
        self.tty.truncate(0)


class TestFooterGeometry(_FooterCase):
    def test_start_sets_the_region_and_stop_restores_stdout(self):
        self.footer.start()
        out = self.tty.getvalue()
        self.assertTrue(self.footer.active)
        self.assertIn("\x1b[1;20r", out)  # rows 1..(24-4)
        self.assertIsInstance(sys.stdout, _LockedWriter)
        self._clear()
        self.footer.stop()
        out = self.tty.getvalue()
        self.assertIs(sys.stdout, self.tty)
        self.assertFalse(self.footer.active)
        self.assertIn("\x1b[r", out)
        # Blanking leaves the cursor on the last screen row, not below it.
        self.assertTrue(out.rstrip().endswith(f"\x1b[{self.footer._term_height};1H"))

        # `\x1b[2K\r\n` per row would scroll the screen; addressed erases cannot.
        self._clear()
        self.footer._blank_footer_rows(self.tty)
        out = self.tty.getvalue()
        self.assertNotIn("\n", out)
        self.assertNotIn("\r", out)
        top = self.footer._term_height - FOOTER_LINES + 1
        for offset in range(FOOTER_LINES):
            self.assertIn(f"\x1b[{top + offset};1H\x1b[2K", out)

    def test_stop_follows_a_resize_that_happened_during_the_turn(self):
        self.footer.start()
        self._clear()
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 30)):
            self.footer.stop()
        out = self.tty.getvalue()
        # Rows 27..30 are the footer zone at the new height.
        for row in range(27, 31):
            self.assertIn(f"\x1b[{row};1H\x1b[2K", out)
        self.assertTrue(out.rstrip().endswith("\x1b[30;1H"))
        self.assertEqual(self.footer._term_height, 30)

    def test_start_is_a_noop_off_a_tty_and_on_a_small_terminal(self):
        self.tearDown()
        plain = FakeNotTTY()
        sys.stdout = plain
        footer = Footer()
        footer.start()
        self.assertFalse(footer.active)
        self.assertEqual(plain.getvalue(), "")
        self.assertIs(sys.stdout, plain)

        sys.stdout = self.tty
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 4)):
            self.footer.start()
            self.assertFalse(self.footer.active)
            self.assertEqual(self.tty.getvalue(), "")
            self.assertIs(sys.stdout, self.tty)
            self.footer.update_status("hello")  # an unarmed footer never writes
            self.assertEqual(self.tty.getvalue(), "")
        self.tearDown()
        sys.stdout = self.saved


class TestFooterRendering(_FooterCase):
    def setUp(self) -> None:
        super().setUp()
        self.footer.start()
        self._clear()

    def tearDown(self) -> None:
        self.footer.stop()
        super().tearDown()

    def test_four_rows_with_a_separator_a_hint_and_no_wrapping(self):
        self.footer.update_status("STATUS-LINE", "INPUT-LINE")
        out = self.tty.getvalue()
        top = self.SIZE[1] - FOOTER_LINES + 1
        for offset in range(FOOTER_LINES):
            self.assertIn(f"\x1b[{top + offset};1H", out)
        # Status and input sit on distinct rows.
        self.assertNotIn("STATUS-LINE\x1b[2KINPUT-LINE", out)
        self.assertIn("─" * 80, out)
        self.assertIn("cancel turn", out)

        self._clear()
        self.footer.update_status("X" * 200, "Y" * 200)
        out = self.tty.getvalue()
        self.assertIn("…", out)
        for chunk in out.split("\x1b[2K"):
            self.assertLessEqual(chunk.count("X"), 80)
            self.assertLessEqual(chunk.count("Y"), 80)

    def test_a_redraw_is_saved_and_restored_and_only_a_change_redraws(self):
        """A redraw must not clobber the streamed cursor column."""
        self.footer.update_status("s", "i")
        out = self.tty.getvalue()
        self.assertTrue(out.startswith("\x1b7"))
        self.assertTrue(out.rstrip().endswith("\x1b8"))

        self._clear()
        self.footer.update_status("same", "")
        first = self.tty.getvalue()
        self.footer.update_status("same", "")
        self.assertEqual(self.tty.getvalue(), first)
        self.footer.update_status("same", "typing")
        self.assertGreater(len(self.tty.getvalue()), len(first))
        # A resize forces one even when nothing changed.
        before = len(self.tty.getvalue())
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 30)):
            self.footer.update_status("same", "typing")
        self.assertGreater(len(self.tty.getvalue()), before)

    def test_a_grow_reapplies_the_region_a_shrink_reanchors_the_cursor(self):
        """The terminal keeps the old margins across a resize; we must not."""
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 30)):
            self.footer.update_status("s", "i")
        out = self.tty.getvalue()
        self.assertIn("\x1b7\x1b[r\x1b[1;26r\x1b8", out)  # 30 - 4, then DECRC
        self.assertNotIn("\r\n", out)  # growing only adds rows: the cursor is safe
        self.assertEqual(self.footer._term_height, 30)
        self.assertIn("\x1b[27;1H", out)  # the reserved rows moved

        # A shrink cannot restore the cursor: it parks on the last content row.
        self._clear()
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 12)):
            self.footer.update_status("s", "i")
        out = self.tty.getvalue()
        self.assertIn("\x1b[1;8r", out)  # 12 - 4
        self.assertIn("\x1b[8;1H\r\n", out)
        self.assertLess(out.index("\x1b[8;1H\r\n"), out.index("\x1b7"))

        # A width change cannot either, even at the same height.
        self._clear()
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(40, 12)):
            self.footer.update_status("s", "i")
        self.assertIn("\x1b[8;1H\r\n", self.tty.getvalue())
        self.assertEqual(self.footer._term_width, 40)

    def test_a_resize_to_too_small_releases_the_region(self):
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(40, 3)):
            self.footer.update_status("status", "input")
        out = self.tty.getvalue()
        self.assertIn("\x1b[r", out)
        rows = [int(m.group(1)) for m in re.finditer(r"\x1b\[(\d+);1H", out)]
        self.assertLessEqual(max(rows), 3)
        self.assertNotIn("cancel turn", out)

    def test_output_written_while_armed_repairs_the_geometry_first(self):
        """A tool can print for a minute while the user resizes; the next write
        must not land in the broken layout."""
        self._clear()
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 14)):
            sys.stdout.write("tool output\n")
        out = self.tty.getvalue()
        self.assertIn("\x1b[1;10r", out)
        self.assertLess(out.index("\x1b[1;10r"), out.index("tool output"))


class TestRealStream(unittest.TestCase):
    """The footer must never write through a Rich FileProxy."""

    def setUp(self) -> None:
        self.saved = sys.stdout

    def tearDown(self) -> None:
        sys.stdout = self.saved

    def test_unwraps_a_file_proxy_but_not_our_own_lock(self):
        from rich.console import Console
        from rich.file_proxy import FileProxy

        real = FakeTTY()
        sys.stdout = real
        self.assertIs(real_stream(), real)
        sys.stdout = FileProxy(Console(), real)
        self.assertIs(real_stream(), real)
        # Our lock must still be taken, so it is not unwrapped...
        locked = _LockedWriter(real)
        sys.stdout = locked
        self.assertIs(real_stream(), locked)
        # ...unless a spinner's proxy sits on top of it.
        sys.stdout = FileProxy(Console(), cast(IO[str], locked))
        self.assertIs(real_stream(), locked)

        class Loop:
            @property
            def rich_proxied_file(self):
                return self

        loop = Loop()
        sys.stdout = loop
        self.assertIs(real_stream(), loop)

    def test_footer_escapes_bypass_the_proxy_on_a_real_tty(self):
        """End-to-end: footer output must not be decoded as literal text."""

        def body() -> None:
            from rich.console import Console

            footer = Footer()
            footer.start()
            status = Console().status("Tool running")
            status.start()
            try:
                Console().print("during spinner")
                footer.update_status(" => model", " x")
            finally:
                status.stop()
                footer.stop()
            os._exit(0)

        text = _run_in_pty(body).decode("utf-8", "replace")
        # A correct control sequence is ESC '[' digits letters. The bug prints
        # the digits and letters as text, the ESC eaten by Rich's decoder.
        leaked = sorted(set(re.findall(r"(?<!\x1b)\[\d+(?:;\d+)*[HJKrf]", text)))
        self.assertEqual(leaked, [], f"footer control sequences leaked as text: {leaked}")


class TestFooterSpinner(_FooterCase):
    """The footer animates its own spinner instead of starting a Rich Live."""

    def setUp(self) -> None:
        super().setUp()
        self.footer.start()
        self._clear()

    def tearDown(self) -> None:
        self.footer.stop()
        super().tearDown()

    def test_one_daemon_thread_that_stop_joins(self):
        self.assertIsNone(self.footer._spinner_thread)
        self.footer.set_spinner("⏳ Processing prompt...")
        thread = self.footer._spinner_thread
        assert thread is not None  # for the type checker
        self.assertTrue(thread.daemon)
        self.assertIn("⏳ Processing prompt...", self.tty.getvalue())
        # Re-setting the same label must not leak a second thread.
        self.footer.set_spinner("⏳ Processing prompt...")
        self.assertIs(self.footer._spinner_thread, thread)
        # stop() must not leave a writer redrawing into a torn-down footer.
        self.footer.stop()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(self.footer._spinner_thread)
        self.footer.start()  # keep tearDown happy

    def test_the_label_animates_and_is_clipped(self):
        self.footer.set_spinner("x" * 300)
        seen: set[str] = set()
        deadline = time.time() + 3.0
        while time.time() < deadline and len(seen) < 3:
            out = self.tty.getvalue()
            seen |= {frame for frame in SPINNER_FRAMES if frame in out}
            time.sleep(SPINNER_INTERVAL / 2)
        self.assertGreaterEqual(len(seen), 3, f"only saw frames {seen}")
        self.assertIn("…", self.tty.getvalue())
        # A cursor-addressing sequence without its leading ESC is the signature
        # of the footer's escapes being decoded and printed as text.
        leaked = re.findall(r"(?<!\x1b)(\[\d+(?:;\d+)*[HJKrf])", self.tty.getvalue())
        self.assertEqual(leaked, [], f"escape sequences leaked as text: {leaked}")

    def test_the_indicators_live_on_the_hints_row_only(self):
        """Nothing the spinner writes may touch the typed steering line."""
        self.footer.set_spinner("working")
        self._clear()
        self.footer.update_status("status", "↳ half typed")
        time.sleep(SPINNER_INTERVAL * 5)
        out = self.tty.getvalue()
        transient = _row(out, 23)
        self.assertIn("half typed", transient)
        for frame in SPINNER_FRAMES:
            self.assertNotIn(frame, transient)
        self.assertTrue(any(f in _row(out, 24) for f in SPINNER_FRAMES),
                        "spinner did not animate on the hints row")
        # Every absolute cursor move lands inside the reserved zone: rows 21..24,
        # with row 20 the bottom of the scroll region.
        for match in re.finditer(r"\x1b\[(\d+);1H", out):
            self.assertTrue(20 <= int(match.group(1)) <= 24, match.group(0))

        # Clearing blanks the hints row and keeps the steering row.
        self._clear()
        self.footer.set_spinner("")
        out = self.tty.getvalue()
        self.assertIn("\x1b[2K", out)
        for frame in SPINNER_FRAMES:
            self.assertNotIn(frame, out)
        self.assertNotIn("working", out)
        self.assertIn("half typed", _row(out, 23))


class TestFooterHintsRow(_FooterCase):
    """The bottom row: key bindings on the left, status indicators on the right."""

    def setUp(self) -> None:
        super().setUp()
        self.footer.start()

    def tearDown(self) -> None:
        self.footer.stop()
        super().tearDown()

    def test_reasoning_is_shown_on_the_hints_row_and_animates_alone(self):
        self.footer.set_thinking(True)
        out = self.tty.getvalue()
        hints = _row(out, 24)
        self.assertIn(THINKING_LABEL, hints)
        # It reads like the stage spinner: a frame, then the label.
        self.assertTrue(any(frame in hints for frame in SPINNER_FRAMES), hints)
        self.assertNotIn(THINKING_LABEL, _row(out, 22))
        self.assertNotIn(THINKING_LABEL, _row(out, 23))
        # Reasoning alone drives the animation thread.
        self.assertIsNotNone(self.footer._spinner_thread)
        self._clear()
        seen: set[str] = set()
        deadline = time.time() + 3.0
        while time.time() < deadline and len(seen) < 3:
            seen |= {f for f in SPINNER_FRAMES if f in _row(self.tty.getvalue(), 24)}
            time.sleep(SPINNER_INTERVAL / 2)
        self.assertGreaterEqual(len(seen), 3, f"only saw frames {seen}")

    def test_the_two_indicators_share_the_row_and_each_stops_on_its_own(self):
        self.footer.set_spinner("🔧 Tool running")
        self.footer.set_thinking(True)
        out = self.tty.getvalue()
        hints = _row(out, 24)
        self.assertIn("🔧 Tool running", hints)
        self.assertLess(hints.index("🔧 Tool running"), hints.index(THINKING_LABEL))
        self.assertNotIn("🔧 Tool running", _row(out, 23))

        self.footer.set_spinner("")
        self.assertIsNotNone(self.footer._spinner_thread,
                             "clearing the stage spinner stopped the reasoning")
        hints = _row(self.tty.getvalue(), 24)
        self.assertNotIn("Processing prompt...", hints)
        self.assertIn(THINKING_LABEL, hints)

        thread = self.footer._spinner_thread
        self._clear()
        self.footer.set_thinking(False)
        self.assertIsNone(self.footer._spinner_thread)
        assert thread is not None
        self.assertFalse(thread.is_alive(), "animation outlived the indicator")
        self.assertNotIn(THINKING_LABEL, self.tty.getvalue())

        # A long label is clipped so the row never wraps.
        self.footer.set_spinner("x" * 100)
        self.assertLessEqual(visible_len(_row(self.tty.getvalue(), 24)), 80)

    def test_start_clears_a_leftover_thinking_indicator(self):
        self.footer.set_thinking(True)
        self._clear()
        self.footer.stop()
        self.footer.start()
        self.assertFalse(self.footer._thinking)
        self.assertNotIn(THINKING_LABEL, self.tty.getvalue())


class TestFooterQuestionRow(_FooterCase):
    """A pending question takes the hints row, and says which keys apply."""

    def setUp(self) -> None:
        super().setUp()
        self.footer.start()

    def tearDown(self) -> None:
        self.footer.stop()
        super().tearDown()

    def _plain(self) -> str:
        return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", self.tty.getvalue())

    def test_announcing_a_question_names_it_offers_tab_and_rings_only_once(self):
        self.footer.set_question("Run this command?")
        out = self._plain()
        self.assertIn("Run this command?", out)
        self.assertIn("Tab to answer", out)
        self.assertIn("Esc to leave it", out)
        self.assertEqual(self.tty.getvalue().count("\a"), 1)
        # Re-announcing the same question is not a new one.
        self.footer.set_question("Run this command?")
        self.assertEqual(self.tty.getvalue().count("\a"), 1)

        # With the line focused, the keys are the way back.
        self._clear()
        self.footer.set_question("Run this command?", focused=True)
        out = self._plain()
        self.assertIn("back to the line", out)
        self.assertNotIn("Tab to answer", out)
        self._clear()
        self.footer.set_question_focus(False)
        self.assertIn("Tab to answer", self._plain())
        # Focus without a question does nothing at all.
        self.footer.set_question("")
        self._clear()
        self.footer.set_question_focus(True)
        self.assertFalse(self.footer._question_focused)
        self.assertEqual(self.tty.getvalue(), "")

    def test_the_options_cost_columns_and_are_truncated_to_fit(self):
        self.footer.set_question("cd /very/long/path && " + "x" * 200,
                                 options="[y/N]")
        self.assertIn("…", self._plain())
        self.assertIn("Tab to answer", self._plain())
        row = self.footer._hint_line(80)
        self.assertLessEqual(visible_len(row), 78)
        self.assertIn("[y/N]", row)
        # The options ride with the question and survive a focus change.
        self.footer.set_question_focus(True)
        self.assertIn("[y/N]", self._plain())
        self.footer.set_question("")
        self.assertEqual(self.footer._question_options, "")

    def test_a_new_turn_clears_a_leftover_question(self):
        self.footer.set_question("Run this command?", options="[y/N]", focused=True)
        self.footer.stop()
        self.footer.start()
        self.assertEqual(self.footer._question_text, "")
        self.assertEqual(self.footer._question_options, "")
        self.assertFalse(self.footer._question_focused)


class TestFooterInPty(unittest.TestCase):
    """End-to-end: a real pty, real terminal size, real escape codes."""

    def test_steer_writes_escape_codes_and_restores_termios(self):
        def body() -> None:
            from agent.steer import SteerInput

            # Reported over the pty itself rather than a shared temp file, which
            # is unlinkable mid-run and races under parallel suites.
            pristine = termios.tcgetattr(0)
            steer = SteerInput()
            steer.start()
            cbreak = not termios.tcgetattr(0)[3] & termios.ICANON
            steer.stop()
            restored = termios.tcgetattr(0) == pristine
            sys.stdout.write(f"cbreak={cbreak}\nrestored={restored}\n")
            sys.stdout.flush()
            os._exit(0)

        result = _run_in_pty(body, rows=30).decode(errors="replace")
        self.assertIn("cbreak=True", result, result)
        self.assertIn("restored=True", result, result)
        # The probe itself must emit no escape codes.
        self.assertNotIn("\x1b", result, f"unexpected escapes: {result!r}")

    def test_footer_emits_complete_esc_sequences_on_a_real_tty(self):
        def body() -> None:
            footer = Footer()
            footer.start()
            footer.update_status(" => model", " ↳ typing")
            footer.stop()
            os._exit(0)

        out = _run_in_pty(body, rows=30)
        # Every escape must be a complete sequence, never a bare ESC.
        self.assertNotIn(b"\x1b\x1b", out)
        self.assertIn(b"\x1b[r", out)


if __name__ == "__main__":
    unittest.main()
