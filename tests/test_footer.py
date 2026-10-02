"""Tests for agent/footer.py: the sticky DECSTBM footer."""

import io
import os
import pty
import re
import sys
from typing import IO, cast
import fcntl
import termios
import struct
import select
import time
import unittest
from unittest import mock

from agent.footer import (
    FOOTER_LINES,
    SPINNER_FRAMES,
    SPINNER_INTERVAL,
    WRITE_LOCK,
    Footer,
    _LockedWriter,
    clip_ansi,
    real_stream,
    visible_len,
)

_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"


class TestVisibleLen(unittest.TestCase):
    def test_plain_text(self):
        self.assertEqual(visible_len("hello"), 5)

    def test_ignores_sgr(self):
        self.assertEqual(visible_len(f"{_BOLD}hello{_RESET}"), 5)

    def test_ignores_cursor_addressing(self):
        self.assertEqual(visible_len("\x1b[55;1H\x1b[2K\x1b[0m"), 0)


class TestClipAnsi(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(clip_ansi("hello", 10), "hello")

    def test_zero_or_negative_width(self):
        self.assertEqual(clip_ansi("hello", 0), "")
        self.assertEqual(clip_ansi("hello", -3), "")

    def test_clips_with_ellipsis(self):
        self.assertEqual(clip_ansi("abcdefgh", 5), "abcd…")

    def test_preserves_styling(self):
        styled = f"{_BOLD}abcdefgh{_RESET}"
        out = clip_ansi(styled, 5)
        self.assertTrue(out.startswith(_BOLD))
        self.assertTrue(out.endswith(_RESET))
        self.assertEqual(visible_len(out), 5)

    def test_width_smaller_than_ellipsis(self):
        self.assertEqual(clip_ansi("abcd", 1), "a")

    def test_styled_text_within_width_is_returned_as_is(self):
        styled = f"{_BOLD}hi{_RESET}"
        self.assertEqual(clip_ansi(styled, 10), styled)


class FakeTTY(io.StringIO):
    """A StringIO that claims to be a terminal."""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:  # pragma: no cover - os.get_terminal_size fallback
        raise OSError("no fileno")


class FakeNotTTY(io.StringIO):
    """A StringIO that is redirected, like piped output."""

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("no fileno")


class TestLockedWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = io.StringIO()
        self.writer = _LockedWriter(self.buffer)

    def test_write_proxies(self):
        self.writer.write("abc")
        self.assertEqual(self.buffer.getvalue(), "abc")

    def test_writelines(self):
        self.writer.writelines(["a", "b"])
        self.assertEqual(self.buffer.getvalue(), "ab")

    def test_proxies_attributes(self):
        self.assertFalse(self.writer.isatty())

    def test_log_receives_repr(self):
        import tempfile

        with tempfile.NamedTemporaryFile("r+", delete=False) as tmp:
            path = tmp.name
        try:
            writer = _LockedWriter(self.buffer, path)
            writer.write("\x1b[1m")
            with open(path) as log:
                self.assertIn("\\x1b", log.read())
        finally:
            os.unlink(path)

    def test_log_failure_is_swallowed(self):
        writer = _LockedWriter(self.buffer, "/nonexistent/dir/x.log")
        writer.write("still written")
        self.assertEqual(self.buffer.getvalue(), "still written")

    def test_isatty_delegates(self):
        writer = _LockedWriter(FakeTTY())
        self.assertTrue(writer.isatty())


class TestFooterGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = sys.stdout
        self.tty = FakeTTY()
        sys.stdout = self.tty

    def tearDown(self) -> None:
        sys.stdout = self.saved

    def _footer(self) -> Footer:
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 24)):
            footer = Footer()
            footer.start()
        return footer

    def test_start_sets_scroll_region(self):
        footer = self._footer()
        self.assertTrue(footer.active)
        out = self.tty.getvalue()
        # Scroll region covers rows 1..(24-4) = 1..20.
        self.assertIn("\x1b[1;20r", out)

    def test_start_installs_locked_stdout(self):
        footer = self._footer()
        self.assertIsInstance(sys.stdout, _LockedWriter)
        footer.stop()
        self.assertIs(sys.stdout, self.tty)

    def test_stop_restores_stdout_and_region(self):
        footer = self._footer()
        footer.stop()
        self.assertFalse(footer.active)
        self.assertIn("\x1b[r", self.tty.getvalue())

    def test_start_is_noop_when_not_a_tty(self):
        plain = FakeNotTTY()
        sys.stdout = plain
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 24)):
            footer = Footer()
            footer.start()
        self.assertFalse(footer.active)
        self.assertEqual(plain.getvalue(), "")
        self.assertIs(sys.stdout, plain)

    def test_start_is_noop_on_small_terminal(self):
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 4)):
            footer = Footer()
            footer.start()
        self.assertFalse(footer.active)
        self.assertEqual(self.tty.getvalue(), "")
        self.assertIs(sys.stdout, self.tty)

    def test_update_is_noop_when_inactive(self):
        footer = Footer()
        footer.update_status("hello")
        self.assertEqual(self.tty.getvalue(), "")


class TestFooterRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = sys.stdout
        self.tty = FakeTTY()
        sys.stdout = self.tty
        self.size = (80, 24)
        # The patch must outlive setUp: update_status() re-reads the size on
        # every redraw, so a mock scoped to start() alone would fall back to
        # the real terminal geometry and make these tests environment-dependent.
        patcher = mock.patch.object(
            Footer, "_read_terminal_size", return_value=self.size
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.footer = Footer()
        self.footer.start()
        self.tty.seek(0)
        self.tty.truncate(0)

    def tearDown(self) -> None:
        self.footer.stop()
        sys.stdout = self.saved

    def test_draws_four_rows(self):
        self.footer.update_status("status", "input")
        out = self.tty.getvalue()
        top = 24 - FOOTER_LINES + 1
        for offset in range(FOOTER_LINES):
            self.assertIn(f"\x1b[{top + offset};1H", out)

    def test_separator_and_hint(self):
        self.footer.update_status("status", "input")
        out = self.tty.getvalue()
        self.assertIn("─" * 80, out)
        self.assertIn("cancel turn", out)

    def test_status_and_input_on_distinct_rows(self):
        self.footer.update_status("STATUS-LINE", "INPUT-LINE")
        out = self.tty.getvalue()
        self.assertIn("STATUS-LINE", out)
        self.assertIn("INPUT-LINE", out)
        # They must not appear on the same physical row.
        self.assertNotIn("STATUS-LINE\x1b[2KINPUT-LINE", out)

    def test_identical_update_is_skipped(self):
        self.footer.update_status("same", "")
        first = self.tty.getvalue()
        self.footer.update_status("same", "")
        self.assertEqual(self.tty.getvalue(), first)

    def test_input_change_triggers_redraw(self):
        self.footer.update_status("s", "")
        first = len(self.tty.getvalue())
        self.footer.update_status("s", "typing")
        self.assertGreater(len(self.tty.getvalue()), first)

    def test_overlong_status_is_clipped(self):
        self.footer.update_status("X" * 200, "")
        out = self.tty.getvalue()
        self.assertIn("…", out)
        # The clipped line must not exceed the terminal width.
        for chunk in out.split("\x1b[2K"):
            if "X" in chunk:
                self.assertLessEqual(chunk.count("X"), 80)

    def test_overlong_input_is_clipped(self):
        self.footer.update_status("s", "Y" * 200)
        out = self.tty.getvalue()
        self.assertIn("…", out)

    def test_resize_is_reread_on_each_redraw(self):
        self.footer._term_width = 200
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(40, 10)):
            self.footer.update_status("status", "input")
        self.assertEqual(self.footer._term_width, 40)
        self.assertEqual(self.footer._term_height, 10)
        out = self.tty.getvalue()
        self.assertIn("\x1b[7;1H", out)  # 10 - 4 + 1 = 7

    def test_resize_to_too_small_keeps_footer_quiet(self):
        with mock.patch.object(Footer, "_read_terminal_size", return_value=(40, 3)):
            self.footer.update_status("status", "input")
        self.assertEqual(self.tty.getvalue(), "")

    def test_cursor_returned_to_scroll_region(self):
        self.footer.update_status("s", "i")
        out = self.tty.getvalue()
        top = 24 - FOOTER_LINES + 1
        self.assertTrue(out.rstrip().endswith(f"\x1b[{top - 1};1H"))


class TestRealStream(unittest.TestCase):
    """The footer must never write through a Rich FileProxy.

    ``console.status()`` wraps ``sys.stdout`` in a ``FileProxy`` which
    *interprets* ANSI escapes: anything written to it is decoded, markup
    parsed and re-rendered, so the footer's cursor-addressing sequences
    would come out as literal ``[24;1H``-style text.
    """

    def setUp(self) -> None:
        self.saved = sys.stdout

    def tearDown(self) -> None:
        sys.stdout = self.saved

    def test_unwraps_file_proxy(self):
        from rich.file_proxy import FileProxy
        from rich.console import Console

        real = FakeTTY()
        sys.stdout = FileProxy(Console(), real)
        self.assertIs(real_stream(), real)

    def test_leaves_locked_writer_in_place(self):
        """Our own lock must still be taken, so it is not unwrapped."""
        real = FakeTTY()
        locked = _LockedWriter(real)
        sys.stdout = locked
        self.assertIs(real_stream(), locked)

    def test_unwraps_locked_writer_under_a_proxy(self):
        """A spinner wrapping the locked writer still resolves to it."""
        from rich.file_proxy import FileProxy
        from rich.console import Console

        real = FakeTTY()
        locked = _LockedWriter(real)
        sys.stdout = FileProxy(Console(), cast(IO[str], locked))
        self.assertIs(real_stream(), locked)

    def test_plain_stream_is_unchanged(self):
        real = FakeTTY()
        sys.stdout = real
        self.assertIs(real_stream(), real)

    def test_self_referential_proxy_terminates(self):
        class Loop:
            @property
            def rich_proxied_file(self):
                return self

        loop = Loop()
        sys.stdout = loop
        self.assertIs(real_stream(), loop)

    def test_footer_escapes_bypass_the_proxy(self):
        """End-to-end: footer output must not be decoded as literal text."""
        import re

        def body() -> None:
            from rich.console import Console

            footer = Footer()
            footer.start()
            status = Console().status("Tool running")
            status.start()
            try:
                # Reachable: a real Spinner prints through this console.
                Console().print("during spinner")
                footer.update_status(" => model", " x")
            finally:
                status.stop()
                footer.stop()
            os._exit(0)

        out = self._pty(body)
        # A control sequence that reached the terminal correctly is ESC '['
        # digits letters. The bug we are guarding against prints the digits
        # and letters as *text*, with the ESC consumed by Rich's decoder.
        text = out.decode("utf-8", "replace")
        leaked = sorted(
            set(re.findall(r"(?<!\x1b)\[\d+(?:;\d+)*[HJKrf]", text))
        )
        self.assertEqual(leaked, [], f"footer control sequences leaked as text: {leaked}")

    def _pty(self, body):
        import fcntl
        import pty
        import select
        import struct
        import termios
        import time

        pid, mfd = pty.fork()
        if pid == 0:  # pragma: no cover - child process
            try:
                body()
            finally:
                os._exit(0)
        fcntl.ioctl(mfd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
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


class TestFooterSpinner(unittest.TestCase):
    """The footer animates its own spinner instead of starting a Rich Live.

    ``console.status()`` builds a ``Live`` display, and ``Live`` replaces
    ``sys.stdout`` with a ``FileProxy`` that *interprets* ANSI escapes. The
    footer's cursor addressing would then be decoded and printed as literal
    text on screen, so the footer draws the animation itself on its transient
    row, under the same write lock as everything else.
    """

    def setUp(self) -> None:
        self.saved = sys.stdout
        self.tty = FakeTTY()
        sys.stdout = self.tty
        # Kept active for the whole test: the spinner thread re-reads the size
        # on every tick, so a mock scoped to start() would let the animation
        # fall back to the real terminal geometry.
        patcher = mock.patch.object(Footer, "_read_terminal_size", return_value=(80, 24))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.footer = Footer()
        self.footer.start()
        self._clear()

    def tearDown(self) -> None:
        self.footer.stop()
        sys.stdout = self.saved

    def _clear(self) -> None:
        self.tty.seek(0)
        self.tty.truncate(0)

    def _wait_for_frames(self, count: int, timeout: float = 3.0) -> str:
        """Let the animation tick, then return everything it wrote."""
        self._clear()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.tty.getvalue().count("\x1b[2K") >= count:
                break
            time.sleep(SPINNER_INTERVAL / 2)
        out = self.tty.getvalue()
        self._clear()
        return out

    # -- lifecycle --

    def test_no_thread_until_a_spinner_is_set(self):
        self.assertIsNone(self.footer._spinner_thread)

    def test_set_spinner_starts_a_thread(self):
        self.footer.set_spinner("⏳ Processing prompt...")
        thread = self.footer._spinner_thread
        self.assertIsNotNone(thread)
        assert thread is not None  # for the type checker
        self.assertTrue(thread.daemon)

    def test_set_spinner_is_idempotent(self):
        """Re-setting the same label must not leak a second thread."""
        self.footer.set_spinner("working")
        first = self.footer._spinner_thread
        self.footer.set_spinner("working")
        self.assertIs(self.footer._spinner_thread, first)

    def test_clearing_stops_the_thread(self):
        self.footer.set_spinner("working")
        thread = self.footer._spinner_thread
        self.footer.set_spinner("")
        self.assertIsNone(self.footer._spinner_thread)
        assert thread is not None
        self.assertFalse(
            thread.is_alive(), "spinner thread outlived set_spinner('')"
        )

    def test_stop_joins_the_thread(self):
        """stop() must not leave a writer redrawing into a torn-down footer."""
        self.footer.set_spinner("working")
        thread = self.footer._spinner_thread
        self.footer.stop()
        assert thread is not None
        self.assertFalse(thread.is_alive())
        self.footer.start()  # keep tearDown happy

    # -- rendering --

    def test_label_is_drawn_immediately(self):
        self.footer.set_spinner("⏳ Processing prompt...")
        self.assertIn("⏳ Processing prompt...", self.tty.getvalue())

    def test_frame_changes_over_time(self):
        """The animation actually animates."""
        self.footer.set_spinner("working")
        self._clear()
        seen = set()
        deadline = time.time() + 3.0
        while time.time() < deadline and len(seen) < 3:
            out = self.tty.getvalue()
            for frame in SPINNER_FRAMES:
                if frame in out:
                    seen.add(frame)
            time.sleep(SPINNER_INTERVAL / 2)
        self.assertGreaterEqual(len(seen), 3, f"only saw frames {seen}")

    def test_typed_input_wins_over_the_spinner(self):
        """A partially typed steering line is never overwritten by a frame."""
        self.footer.set_spinner("working")
        self._clear()
        self.footer.update_status("status", "⤷ half typed")
        drawn = self.tty.getvalue()
        self.assertIn("half typed", drawn)
        # Now let several animation ticks go by: none of them may write to
        # the row, so the typed text stays put and legible.
        self._clear()
        time.sleep(SPINNER_INTERVAL * 5)
        ticks = self.tty.getvalue()
        for frame in SPINNER_FRAMES:
            self.assertNotIn(frame, ticks)
        self.assertNotIn("2K", ticks)

    def test_input_row_returns_to_the_spinner_when_typing_stops(self):
        self.footer.set_spinner("working")
        self.footer.update_status("status", "typed")
        self.footer.update_status("status", "")
        self._clear()
        time.sleep(SPINNER_INTERVAL / 2)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            out = self.tty.getvalue()
            if any(f in out for f in SPINNER_FRAMES):
                break
            time.sleep(SPINNER_INTERVAL / 2)
        self.assertTrue(
            any(f in self.tty.getvalue() for f in SPINNER_FRAMES),
            "spinner did not resume after the input line was cleared",
        )

    def test_clearing_spinner_blanks_the_row(self):
        self.footer.set_spinner("working")
        self.footer.update_status("status", "")
        self._clear()
        self.footer.set_spinner("")
        out = self.tty.getvalue()
        self.assertIn("\x1b[2K", out)
        for frame in SPINNER_FRAMES:
            self.assertNotIn(frame, out)
        self.assertNotIn("working", out)

    def test_spinner_row_is_within_the_reserved_zone(self):
        """Never write outside the footer rows, or the response would scroll."""
        self.footer.set_spinner("working")
        self._clear()
        time.sleep(SPINNER_INTERVAL * 5)
        out = self.tty.getvalue()
        # Every absolute cursor move must land inside the reserved zone:
        # rows 21..24 are the footer rows, row 20 is the bottom of the scroll
        # region where the content cursor belongs.
        for match in re.finditer(r"\x1b\[(\d+);1H", out):
            row = int(match.group(1))
            self.assertTrue(20 <= row <= 24, f"write to row {row} outside the footer")

    def test_no_leaked_escapes_as_text(self):
        self.footer.set_spinner("working")
        time.sleep(SPINNER_INTERVAL * 3)
        self.footer.set_spinner("")
        time.sleep(SPINNER_INTERVAL)
        # Look for a cursor-addressing sequence that reached the stream
        # *without* its leading ESC: that is the signature of the footer's
        # escapes having been decoded and printed as text.
        raw = self.tty.getvalue()
        leaked = re.findall(r"(?<!\x1b)(\[\d+(?:;\d+)*[HJKrf])", raw)
        self.assertEqual(leaked, [], f"escape sequences leaked as text: {leaked}")

    def test_spinner_text_is_clipped_to_the_width(self):
        self.footer.set_spinner("x" * 300)
        self._clear()
        time.sleep(SPINNER_INTERVAL * 3)
        out = self.tty.getvalue()
        self.assertIn("…", out)


class TestWriteLockIsShared(unittest.TestCase):
    def test_lock_is_reentrant(self):
        with WRITE_LOCK:
            with WRITE_LOCK:
                pass


class TestFooterInPty(unittest.TestCase):
    """End-to-end: a real pty, real terminal size, real escape codes."""

    def _run(self, body):
        pid, mfd = pty.fork()
        if pid == 0:  # pragma: no cover - child process
            try:
                body()
            finally:
                os._exit(0)
        fcntl.ioctl(
            mfd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0)
        )
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

    def test_writes_escape_codes_and_restores_termios(self):
        def body() -> None:
            from agent.steer import SteerInput

            with open("/tmp/_footer_pty_result", "w") as res:
                pristine = termios.tcgetattr(0)
                steer = SteerInput()
                steer.start()
                res.write("cbreak=%s\n" % (
                    not termios.tcgetattr(0)[3] & termios.ICANON
                ))
                steer.stop()
                res.write("restored=%s\n" % (
                    termios.tcgetattr(0) == pristine
                ))
                res.close()
                os._exit(0)

        out = self._run(body)
        with open("/tmp/_footer_pty_result") as res:
            result = res.read()
        os.unlink("/tmp/_footer_pty_result")
        self.assertIn("cbreak=True", result)
        self.assertIn("restored=True", result)
        self.assertEqual(out.count(b"\x1b"), 0, "no bytes expected in this probe")

    def test_footer_emits_esc_sequences_on_a_real_tty(self):
        def body() -> None:
            footer = Footer()
            footer.start()
            footer.update_status(" => model", " ⤷ typing")
            footer.stop()
            os._exit(0)

        out = self._run(body)
        self.assertGreater(out.count(b"\x1b"), 0)
        # Every escape must be a complete sequence, never a bare ESC or a
        # truncated one: the last sequence must terminate with a final byte.
        self.assertNotIn(b"\x1b\x1b", out)
        self.assertIn(b"\x1b[r", out)


if __name__ == "__main__":
    unittest.main()
