"""Tests for agent/steer.py: mid-turn steering input."""

import os
import unittest

from agent.steer import SteerInput


class TestSteerInputBuffer(unittest.TestCase):
    """The line buffer is exercised directly, without a tty."""

    def setUp(self) -> None:
        self.submitted: list[str] = []
        self.changes = 0
        self.steer = SteerInput(
            on_submit=self.submitted.append,
            on_change=self._change,
        )

    def _change(self) -> None:
        self.changes += 1

    def _type(self, text: str) -> None:
        for char in text:
            self.steer._handle_byte(ord(char))

    def _enter(self) -> None:
        self.steer._handle_byte(0x0D)

    # typing

    def test_typing_builds_line(self):
        self._type("use pandas")
        self.assertEqual(self.steer.line(), "use pandas")
        self.assertEqual(self.changes, len("use pandas"))

    def test_enter_submits_stripped_line(self):
        self._type("  hello  ")
        self._enter()
        self.assertEqual(self.submitted, ["hello"])
        self.assertEqual(self.steer.line(), "")

    def test_empty_enter_submits_nothing(self):
        self._enter()
        self.assertEqual(self.submitted, [])

    def test_backspace_deletes_last_char(self):
        self._type("ab")
        self.steer._handle_byte(0x7F)
        self.assertEqual(self.steer.line(), "a")
        self.steer._handle_byte(0x08)
        self.assertEqual(self.steer.line(), "")

    def test_backspace_on_empty_line_is_a_noop(self):
        before = self.changes
        self.steer._handle_byte(0x7F)
        self.assertEqual(self.changes, before)

    def test_ctrl_u_clears_line(self):
        self._type("discard me")
        self.steer._handle_byte(0x15)
        self.assertEqual(self.steer.line(), "")

    def test_line_length_is_capped(self):
        steer = SteerInput(on_submit=lambda _t: None, max_len=4)
        for char in "abcdefgh":
            steer._handle_byte(ord(char))
        self.assertEqual(steer.line(), "abcd")

    def test_control_characters_are_ignored(self):
        self._type("ab")
        for byte in (0x01, 0x07, 0x0C, 0x1A):
            self.steer._handle_byte(byte)
        self.assertEqual(self.steer.line(), "ab")

    # Ctrl+C is delivered as SIGINT by the kernel (ISIG stays enabled), so
    # the reader never sees a 0x03 byte. If one ever arrives anyway, it must
    # not be inserted into the line.
    def test_ctrl_c_byte_is_never_buffered(self):
        self._type("ab")
        self.steer._handle_byte(0x03)
        self.assertEqual(self.steer.line(), "ab")

    # escape sequences

    def test_escape_sequences_do_not_leak_into_the_line(self):
        # Up arrow: ESC [ A
        for byte in b"\x1b[A":
            self.steer._handle_byte(byte)
        self.assertEqual(self.steer.line(), "")

    def test_sixteen_escape_sequences_do_not_leak(self):
        for byte in b"\x1b[1;5C":
            self.steer._handle_byte(byte)
        self.assertEqual(self.steer.line(), "")

    # queue

    def test_queue_is_fifo(self):
        self._steer("first")
        self._steer("second")
        self.assertEqual(self.steer.pending(), "first")
        self.assertEqual(self.steer.take(), "first")
        self.assertEqual(self.steer.take(), "second")
        self.assertIsNone(self.steer.take())

    def _steer(self, text: str) -> None:
        self._type(text)
        self._enter()

    def test_callback_errors_do_not_kill_the_reader(self):
        def boom(_line: str) -> None:
            raise RuntimeError("nope")

        steer = SteerInput(on_submit=boom)
        steer._handle_byte(ord("x"))
        steer._handle_byte(0x0D)  # must not raise

    def test_stop_clears_the_partial_line_but_keeps_the_queue(self):
        self._steer("queued line")
        self._type("half")
        steer = self.steer
        steer._armed = True
        steer.stop()
        self.assertEqual(steer.line(), "")
        self.assertEqual(steer.pending(), "queued line")


def _cooked_attrs(attrs):
    """Return *attrs* with the tty forced back to cooked, echoing mode."""
    import termios

    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = (
        attrs[0], attrs[1], attrs[2], attrs[3], attrs[4], attrs[5], list(attrs[6])
    )
    lflag |= termios.ICANON | termios.ECHO | termios.ISIG
    cc[termios.VMIN] = 1
    cc[termios.VTIME] = 0
    return [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]


class TestSteerCbreakMode(unittest.TestCase):
    """The tty must keep ISIG enabled so Ctrl+C still reaches the turn."""

    def test_cbreak_keeps_isig(self):
        """The cbreak flag mask must clear ICANON/ECHO but not ISIG."""
        import termios

        fd = os.open("/dev/tty", os.O_RDWR) if os.path.exists("/dev/tty") else None
        if fd is None or not os.isatty(fd):
            if fd is not None:
                os.close(fd)
            self.skipTest("no controlling tty")
        try:
            pristine = termios.tcgetattr(fd)
            # Start from a known cooked state, whatever the test runner is in.
            termios.tcsetattr(fd, termios.TCSANOW, _cooked_attrs(pristine))
            steer = SteerInput(on_submit=lambda _t: None)
            steer._fd = fd
            self.assertTrue(steer._enter_cbreak(fd))
            lflag = termios.tcgetattr(fd)[3]
            self.assertFalse(lflag & termios.ICANON, "ICANON must be off")
            self.assertFalse(lflag & termios.ECHO, "ECHO must be off")
            self.assertTrue(lflag & termios.ISIG, "ISIG must stay on for Ctrl+C")
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, pristine)
            os.close(fd)

    def test_ctrl_c_is_delivered_as_sigint(self):
        """End-to-end: Ctrl+C reaches the main thread as KeyboardInterrupt."""
        import pty
        import select
        import signal
        import struct
        import fcntl
        import termios
        import time

        log = "/tmp/_steer_isig_result"
        if os.path.exists(log):
            os.unlink(log)

        def body() -> None:  # pragma: no cover - child process
            import termios as T

            with open(log, "w") as out:
                steer = SteerInput(on_submit=lambda _t: None)
                armed = steer.start()
                lflag = T.tcgetattr(0)[3]
                out.write("armed=%s isig=%s icanon=%s\n" % (
                    armed,
                    bool(lflag & T.ISIG),
                    bool(lflag & T.ICANON),
                ))
                out.flush()
                try:
                    time.sleep(2.5)
                    out.write("none\n")
                except KeyboardInterrupt:
                    out.write("sigint\n")
                out.flush()
                steer.stop()
            os._exit(0)

        pid, mfd = pty.fork()
        if pid == 0:  # pragma: no cover - child process
            try:
                body()
            finally:
                os._exit(0)
        fcntl.ioctl(mfd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
        time.sleep(0.7)
        os.write(mfd, b"\x03")  # Ctrl+C, in through the master
        deadline = time.time() + 6
        while time.time() < deadline:
            r, _, _ = select.select([mfd], [], [], 0.2)
            if not r:
                continue
            try:
                chunk = os.read(mfd, 65536)
            except OSError:
                break
            if not chunk:
                break
        os.waitpid(pid, 0)

        with open(log) as result:
            text = result.read()
        os.unlink(log)
        self.assertIn("armed=True", text)
        self.assertIn("isig=True", text)
        self.assertIn("icanon=False", text)
        self.assertIn("sigint", text, "Ctrl+C did not reach the main thread")


class TestSteerInputNoTty(unittest.TestCase):
    """Without a terminal, steering must be a silent no-op."""

    def test_start_returns_false_without_a_tty(self):
        steer = SteerInput(on_submit=lambda _t: None)
        if os.isatty(0):
            self.skipTest("running with a real tty")
        self.assertFalse(steer.start())
        self.assertFalse(steer.armed)

    def test_stop_without_start_is_safe(self):
        SteerInput(on_submit=lambda _t: None).stop()

    def test_take_on_empty_queue(self):
        self.assertIsNone(SteerInput(on_submit=lambda _t: None).take())


if __name__ == "__main__":
    unittest.main()


class TestSteerQueue(unittest.TestCase):
    """The FIFO backing mid-turn injection.

    Two consumers care about it: the injection seam, which drains *all*
    pending lines at a tool boundary, and the remote path, which drops its own
    copy of a line only once the server confirms delivery. So ``ack`` must
    remove the first equal line, not merely the oldest.
    """

    def setUp(self) -> None:
        self.steer = SteerInput(on_submit=lambda _t: None)

    def _queue(self, *lines: str) -> None:
        for line in lines:
            self.steer._queue.append(line)

    def test_take_is_fifo(self):
        self._queue("first", "second")
        self.assertEqual(self.steer.take(), "first")
        self.assertEqual(self.steer.take(), "second")
        self.assertIsNone(self.steer.take())

    def test_take_on_empty_queue(self):
        self.assertIsNone(self.steer.take())

    def test_pending_does_not_consume(self):
        self._queue("first")
        self.assertEqual(self.steer.pending(), "first")
        self.assertEqual(self.steer.pending(), "first")
        self.assertEqual(self.steer.take(), "first")

    def test_take_all_drains_everything(self):
        self._queue("a", "b", "c")
        self.assertEqual(self.steer.take_all(), ["a", "b", "c"])
        self.assertEqual(self.steer.take_all(), [])
        self.assertIsNone(self.steer.pending())

    def test_take_all_on_empty_queue(self):
        self.assertEqual(self.steer.take_all(), [])

    def test_take_all_preserves_order(self):
        self._queue("z", "y", "x")
        self.assertEqual(self.steer.take_all(), ["z", "y", "x"])

    def test_ack_removes_the_named_line(self):
        """Delivery is confirmed out of order relative to typing order."""
        self._queue("first", "second")
        self.assertTrue(self.steer.ack("second"))
        self.assertEqual(self.steer.take_all(), ["first"])

    def test_ack_reports_miss(self):
        self._queue("first")
        self.assertFalse(self.steer.ack("never sent"))
        self.assertEqual(self.steer.pending(), "first")

    def test_ack_of_blank_queue(self):
        self.assertFalse(self.steer.ack("anything"))

    def test_ack_only_removes_one_occurrence(self):
        """The same text typed twice is two separate submissions."""
        self._queue("go on", "go on")
        self.assertTrue(self.steer.ack("go on"))
        self.assertEqual(self.steer.take_all(), ["go on"])
