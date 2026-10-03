"""Tests for agent/steer.py: mid-turn steering input."""

import os
import sys
import unittest
from typing import TYPE_CHECKING, Any, cast
from unittest import mock

from agent.steer import SteerInput

if TYPE_CHECKING:
    from agent.output import RichOutputAdapter


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


class TestSteerCtrlCFromKittyProtocol(unittest.TestCase):
    """Ctrl+C as the kitty keyboard protocol reports it.

    With disambiguation on, the terminal sends ``CSI 99 ; mods u`` instead of
    ``0x03``. The kernel never sees the byte, so ``ISIG`` has nothing to act
    on and no SIGINT is raised -- the reader thread has to re-raise it, or
    Ctrl+C does nothing for the whole duration of a turn.
    """

    CTRL_C = b"\x1b[99;133u"  # key 'c', ctrl + num_lock

    def _steer(self, interrupts: list[str]) -> SteerInput:
        steer = SteerInput()
        # _interrupt is what raises SIGINT for real; here we only want to
        # observe that it was reached, since there is no tty (and no turn)
        # to interrupt.
        patcher = mock.patch.object(
            SteerInput, "_interrupt", lambda _self: interrupts.append("sigint")
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return steer

    def test_it_raises_an_interrupt(self):
        interrupts: list[str] = []
        steer = self._steer(interrupts)
        for byte in self.CTRL_C:
            steer._handle_byte(byte)
        self.assertEqual(interrupts, ["sigint"])

    def test_the_sequence_does_not_leak_into_the_line(self):
        interrupts: list[str] = []
        steer = self._steer(interrupts)
        for byte in b"ab" + self.CTRL_C + b"cd":
            steer._handle_byte(byte)
        self.assertEqual(steer.line(), "abcd")

    def test_ordinary_control_c_still_goes_through_isig(self):
        """The bare 0x03 is the kernel's business, not the reader's."""
        interrupts: list[str] = []
        steer = self._steer(interrupts)
        for byte in b"ab\x03cd":
            steer._handle_byte(byte)
        self.assertEqual(interrupts, [])
        self.assertEqual(steer.line(), "abcd")

    def test_escape_state_is_left_clean(self):
        interrupts: list[str] = []
        steer = self._steer(interrupts)
        for byte in self.CTRL_C:
            steer._handle_byte(byte)
        self.assertEqual(steer._esc_mode, 0)
        # And typing still works afterwards.
        for byte in b"x":
            steer._handle_byte(byte)
        self.assertEqual(steer.line(), "x")

    def test_other_escape_sequences_do_not_interrupt(self):
        """Arrows and friends must not be mistaken for a cancel."""
        interrupts: list[str] = []
        steer = self._steer(interrupts)
        for keys in (b"\x1b[D", b"\x1b[A", b"\x1b[13;130u", b"\x1bOP", b"\x1b[99;99u"):
            steer._buffer.clear()
            for byte in keys:
                steer._handle_byte(byte)
        self.assertEqual(interrupts, [])
        self.assertEqual(steer.line(), "")

    def test_ctrl_c_does_not_disturb_the_queue(self):
        interrupts: list[str] = []
        steer = self._steer(interrupts)
        steer._handle_byte(ord("q"))
        for byte in self.CTRL_C:
            steer._handle_byte(byte)
        self.assertEqual(steer.line(), "q")
        self.assertIsNone(steer.pending())

    def test_it_matches_the_sequence_the_keys_module_declares(self):
        """Keep steer.py and agent/keys.py from drifting apart."""
        from agent.keys import is_ctrl_c_sequence

        self.assertTrue(is_ctrl_c_sequence(self.CTRL_C.decode()))


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
        # Start from a known cooked state, whatever the test runner is in.
        # EIO here means this sandbox has a /dev/tty that can be opened and
        # queried but not reconfigured. The cbreak transition under test is a
        # tcsetattr, so there is nothing observable here and the test would
        # fail for an environment reason rather than a code one. Checked
        # *before* the try/finally, so a skip is not masked by the restore.
        try:
            pristine = termios.tcgetattr(fd)
            termios.tcsetattr(fd, termios.TCSANOW, _cooked_attrs(pristine))
        except termios.error as exc:
            os.close(fd)
            self.skipTest(f"/dev/tty cannot be reconfigured: {exc}")
        try:
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
        """End-to-end: Ctrl+C reaches the main thread as KeyboardInterrupt.

        Driven entirely over the pty, with a handshake: the child announces
        that it is armed, and only then does the parent send Ctrl+C. Two
        things depend on that ordering.

        A fixed sleep is a race. If Ctrl+C arrives before the child has
        installed a handler, the kernel's *default* disposition applies and it
        terminates the child outright -- the test then reports a missing file
        rather than a missing SIGINT. Under load (several suites at once) the
        imports alone can outlast the sleep, which is exactly how this test
        used to fail intermittently.

        The result also goes over the pty rather than a temp file: a file has
        to be created, written, closed and unlinked across a process boundary,
        so parallel runs can race each other out of existence.
        """
        import pty
        import select
        import struct
        import fcntl
        import termios
        import time

        def body() -> None:  # pragma: no cover - child process
            import signal
            import termios as T

            # unittest installs a SIGINT handler in the *parent* to report
            # Ctrl+C as a test result, and fork() inherits it. With that
            # handler in place the kernel never raises KeyboardInterrupt, so
            # the child could not observe SIGINT no matter what SteerInput did
            # with ISIG -- it would fail for a reason unrelated to the code
            # under test. (SIG_DFL would kill the child outright; the default
            # *int handler* is what turns SIGINT into KeyboardInterrupt.)
            signal.signal(signal.SIGINT, signal.default_int_handler)

            steer = SteerInput(on_submit=lambda _t: None)
            armed = steer.start()
            lflag = T.tcgetattr(0)[3]
            sys.stdout.write("READY armed=%s isig=%s icanon=%s\n" % (
                armed, bool(lflag & T.ISIG), bool(lflag & T.ICANON),
            ))
            sys.stdout.flush()
            try:
                time.sleep(3.0)
                sys.stdout.write("none\n")
            except KeyboardInterrupt:
                sys.stdout.write("sigint\n")
            sys.stdout.flush()
            steer.stop()
            os._exit(0)

        pid, mfd = pty.fork()
        if pid == 0:  # pragma: no cover - child process
            try:
                body()
            finally:
                os._exit(0)
        fcntl.ioctl(mfd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))

        out = b""
        sent = False
        deadline = time.time() + 15
        while time.time() < deadline:
            r, _, _ = select.select([mfd], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(mfd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
            if not sent and b"READY" in out:
                os.write(mfd, b"\x03")  # Ctrl+C, in through the master
                sent = True
            if sent and b"sigint" in out:
                break
        try:
            os.close(mfd)
        except OSError:
            pass
        os.waitpid(pid, 0)

        text = out.decode(errors="replace")
        self.assertIn("READY", text, f"child never armed: {text!r}")
        self.assertIn("armed=True", text)
        self.assertIn("isig=True", text)
        self.assertIn("icanon=False", text)
        self.assertIn("sigint", text, f"Ctrl+C did not reach the main thread: {text!r}")


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


class _FakeSteer:
    """A SteerInput stand-in that records lifecycle calls."""

    def __init__(self, armed: bool = False) -> None:
        self._armed = armed
        self.starts = 0
        self.stops = 0

    @property
    def armed(self) -> bool:
        return self._armed

    def start(self) -> bool:
        self.starts += 1
        self._armed = True
        return True

    def stop(self) -> None:
        self.stops += 1
        self._armed = False


class TestSteerPausedArmsOnlyIfAlreadyArmed(unittest.TestCase):
    """A subprocess must not leave a second reader on the tty.

    Between turns the agent calls ``output.steer_stop()``, so when the user
    runs ``/config edit`` or ``/edit`` the reader is already disarmed and
    prompt_toolkit owns the tty. ``steer_paused`` used to re-arm it
    unconditionally, leaving two threads reading the same fd: the terminal
    stayed in cbreak mode and every keystroke became a race between them, so
    most keys vanished into the discarded steering buffer.
    """

    def _adapter(self, armed: bool) -> Any:
        from agent.output import RichOutputAdapter

        out = RichOutputAdapter.__new__(RichOutputAdapter)
        # A stand-in is what we are testing the *policy* against; the real
        # reader needs a tty, which this test deliberately does not assume.
        out._steer = cast(Any, _FakeSteer(armed))
        return out

    @staticmethod
    def _fake(out: Any) -> _FakeSteer:
        return cast(_FakeSteer, out._steer)

    def test_disarmed_reader_stays_disarmed(self):
        """The real case: /config edit and /edit run between turns."""
        out = self._adapter(armed=False)
        with out.steer_paused():
            self.assertEqual(self._fake(out).stops, 1)
        self.assertEqual(self._fake(out).starts, 0)
        self.assertFalse(self._fake(out).armed)

    def test_armed_reader_is_rearmed(self):
        """Mid-turn, the reader was armed and must survive the subprocess."""
        out = self._adapter(armed=True)
        with out.steer_paused():
            self.assertFalse(self._fake(out).armed)
        self.assertEqual(self._fake(out).stops, 1)
        self.assertEqual(self._fake(out).starts, 1)
        self.assertTrue(self._fake(out).armed)

    def test_reader_is_stopped_even_if_the_body_raises(self):
        out = self._adapter(armed=True)
        with self.assertRaises(RuntimeError):
            with out.steer_paused():
                raise RuntimeError("editor exploded")
        self.assertEqual(self._fake(out).starts, 1)

    def test_no_reader_is_a_no_op(self):
        out = self._adapter(armed=False)
        out._steer = None
        with out.steer_paused():
            pass


def _pty_state_after_run_subprocess() -> str:
    """Child: turn end -> run_subprocess -> report tty state on stdout.

    Run under a real pty, because the bug is only visible when there is a
    terminal: the reader switches the tty to cbreak, and two readers on one
    fd is invisible in a pipe.
    """
    import pty
    import select
    import subprocess
    import termios
    import time

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child = (
        "import os, sys, termios\n"
        f"sys.path.insert(0, {repo!r})\n"
        "from agent.output import RichOutputAdapter\n"
        "from agent.steer import SteerInput\n"
        "a = termios.tcgetattr(0)\n"
        "out = RichOutputAdapter()\n"
        "out._steer = SteerInput()\n"
        "out._steer.start()\n"
        "out._steer.stop()\n"
        "out.run_subprocess(['true'])\n"
        "b = termios.tcgetattr(0)\n"
        "print('ARMED', out._steer.armed,\n"
        "      'ICANON', bool(b[3] & termios.ICANON),\n"
        "      'ECHO', bool(b[3] & termios.ECHO))\n"
        "sys.stdout.flush()\n"
    )

    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-u", "-c", child])

    out = b""
    deadline = time.time() + 20
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.2)
        if not r:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
        # Wait for the *last* field, not the first: the child prints one
        # line in several write() calls, so b"ARMED" can arrive while the
        # rest of the line is still in flight. Breaking there truncated the
        # report and made the assertion below fail intermittently.
        if b"ECHO" in out and out.rstrip().endswith(b"True"):
            break
    os.close(fd)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return out.decode(errors="replace")


class TestEditLeavesTheTtyUsable(unittest.TestCase):
    """End-to-end: the prompt still owns the tty after an editor runs."""

    def test_tty_is_cooked_and_reader_disarmed(self):
        text = _pty_state_after_run_subprocess()
        self.assertIn("ARMED False", text, text)
        self.assertIn("ICANON True", text, text)
        self.assertIn("ECHO True", text, text)


if __name__ == "__main__":
    unittest.main()
