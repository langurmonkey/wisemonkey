"""Tests for agent/steer.py: mid-turn steering input."""

import os
import sys
import threading
import time
import unittest
from typing import TYPE_CHECKING, Any, cast
from unittest import mock

from agent.steer import SteerInput

if TYPE_CHECKING:
    from agent.output import RichOutputAdapter


class _Typing(unittest.TestCase):
    """A `SteerInput` driven a byte at a time, without a tty."""

    def setUp(self) -> None:
        self.submitted: list[str] = []
        self.changes = 0
        self.steer = SteerInput(
            on_submit=self.submitted.append, on_change=self._change
        )

    def _change(self) -> None:
        self.changes += 1

    def _type(self, text: str) -> None:
        for char in text:
            self.steer._handle_byte(ord(char))

    def _send(self, data: bytes) -> None:
        for byte in data:
            self.steer._handle_byte(byte)

    def _enter(self) -> None:
        self.steer._handle_byte(0x0D)

    def _steer(self, text: str) -> None:
        self._type(text)
        self._enter()


class TestSteerBuffer(_Typing):
    def test_typing_builds_the_line_and_enter_submits_it_stripped(self):
        self._type("  hello  ")
        self.assertEqual(self.steer.line(), "  hello  ")
        self.assertEqual(self.changes, len("  hello  "))
        self._enter()
        self.assertEqual(self.submitted, ["hello"])
        self.assertEqual(self.steer.line(), "")
        # Nothing typed: nothing submitted.
        self._enter()
        self.assertEqual(self.submitted, ["hello"])

    def test_backspace_and_control_bytes(self):
        self._type("ab")
        for key in (0x7F, 0x08):
            self.steer._handle_byte(key)
        self.assertEqual(self.steer.line(), "")
        before = self.changes
        self.steer._handle_byte(0x7F)  # empty: no text, no redraw
        self.assertEqual(self.changes, before)

        steer = SteerInput(on_submit=lambda _t: None, max_len=4)
        for char in "abcdefgh":
            steer._handle_byte(ord(char))
        self.assertEqual(steer.line(), "abcd")
        # Ctrl+C arrives as SIGINT (ISIG stays on), so the reader never sees
        # 0x03; if one arrives anyway it must not reach the line.
        for byte in (0x01, 0x07, 0x0C, 0x1A, 0x03):
            steer._handle_byte(byte)
        self.assertEqual(steer.line(), "abcd")

    def test_escape_sequences_do_not_leak_and_a_failing_submit_survives(self):
        for sequence in (b"\x1b[A", b"\x1b[1;5C", b"\x1b[13;130u"):
            self._send(sequence)
            self.assertEqual(self.steer.line(), "", sequence)
            self.assertEqual(self.steer._esc_mode, 0, sequence)

        def boom(_line: str) -> None:
            raise RuntimeError("nope")

        steer = SteerInput(on_submit=boom)
        steer._handle_byte(ord("x"))
        steer._handle_byte(0x0D)  # must not raise

    def test_stop_clears_the_partial_line_but_suspend_keeps_it(self):
        """An interruption is not the end of a turn."""
        self._steer("queued line")
        self._type("half typed")
        self._send(b"\x01")  # Ctrl+A: caret to the start
        self.steer._armed = True
        self.steer.suspend()
        self.assertFalse(self.steer.armed)
        self.assertEqual((self.steer.line(), self.steer.caret()), ("half typed", 0))
        self.assertEqual(self.steer.pending(), "queued line")
        # No tty here, so resume() cannot arm the reader -- but it must not
        # touch the buffer, and stop() is still the end of it.
        self.steer.resume()
        self.assertEqual(self.steer.line(), "half typed")
        self.steer._armed = True
        self.steer.stop()
        self.assertEqual(self.steer.line(), "")
        self.assertEqual(self.steer.pending(), "queued line")


class TestSteerQueue(unittest.TestCase):
    """The FIFO backing mid-turn injection."""

    def setUp(self) -> None:
        self.steer = SteerInput(on_submit=lambda _t: None)
        self.steer._queue.extend(["first", "second"])

    def test_take_is_fifo_and_take_all_drains_in_order(self):
        self.assertEqual(self.steer.pending(), "first")  # peeking does not consume
        self.assertEqual(self.steer.pending(), "first")
        self.assertEqual(self.steer.take(), "first")
        self.assertEqual(self.steer.take(), "second")
        self.assertIsNone(self.steer.take())

        self.assertEqual(self.steer.take_all(), [])
        self.steer._queue.extend(["a", "b"])
        self.assertEqual(self.steer.take_all(), ["a", "b"])
        self.assertEqual(self.steer.take_all(), [])
        self.assertIsNone(self.steer.pending())

    def test_ack_removes_one_named_occurrence(self):
        """Delivery is confirmed out of order relative to typing order."""
        self.steer._queue.append("first")
        self.assertTrue(self.steer.ack("second"))
        self.assertEqual(self.steer.take_all(), ["first", "first"])
        self.assertFalse(self.steer.ack("never sent"))
        self.assertFalse(SteerInput().ack("anything"))


class TestSteerLineEditing(_Typing):
    """The caret moves, and the text is edited around it."""

    # Legacy, SS3, the protocol's own (129 is num_lock), and the tilde form.
    LEFT, RIGHT = b"\x1b[D", b"\x1b[C"
    HOME = (b"\x1b[H", b"\x1bOH", b"\x1b[1;129H", b"\x1b[1~")
    END = (b"\x1b[F", b"\x1bOF", b"\x1b[1;129F", b"\x1b[4~")

    def test_every_encoding_of_left_right_home_and_end(self):
        for left, right in (
            (self.LEFT, self.RIGHT),
            (b"\x1b[1;129D", b"\x1b[1;129C"),
            (b"\x1bOD", b"\x1bOC"),
        ):
            with self.subTest(left=left):
                steer = SteerInput()
                for char in "ab":
                    steer._handle_byte(ord(char))
                for byte in left:
                    steer._handle_byte(byte)
                assert steer.caret() == 1, (left, steer.caret())
                assert steer.line() == "ab", left  # moving changes no text
                for byte in right:
                    steer._handle_byte(byte)
                assert steer.caret() == 2, (right, steer.caret())
        for home in self.HOME:
            for end in self.END:
                with self.subTest(home=home, end=end):
                    steer = SteerInput()
                    for char in "abc":
                        steer._handle_byte(ord(char))
                    for byte in home:
                        steer._handle_byte(byte)
                    assert steer.caret() == 0, home
                    for byte in end:
                        steer._handle_byte(byte)
                    assert steer.caret() == 3, end

    def test_ctrl_arrows_move_by_word_and_ctrl_letters_by_one(self):
        self._type("use pandas instead")
        self._send(b"\x1b[1;5D")
        self.assertEqual(self.steer.caret(), 11)
        self._send(b"\x1b[1;5D")
        self.assertEqual(self.steer.caret(), 4)
        self._send(b"\x1b[1;5C")
        self.assertEqual(self.steer.caret(), 11)
        # Ctrl+A / Ctrl+F / Ctrl+E / Ctrl+B.
        steer = SteerInput()
        for char in "abc":
            steer._handle_byte(ord(char))
        for key, caret in ((0x01, 0), (0x06, 1), (0x05, 3), (0x02, 2)):
            steer._handle_byte(key)
            self.assertEqual(steer.caret(), caret, hex(key))

    def test_typing_inserts_at_the_caret_and_deletion_works_on_both_sides(self):
        self._type("ac")
        self._send(self.LEFT)
        self._type("b")
        self.assertEqual((self.steer.line(), self.steer.caret()), ("abc", 2))
        self._send(b"\x7f")
        self.assertEqual((self.steer.line(), self.steer.caret()), ("ac", 1))
        # At the start of the line there is nothing to delete: no text, no redraw.
        self._send(self.LEFT)
        before = self.changes
        self._send(b"\x7f")
        self.assertEqual((self.steer.line(), self.changes), ("ac", before))

        for keys in (b"\x1b[3~", b"\x1b[3;129u", b"\x04"):
            with self.subTest(keys=keys):
                steer = SteerInput()
                for char in "abc":
                    steer._handle_byte(ord(char))
                for byte in self.LEFT:
                    steer._handle_byte(byte)
                for byte in keys:
                    steer._handle_byte(byte)
                assert steer.line() == "ab", (keys, steer.line())
                assert steer.caret() == 2, keys

        # Enter submits and resets the caret.
        steer = SteerInput(on_submit=self.submitted.append)
        for char in "abc":
            steer._handle_byte(ord(char))
        for byte in self.LEFT + self.LEFT:
            steer._handle_byte(byte)
        steer._handle_byte(0x0D)
        assert (steer.line(), steer.caret()) == ("", 0)
        assert self.submitted == ["abc"]

    def test_the_kill_keys(self):
        self._type("hello world")
        for _ in range(6):
            self._send(self.LEFT)
        self._send(b"\x0b")  # Ctrl+K: kill to the end
        self.assertEqual((self.steer.line(), self.steer.caret()), ("hello", 5))

        steer = SteerInput()
        for char in "use pandas":
            steer._handle_byte(ord(char))
        steer._handle_byte(0x17)  # Ctrl+W: kill the word before the caret
        self.assertEqual(steer.line(), "use ")
        steer._handle_byte(0x17)
        self.assertEqual((steer.line(), steer.caret()), ("", 0))

        # Ctrl+U clears the whole line, caret included, wherever it sits.
        for char in "discard me":
            steer._handle_byte(ord(char))
        for byte in self.LEFT:
            steer._handle_byte(byte)
        steer._handle_byte(0x15)
        self.assertEqual((steer.line(), steer.caret()), ("", 0))

    def test_swallowed_sequences_and_moves_that_change_nothing(self):
        """Decoded sequences move the caret; unknown ones are swallowed."""
        self._type("ab")
        before = self.changes
        self._send(self.LEFT)
        self.assertEqual(self.changes, before + 1)
        self._send(self.RIGHT)
        self._send(self.RIGHT)  # already at the end
        self.assertEqual((self.steer.caret(), self.changes), (2, before + 2))

        # PageUp, F-keys and the keypad move nothing and print nothing.
        for keys in (b"\x1b[5~", b"\x1b[15~", b"\x1b[57414;129u"):
            self._send(keys)
            self.assertEqual(self.steer.line(), "ab", keys)
        self._type("c")
        self.assertEqual(self.steer.line(), "abc")

        # An Escape that starts nothing must not eat the character after it.
        steer = SteerInput()
        steer._handle_byte(0x1B)
        steer._handle_byte(ord("x"))
        self.assertEqual(steer.line(), "x")


class TestSteerCtrlCFromKittyProtocol(unittest.TestCase):
    """Ctrl+C as the kitty keyboard protocol reports it."""

    CTRL_C = b"\x1b[99;133u"  # key 'c', ctrl + num_lock

    def setUp(self) -> None:
        self.interrupts: list[str] = []
        # _interrupt raises SIGINT for real; here we only observe it was reached,
        # since there is no tty and no turn to interrupt.
        patcher = mock.patch.object(
            SteerInput, "_interrupt",
            lambda _self: self.interrupts.append("sigint"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.steer = SteerInput()

    def _feed(self, data: bytes) -> None:
        for byte in data:
            self.steer._handle_byte(byte)

    def test_it_raises_an_interrupt_without_leaking_into_the_line(self):
        self._feed(b"ab" + self.CTRL_C + b"cd")
        self.assertEqual(self.interrupts, ["sigint"])
        self.assertEqual(self.steer.line(), "abcd")
        # The escape state is left clean, so typing resumes normally.
        self.assertEqual(self.steer._esc_mode, 0)
        # Hyper is a real modifier too: Ctrl+Hyper+C is still a cancel.
        self._feed(b"\x1b[99;99u")
        self.assertEqual(self.interrupts, ["sigint", "sigint"])
        # The sequence is the one agent/keys.py declares: no drift between them.
        from agent.keys import is_ctrl_c_sequence

        self.assertTrue(is_ctrl_c_sequence(self.CTRL_C.decode()))

    def test_ordinary_control_c_and_every_other_sequence_are_left_alone(self):
        """The bare 0x03 arrives as SIGINT (ISIG); the reader must ignore it."""
        self._feed(b"ab\x03cd")
        self.assertEqual(self.interrupts, [])
        self.assertEqual(self.steer.line(), "abcd")

        self.steer = SteerInput()  # a fresh line for the swallowing table
        for keys in (
            b"\x1b[D", b"\x1b[A", b"\x1b[13;130u", b"\x1bOP",
            # The whole re-encoded set: lock and shifted variants of every
            # functional key.
            b"\x1b[1;129A", b"\x1b[1;129B", b"\x1b[1;129C", b"\x1b[1;129D",
            b"\x1b[1;129H", b"\x1b[1;129F", b"\x1b[3;129~", b"\x1b[127;130u",
            b"\x1b[9;130u", b"\x1b[57419;129u", b"\x1b[57414;129u",
            b"\x1b[57376;129u", b"\x1b[27;130;13~",
            # SS3 function keys, whose single final byte is not a CSI one.
            b"\x1bOQ", b"\x1bOR", b"\x1bOS",
        ):
            with self.subTest(keys=keys):
                self._feed(keys)
                assert self.steer._esc_mode == 0, f"{keys!r} left the reader mid-escape"
                assert self.steer._esc_buf == b"", keys
                assert self.steer.line() == "", keys
        self.assertEqual(self.interrupts, [])
        # Typing resumes after a swallowed sequence: SS3's single final byte is
        # the trap that used to swallow the next keystroke too.
        self._feed(b"hello")
        self.assertEqual(self.steer.line(), "hello")


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

    def test_cbreak_clears_icanon_and_echo_but_keeps_isig(self):
        import termios

        # `os.path.exists` is not enough: opening /dev/tty raises ENXIO without a
        # controlling terminal, which is exactly a `spawn` job's state.
        try:
            fd = os.open("/dev/tty", os.O_RDWR)
        except OSError:
            self.skipTest("no controlling tty")
        if not os.isatty(fd):
            os.close(fd)
            self.skipTest("no controlling tty")
        # EIO means this sandbox's /dev/tty can be opened but not reconfigured:
        # skip, checked before the try/finally.
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
        """End-to-end: Ctrl+C reaches the main thread as KeyboardInterrupt."""
        import fcntl
        import pty
        import select
        import struct
        import termios

        def body() -> None:  # pragma: no cover - child process
            import signal
            import termios as T

            # unittest installs a SIGINT handler in the parent and fork() inherits
            # it, so the child could never see KeyboardInterrupt.
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
        self.assertIn("sigint", text, f"Ctrl+C did not reach the thread: {text!r}")


class TestSteerInputNoTty(unittest.TestCase):
    """Without a terminal, steering must be a silent no-op."""

    def test_start_returns_false_and_stop_is_safe(self):
        if os.isatty(0):
            self.skipTest("running with a real tty")
        steer = SteerInput(on_submit=lambda _t: None)
        self.assertFalse(steer.start())
        self.assertFalse(steer.armed)
        # Never started: stopping must not raise.
        SteerInput(on_submit=lambda _t: None).stop()


class _FakeSteer:
    """A SteerInput stand-in that records lifecycle calls."""

    def __init__(self, armed: bool = False, line: str = "") -> None:
        self._armed = armed
        self.starts = self.stops = self.suspends = self.resumes = 0
        # The partial line, so a policy test can see whether it survived.
        self.typed = line

    @property
    def armed(self) -> bool:
        return self._armed

    def start(self) -> bool:
        self.starts += 1
        self._armed = True
        return True

    def stop(self) -> None:
        """The end of a turn: the partial line has nowhere left to go."""
        self.stops += 1
        self._armed = False
        self.typed = ""

    def suspend(self) -> None:
        """A pause: the line is mid-sentence and must survive."""
        self.suspends += 1
        self._armed = False

    def resume(self) -> bool:
        self.resumes += 1
        self._armed = True
        return True


class TestSteerPaused(unittest.TestCase):
    """A subprocess must not leave a second reader on the tty, nor eat the line."""

    def _adapter(self, armed: bool, line: str = "") -> Any:
        from agent.output import RichOutputAdapter

        out = RichOutputAdapter.__new__(RichOutputAdapter)
        # A stand-in is what we are testing the *policy* against; the real
        # reader needs a tty, which this test deliberately does not assume.
        out._steer = cast(Any, _FakeSteer(armed, line))
        return out

    @staticmethod
    def _fake(out: Any) -> _FakeSteer:
        return cast(_FakeSteer, out._steer)

    def test_an_armed_reader_is_suspended_and_rearmed_even_on_an_exception(self):
        """Mid-turn, the reader was armed and must survive the subprocess."""
        out = self._adapter(armed=True)
        with self.assertRaises(RuntimeError):
            with out.steer_paused():
                self.assertFalse(self._fake(out).armed)
                raise RuntimeError("editor exploded")
        self.assertEqual(self._fake(out).suspends, 1)
        self.assertEqual(self._fake(out).resumes, 1)
        self.assertTrue(self._fake(out).armed)

        # The real case: /config edit and /edit run between turns, disarmed.
        out = self._adapter(armed=False)
        with out.steer_paused():
            self.assertEqual(self._fake(out).suspends, 1)
        self.assertEqual((self._fake(out).starts, self._fake(out).resumes), (0, 0))
        self.assertFalse(self._fake(out).armed)
        # No reader at all is a no-op.
        out._steer = None
        with out.steer_paused():
            pass

    def test_the_typed_line_survives_a_pause(self):
        """A pause is not a submit."""
        for raise_in_body in (False, True):
            out = self._adapter(armed=True, line="use pandas")
            if raise_in_body:
                with self.assertRaises(RuntimeError):
                    with out.steer_paused():
                        raise RuntimeError("the tool blew up")
            else:
                with out.steer_paused():
                    pass
            self.assertEqual(self._fake(out).typed, "use pandas")
        # The same invariant against the real reader.
        steer = SteerInput()
        for char in "use pandas":
            steer._handle_byte(ord(char))
        steer._armed = True  # pretend a tty was there
        steer.suspend()
        self.assertEqual((steer.line(), steer.caret()), ("use pandas", len("use pandas")))

    def test_the_row_is_redrawn_when_the_reader_comes_back(self):
        """The caret is dropped while paused, so re-arming must redraw it."""
        out = self._adapter(armed=True, line="half typed")
        redraws: list[int] = []
        out._steer_on_change = lambda: redraws.append(1)
        with out.steer_paused():
            self.assertEqual(redraws, [])  # nothing to redraw on the way in
        self.assertEqual(len(redraws), 1)
        # Never armed: nothing to redraw, either.
        self._fake(out)._armed = False
        redraws.clear()
        with out.steer_paused():
            pass
        self.assertEqual(redraws, [])


def _pty_state_after_run_subprocess() -> str:
    """Child: turn end -> run_subprocess -> report tty state on stdout."""
    import pty
    import select
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
        # Wait for the *last* field: the child prints the line in several write()
        # calls, so breaking on the first truncates the report.
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


class TestQuestionFocus(unittest.TestCase):
    """Where the keyboard starts when a tool asks something mid-turn."""

    def setUp(self) -> None:
        self.steer = SteerInput()
        # There is no tty in a test; the reader only checks this flag before
        # accepting a question.
        self.steer._armed = True
        self.addCleanup(self.steer.stop)

    def _ask(self, kind: str = "confirm", default: Any = False) -> tuple[threading.Thread, dict]:
        """Start a question in a thread; return it with a result holder."""

        def run() -> None:
            result["answer"] = self.steer.ask_question(kind, "Run it?", default)

        result: dict = {}
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for _ in range(400):
            if self.steer._question is not None:
                break
            time.sleep(0.005)
        self.assertIsNotNone(self.steer._question, "the question never started")
        return thread, result

    def _type(self, text: str) -> None:
        for char in text:
            self.steer._handle_byte(ord(char))

    def test_an_empty_line_gives_the_question_the_keyboard(self):
        thread, result = self._ask()
        self.assertEqual(self.steer.focus(), "ask")
        self.steer._handle_byte(ord("y"))
        thread.join(2)
        self.assertEqual(result["answer"], "yes")
        self.assertEqual((self.steer.focus(), self.steer.line()), ("line", ""))
        # A string question uses the same line editor, and its buffer is not
        # the steering line's.
        thread, result = self._ask("string", "dflt")
        self._type("hello")
        self.steer._handle_byte(0x0D)
        thread.join(2)
        self.assertEqual(result["answer"], "hello")
        self.assertEqual(self.steer.line(), "")

    def test_enter_takes_the_default(self):
        for kind, default, expected in (("string", "dflt", "dflt"),
                                        ("confirm", True, "yes")):
            thread, result = self._ask(kind, default)
            self.steer._handle_byte(0x0D)
            thread.join(2)
            self.assertEqual(result["answer"], expected, kind)

    def test_a_half_written_line_or_a_queue_keeps_the_keyboard_until_tab(self):
        self._type("note")
        thread, result = self._ask()
        self.assertEqual(self.steer.focus(), "line")
        self._type("y")  # answer-shaped, but the line has the keyboard
        self.assertEqual(self.steer.line(), "notey")
        self.assertNotIn("answer", result)
        # Tab is the only way onto the question, and the line survives it.
        self.steer._handle_byte(0x09)
        self.assertEqual(self.steer.focus(), "ask")
        self.steer._handle_byte(0x09)  # Tab goes back, without answering
        self.assertEqual(self.steer.focus(), "line")
        self.steer._handle_byte(0x09)
        self.steer._handle_byte(ord("n"))
        thread.join(2)
        self.assertEqual(result["answer"], "no")
        self.assertEqual(self.steer.line(), "notey")

        # Nothing typed but a line in flight is still steering.
        self.steer._handle_byte(0x15)  # Ctrl+U clears the line
        self._type("queued")
        self.steer._handle_byte(0x0D)
        thread, result = self._ask()
        self.assertEqual(self.steer.focus(), "line")
        self.steer._handle_byte(0x09)
        self._type("y")
        thread.join(2)
        self.assertEqual(result["answer"], "yes")

    def test_stop_cancels_a_pending_question(self):
        thread, result = self._ask()
        self.steer.stop()
        thread.join(2)
        self.assertIsNone(result["answer"])
