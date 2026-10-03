"""Mid-turn steering input for the classic REPL (the "cheap" version).

While an assistant turn is running, the prompt_toolkit prompt is off-screen
and the main thread is blocked inside ``Core.run_turn()``. The terminal
itself is idle, though, and the sticky footer already reserves the bottom
rows, so a queued follow-up can be collected with very little machinery:

* :meth:`SteerInput.start` puts the controlling tty into **cbreak** mode
  (``ICANON | ECHO`` cleared, ``ISIG`` left on so Ctrl+C keeps working) and
  starts a daemon reader thread.
  Nothing is echoed to the terminal \u2014 the partial line lives only in the
  buffer and is surfaced in the footer status line, so there is no second
  writer on stdout and no cursor management to get wrong.
* Printable bytes append to the line; ``Backspace``/``Ctrl+H`` delete,
  ``Ctrl+U`` clears, ``Enter`` submits. Ctrl+C is not a line-editing key:
  normally ``ISIG`` makes the kernel raise ``KeyboardInterrupt`` in the main
  thread, which is what the turn already knows how to unwind. When the kitty
  keyboard protocol is enabled (see :mod:`agent.keys`) the terminal reports
  Ctrl+C as an escape sequence instead of ``0x03``, so the kernel is bypassed
  entirely and the reader re-raises SIGINT itself.
* Submitted lines are queued. The REPL main loop drains them right after the
  turn ends, so they go through the exact same ``@``-expansion, command
  dispatch and turn path as a typed prompt.

This is deliberately *not* a line editor: no history, no completion, no
prompt_toolkit. It exists to steer a running turn ("actually, use pandas"),
and it degrades to a no-op whenever the terminal is not interactive or the
tty cannot be put into cbreak mode.

The reader thread is also the reason :class:`agent.footer.Footer` serializes
its writes with a lock: status updates now come from two threads.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import termios
import threading
from collections.abc import Callable

from agent.keys import is_ctrl_c_sequence

# Bytes that terminate a CSI/SS3 escape sequence (we swallow them whole so
# arrow keys and friends do not leak into the line buffer).
_CSI_FINAL = frozenset(
    b"@ABCDEFGHJKSTfmnsu\x7e"
)


class SteerInput:
    """Collect queued follow-up lines from the tty while a turn is running.

    Parameters
    ----------
    on_submit:
        Optional notification (from the reader thread) for each submitted
        line. The line itself is appended to an internal FIFO drained with
        :meth:`take`; the callback is used to refresh the footer.
    on_change:
        Called (from the reader thread) whenever the buffered line changes, so
        the caller can surface it (e.g. in the sticky footer status line).
    max_len:
        Hard cap on the buffered line, in characters.
    """

    def __init__(
        self,
        on_submit: Callable[[str], None] | None = None,
        on_change: Callable[[], None] | None = None,
        max_len: int = 1000,
    ) -> None:
        self._on_submit = on_submit
        self._on_change = on_change
        self._max_len = max_len

        self._lock = threading.Lock()
        self._buffer: list[str] = []
        self._queue: list[str] = []

        self._esc_mode = 0
        # Raw bytes of the escape sequence currently being consumed. Needed
        # because a kitty-protocol Ctrl+C arrives as an escape sequence
        # rather than as 0x03, and cannot be recognised one byte at a time.
        self._esc_buf: bytes = b""
        self._fd: int | None = None
        self._saved_attrs: list | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._armed = False

    # ── state ────────────────────────────────────────────────────────────

    @property
    def armed(self) -> bool:
        """True while the reader thread is capturing keystrokes."""
        return self._armed

    def take(self) -> str | None:
        """Pop the oldest queued line, or None if nothing is queued."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def ack(self, text: str) -> bool:
        """Remove the first queued line equal to *text*; report whether found.

        Used by the remote path: the client submits a line and sends it to the
        daemon, but keeps its own copy until the daemon confirms delivery with
        an ``injected`` event. Only then is the local copy dropped. If the turn
        ends first and the confirmation never arrives, the line is still in the
        queue and the main loop runs it as an ordinary next prompt.
        """
        with self._lock:
            try:
                self._queue.remove(text)
                return True
            except ValueError:
                return False

    def take_all(self) -> list[str]:
        """Pop and return every queued line, oldest first.

        Used by the turn's injection seam, which wants to hand all pending
        steering text to the model in one go rather than one line per
        tool-loop round.
        """
        with self._lock:
            lines, self._queue = self._queue, []
        return lines

    def pending(self) -> str | None:
        """Return the oldest queued line without consuming it (for display)."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def line(self) -> str:
        """Return the line currently being typed (for display)."""
        with self._lock:
            return "".join(self._buffer)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Arm the reader. Returns True if keystrokes are being captured."""
        if self._armed:
            return True
        fd = self._open_tty()
        if fd is None:
            return False
        if not self._enter_cbreak(fd):
            return False
        self._fd = fd
        self._esc_mode = 0
        self._stop = False
        self._armed = True
        self._thread = threading.Thread(
            target=self._read_loop, name="wisemonkey-steer", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Disarm the reader, restore the tty, and drop the partial line."""
        if not self._armed:
            return
        self._armed = False
        self._stop = True
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        self._thread = None
        self._restore_tty()
        self._fd = None
        with self._lock:
            self._buffer.clear()

    # Steered turns can be interrupted by the agent itself asking a question
    # (ask_confirm/ask_string) or by a subprocess taking over the terminal.
    # Those need the tty in its normal state, hence the aliases.
    suspend = stop

    def resume(self) -> bool:
        """Re-arm after a :meth:`suspend`, keeping the queued lines."""
        return self.start()

    # ── tty handling ──────────────────────────────────────────────────────

    @staticmethod
    def _open_tty() -> int | None:
        """Return a readable fd for the controlling terminal, or None."""
        try:
            if not sys.stdin.isatty():
                return None
            return sys.stdin.fileno()
        except (AttributeError, ValueError, OSError):
            pass
        # stdin may be redirected while stdout is still a terminal: fall back
        # to the controlling terminal itself.
        try:
            fd = os.open("/dev/tty", os.O_RDONLY)
        except OSError:
            return None
        return fd

    def _enter_cbreak(self, fd: int) -> bool:
        """Put the tty in cbreak mode.

        Clears ``ICANON`` (so keys arrive one byte at a time) and ``ECHO``
        (the tty must not echo behind the footer's back), but deliberately
        **keeps** ``ISIG`` on.

        With ``ISIG`` cleared the kernel would not turn Ctrl+C into SIGINT:
        the reader thread would see a literal ``0x03`` byte and could only set
        a flag, which the turn never checks while it is blocked in
        ``router.chat()`` or inside a tool — so Ctrl+C would appear to do
        nothing exactly during the phases the user presses it for (waiting
        for the prompt, a tool running). Keeping ``ISIG`` on preserves the
        original behaviour, where Ctrl+C raises ``KeyboardInterrupt`` in the
        main thread. Ctrl+U (or Backspace) clears the steering line instead.

        ``ISIG`` on its own is not enough once the kitty keyboard protocol is
        enabled: the terminal then reports Ctrl+C as ``CSI 99 ; mods u``
        instead of ``0x03``, so the kernel never sees it at all.
        :meth:`_interrupt` covers that case.
        """
        if not hasattr(termios, "TCSANOW"):  # not a POSIX tty
            return False
        try:
            attrs = termios.tcgetattr(fd)
        except (termios.error, OSError, ValueError):
            return False
        # tcgetattr returns [iflag, oflag, cflag, lflag, ispeed, ospeed, cc],
        # where only the control-character list is indexable.
        iflag, oflag, cflag, lflag, ispeed, ospeed, cc = (
            attrs[0], attrs[1], attrs[2], attrs[3], attrs[4], attrs[5], list(attrs[6])
        )
        self._saved_attrs = [iflag, oflag, cflag, lflag, ispeed, ospeed, list(cc)]
        cc[termios.VMIN] = 1
        cc[termios.VTIME] = 0
        lflag &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(
            fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
        )
        return True

    def _restore_tty(self) -> None:
        fd, attrs = self._fd, self._saved_attrs
        self._saved_attrs = None
        if fd is None or attrs is None or not hasattr(termios, "TCSANOW"):
            return
        try:
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except (termios.error, OSError, ValueError):
            pass

    # ── reader thread ─────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Read raw bytes until stopped; feed them to the line buffer."""
        fd = self._fd
        if fd is None:
            return
        while not self._stop:
            try:
                ready, _, _ = select.select([fd], [], [], 0.2)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                data = os.read(fd, 64)
            except (OSError, ValueError):
                break
            if not data:
                continue
            for byte in data:
                if self._stop:
                    break
                self._handle_byte(byte)

    def _handle_byte(self, byte: int) -> None:
        """Apply a single input byte to the line buffer.

        Escape sequences (arrow keys, function keys) are swallowed whole so
        they never leak into the line as stray characters.
        """
        if self._esc_mode:
            # Swallow the remainder of an escape sequence.
            if self._esc_mode == 1:
                if byte not in (ord("["), ord("O")):
                    self._esc_mode = 0
                    self._esc_buf = b""
                else:
                    self._esc_mode = 2
                    self._esc_buf = bytes([byte])
            else:
                self._esc_buf += bytes([byte])
                if byte not in _CSI_FINAL:
                    self._esc_mode = 2
                    return
                self._esc_mode = 0
                sequence = b"\x1b" + self._esc_buf
                self._esc_buf = b""
                if is_ctrl_c_sequence(sequence.decode("ascii", "replace")):
                    self._interrupt()
            return
        if byte == 0x1B:
            self._esc_mode = 1
            return
        if byte in (0x0D, 0x0A):  # Enter
            text = self.line().strip()
            with self._lock:
                self._buffer.clear()
                if text:
                    self._queue.append(text)
            self._changed()
            if text:
                self._safe(self._on_submit, text)
            return
        if byte in (0x03, 0x04, 0x1A, 0x1C):
            # Ctrl+C / Ctrl+D / Ctrl+\ / Ctrl+Z. ISIG is left enabled, so the
            # kernel turns Ctrl+C into SIGINT (and Ctrl+\ into SIGQUIT)
            # before these bytes ever reach us; the rest of the control
            # characters are ignored. Never buffer them.
            return
        if byte in (0x7F, 0x08):  # Backspace
            with self._lock:
                if self._buffer:
                    self._buffer.pop()
                    changed = True
                else:
                    changed = False
            if changed:
                self._changed()
            return
        if byte == 0x15:  # Ctrl+U: clear the line
            with self._lock:
                had = bool(self._buffer)
                self._buffer.clear()
            if had:
                self._changed()
            return
        if byte < 0x20 or byte == 0x7F:
            return  # other control characters: ignore
        try:
            char = bytes([byte]).decode("utf-8")
        except UnicodeDecodeError:
            return
        if not char.isprintable():
            return
        with self._lock:
            if sum(len(c) for c in self._buffer) < self._max_len:
                self._buffer.append(char)
                changed = True
            else:
                changed = False
        if changed:
            self._changed()

    def _interrupt(self) -> None:
        """Raise SIGINT in the main thread, on behalf of Ctrl+C.

        With the kitty keyboard protocol enabled the terminal reports Ctrl+C
        as ``CSI 99 ; mods u`` rather than ``0x03``, so the kernel never
        generates the signal and ``ISIG`` has nothing to act on. Nothing is
        listening on this thread for it either, so we re-raise it ourselves:
        the turn already unwinds on ``KeyboardInterrupt`` in the main thread,
        which is exactly the behaviour Ctrl+C had before the protocol was
        enabled.

        Called from the reader thread; :func:`signal.raise_signal` is safe
        from any thread because Python defers signal handling to the main
        thread.
        """
        try:
            signal.raise_signal(signal.SIGINT)
        except (OSError, ValueError):
            pass

    def _changed(self) -> None:
        """Notify the caller that the buffered line changed."""
        if self._on_change is not None:
            self._safe(self._on_change)

    @staticmethod
    def _safe(func, *args) -> None:
        """Call a callback, swallowing any error from the reader thread."""
        try:
            func(*args)
        except Exception:
            pass