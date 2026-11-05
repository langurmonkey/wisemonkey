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
* Printable bytes insert at the caret; the line editor is small but real:
  arrow keys and Home/End move it, ``Backspace``/``Ctrl+H`` and ``Delete``/
  ``Ctrl+D`` delete around it, ``Ctrl+A``/``Ctrl+E``/``Ctrl+B``/``Ctrl+F`` are
  their readline aliases, ``Ctrl+W`` and ``Ctrl+Left``/``Ctrl+Right`` move by
  word, ``Ctrl+K`` kills to the end, ``Ctrl+U`` clears everything, and
  ``Enter`` submits. The caret position is published with the line
  (:meth:`caret`) so the frontend can draw it where it actually is.
  Ctrl+C is not a line-editing key:
  normally ``ISIG`` makes the kernel raise ``KeyboardInterrupt`` in the main
  thread, which is what the turn already knows how to unwind. When the kitty
  keyboard protocol is enabled (see :mod:`agent.keys`) the terminal reports
  Ctrl+C as an escape sequence instead of ``0x03``, so the kernel is bypassed
  entirely and the reader re-raises SIGINT itself.
* Submitted lines are queued and handed to the model at the turn's next
  tool-loop seam, so a remark typed while a tool runs still steers the turn
  it was typed into. A line the turn never gets to -- the model answered
  without calling another tool -- is **not** run on the user's behalf: it is
  pasted into the prompt as editable default text (``Agent._steer_requeue``),
  where it can be sent, edited or cleared with Ctrl+U. Auto-submitting it
  would start a turn from a mid-sentence remark the user never reviewed.
* The reader is also how a tool's **confirmation** is answered, without ever
  taking the keyboard away from the line being typed (:meth:`ask_question`).
  The question is announced on the footer and waits; **Tab** moves the focus
  onto it, Tab or Escape moves it back, and ``y``/``n``/Enter (or a typed
  string) answer it there. Handing the tty to ``readline`` instead -- the
  old behaviour -- meant the next character typed into a half-written
  steering line was silently reinterpreted as the answer to a question the
  user had not seen yet.

It is deliberately *not* prompt_toolkit: no history, no completion, no
key-binding framework. It exists to steer a running turn ("actually, use
pandas"), and it degrades to a no-op whenever the terminal is not interactive
or the tty cannot be put into cbreak mode.

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

from agent.keys import LOCK_MODIFIER_BITS, is_ctrl_c_sequence

# Bytes that terminate a CSI/SS3 escape sequence (we swallow them whole so
# arrow keys and friends do not leak into the line buffer).
#
# By the ECMA-48 grammar a final byte is 0x40-0x7E, which is every one of
# ``@A-Z[\]^_`a-z{|}~``. The previous hand-written set (``@ABCDEFGHJKSTfmnsu``
# plus ``~``) was missing ``P``, ``Q`` and ``R``, so an SS3 function key
# (F1-F4 send ``ESC O P``-``S``) never closed its sequence and left the reader
# mid-escape, silently eating every keystroke after it.
#
# Digits and ``;`` are deliberately *not* here: they are parameter bytes, and
# the ``CSI u`` sequences for the keypad and F13+ run to five digits
# (``CSI 57414;129u``). Treating ``5`` as a final would cut that sequence in
# half and leave the tail to leak into the line as text.
_CSI_FINAL = frozenset(range(0x40, 0x7F))


# --------------------------------------------------------------------------
# Navigation keys
# --------------------------------------------------------------------------
#
# The reader swallows escape sequences, and it has to *decode* the handful
# that move the caret rather than dropping them. Two encodings are live at
# once and a decoder written against either alone is silently wrong on half
# the terminals:
#
# - the **legacy** one (``CSI D``, ``CSI 1~``, ``ESC O H``) -- what a terminal
#   that ignores the kitty protocol sends, and what prompt_toolkit's table
#   covers;
# - the **kitty protocol** one, which is pushed at REPL start and *stays*
#   pushed (see :func:`agent.keys.enable_kitty_keyboard`). It keeps arrows in
#   the letter form but adds a modifier field, so a plain Left arrives as
#   ``CSI 1;129D`` once num_lock is on -- the very case that made
#   prompt_toolkit type escape codes into its own prompt.
#
# The final letter and the number in the tilde/``u`` forms are the spec's own
# tables (mirrored from ``agent.keys``), not a hand-picked set.

# ``CSI <letter>`` -- the functional keys that keep this form under the
# protocol. Up/Down are deliberately absent: this is a one-line field, so
# they mean nothing here and are swallowed like any other unbound key.
_NAV_LETTERS = {
    "C": "right",
    "D": "left",
    "F": "end",
    "H": "home",
}

# ``CSI <number> ~``
_NAV_TILDES = {
    1: "home",
    3: "delete",
    4: "end",
    7: "home",
    8: "end",
}

# ``CSI <codepoint> u`` -- only the codepoints with a caret meaning. The
# keypad's own navigation codes (``57423`` Home and friends) are left out on
# purpose: they are a separate table with no modifiers, and treating a keypad
# digit as Home would be worse than ignoring it.
_NAV_CODEPOINTS = {
    3: "delete",
    7: "home",
    8: "end",
}

# Modifier bits that turn a plain move into a word move, as readline's
# Alt+B/Alt+F and most terminals' Ctrl+Left/Right do. Alt is included
# because that is the spelling that works everywhere; a terminal that sends
# Esc-prefixed keys gets ``ESC b``, which the reader does not decode (it
# drops the prefix and inserts the letter).
_WORD_MODIFIER_BITS = 2 | 4  # alt | ctrl


def _nav_action(sequence: str) -> str | None:
    """The editing action an escape *sequence* names, or None.

    *sequence* is the whole thing including the leading ``ESC``. Everything
    the reader does not bind -- function keys, PageUp, the ``modifyOtherKeys``
    spellings -- returns None and is swallowed as before, which is what keeps
    an unknown key from leaking its bytes into the line.
    """
    if not sequence.startswith("\x1b"):
        return None
    body = sequence[1:]
    if body[:1] == "O":
        # SS3 (application cursor mode): ``ESC O H``. One key, no parameters,
        # so a modifier would have to arrive as a separate encoding.
        return _NAV_LETTERS.get(body[1:])
    if body[:1] != "[" or not body:
        return None
    body = body[1:]
    if not body:
        return None

    final = body[-1]
    if not final.isalpha() and final != "~":
        return None
    params = body[:-1].split(";") if body[:-1] else []

    def bits() -> int:
        """The held modifiers, with the lock bits stripped.

        Lock modifiers describe the keyboard's state, not a key the user is
        holding, so ``CSI 1;129D`` is plain Left -- not some shifted variant
        of it.
        """
        if len(params) < 2:
            return 0
        try:
            return (int(params[1]) - 1) & ~LOCK_MODIFIER_BITS
        except ValueError:
            return 0

    action: str | None = None
    if final in _NAV_LETTERS:
        action = _NAV_LETTERS[final]
    elif final == "~":
        try:
            action = _NAV_TILDES.get(int(params[0]))
        except (ValueError, IndexError):
            return None
    elif final == "u":
        try:
            action = _NAV_CODEPOINTS.get(int(params[0]))
        except (ValueError, IndexError):
            return None
    else:
        return None

    if action in ("left", "right") and bits() & _WORD_MODIFIER_BITS:
        return f"word-{action}"
    return action


def _word_left(buffer: list[str], pos: int) -> int:
    """Caret position after a backwards word move: skip spaces, then a word."""
    index = pos
    while index > 0 and buffer[index - 1].isspace():
        index -= 1
    while index > 0 and not buffer[index - 1].isspace():
        index -= 1
    return index


def _word_right(buffer: list[str], pos: int) -> int:
    """Caret position after a forwards word move: skip a word, then spaces."""
    index = pos
    length = len(buffer)
    while index < length and not buffer[index].isspace():
        index += 1
    while index < length and buffer[index].isspace():
        index += 1
    return index


def _apply_action(action: str, buf: list[str], pos: int) -> int:
    """Apply one line-editing *action* to *buf*; return the new caret.

    Shared by the steering line and by a string answer typed while the focus
    is on a pending question: they are the same editor over two buffers.
    """
    if action == "left":
        return max(0, pos - 1)
    if action == "right":
        return min(len(buf), pos + 1)
    if action == "home":
        return 0
    if action == "end":
        return len(buf)
    if action == "backspace":
        if pos > 0:
            del buf[pos - 1]
            return pos - 1
        return pos
    if action == "delete":
        if pos < len(buf):
            del buf[pos]
        return pos
    if action == "word-left":
        return _word_left(buf, pos)
    if action == "word-right":
        return _word_right(buf, pos)
    if action == "kill-to-end":
        del buf[pos:]
        return pos
    if action == "kill-word":
        start = _word_left(buf, pos)
        del buf[start:]
        return start
    if action == "clear":
        buf.clear()
        return 0
    return pos


#: The line-editing action a single control byte names, if any. One table, so
#: the steering line and a question's answer field cannot drift apart.
_EDIT_BYTES = {
    0x7F: "backspace",
    0x08: "backspace",  # Ctrl+H
    0x04: "delete",  # Ctrl+D
    0x15: "clear",  # Ctrl+U
    0x0B: "kill-to-end",  # Ctrl+K
    0x17: "kill-word",  # Ctrl+W
    0x01: "home",  # Ctrl+A
    0x02: "left",  # Ctrl+B
    0x05: "end",  # Ctrl+E
    0x06: "right",  # Ctrl+F
}


def _is_tab_sequence(sequence: str) -> bool:
    """True for an escape sequence that means plain Tab or Shift+Tab.

    The kitty keyboard protocol reports Tab as ``CSI 9 u`` (with a modifier
    field once Shift is held) rather than as ``0x09``, exactly as it does for
    every other functional key. The legacy Shift+Tab is ``CSI Z``.
    """
    body = sequence[1:]
    if body[:1] != "[":
        return False
    body = body[1:]
    if body == "Z":
        return True
    return body.endswith("u") and body[:-1].split(";")[0] == "9"


class _PendingQuestion:
    """A question the agent asked mid-turn, waiting to be answered.

    The caller blocks on :attr:`event`; the reader thread fills :attr:`result`
    and sets it. Everything is read and written under the owning
    :class:`SteerInput`'s lock, except :attr:`event` itself.
    """

    __slots__ = ("kind", "message", "default", "buffer", "pos", "event", "result")

    def __init__(self, kind: str, message: str, default) -> None:
        self.kind = kind
        self.message = message
        self.default = default
        # Only a string question uses these: a confirmation is answered by a
        # single key and has nothing to edit.
        self.buffer: list[str] = []
        self.pos = 0
        self.event = threading.Event()
        self.result: str | None = None


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
        # Caret index into ``_buffer``: the position the next character is
        # inserted at, and what the frontend draws the block cursor over.
        self._pos = 0
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
        # A question the agent asked mid-turn (a tool's confirmation), waiting
        # for the user to move the focus to it with Tab. It never takes the
        # keyboard on its own: the next keystroke the user types must keep
        # meaning what it meant a moment ago.
        self._question: _PendingQuestion | None = None
        # Which of the two input fields the keyboard is attached to:
        # "line" (the steering line, the default) or "ask" (the question).
        self._focus = "line"

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
        queue and the main loop carries it into the prompt as editable text
        instead of running it (see ``Agent._steer_requeue``).
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

    def caret(self) -> int:
        """Return the caret's index within :meth:`line` (0 .. ``len(line)``).

        Published alongside the line because the two describe one thing: the
        frontend paints its block cursor over the character at this index, so a
        caret that moved left is drawn *there* rather than always at the end.
        """
        with self._lock:
            return self._pos

    def focus(self) -> str:
        """Which field the keyboard is attached to: ``"line"`` or ``"ask"``."""
        return self._focus

    def question(self) -> tuple[str, str] | None:
        """The pending question as ``(kind, message)``, or None."""
        q = self._question
        return None if q is None else (q.kind, q.message)

    def answer(self) -> tuple[str, int]:
        """The pending question's answer text and caret (for the footer)."""
        with self._lock:
            q = self._question
            if q is None:
                return "", 0
            return "".join(q.buffer), q.pos

    # ── asking the user without stealing the keyboard ──────────────────────

    def ask_question(self, kind: str, message: str, default="") -> str | None:
        """Ask *message* and block until the user answers; None if none came.

        *kind* is ``"confirm"`` (the answer is ``"yes"``/``"no"``, and
        *default* a bool) or ``"string"`` (the answer is what was typed, and
        *default* the value returned for an empty one).

        This is the whole point of the focus design: the question does **not**
        take the terminal. The steering line keeps the keyboard, the question
        is announced on the footer, and only Tab moves the focus onto it. A
        modal prompt would mean the next character typed into a half-written
        steering line was silently reinterpreted as the answer to a question
        the user had not seen yet -- which is exactly what handing the tty to
        ``readline`` did.

        Returns None when nothing can answer (no reader armed, a question
        already pending, or the turn ended first); the caller then falls back
        to an ordinary prompt or a safe default.
        """
        if kind not in ("confirm", "string"):
            return None
        q = _PendingQuestion(kind, message, default)
        with self._lock:
            if self._question is not None or not self._armed:
                return None
            self._question = q
            # Focus stays on the line: the question waits for an explicit Tab.
            self._focus = "line"
        self._changed()
        try:
            q.event.wait()
        except BaseException:
            # Ctrl+C while waiting. The turn unwinds on it, so the question
            # must not survive: a left-behind one would refuse every later
            # question and hang the tool that asked it.
            self._resolve(q, None)
            raise
        return q.result

    def _resolve(self, q: _PendingQuestion, value: str | None) -> bool:
        """Answer *q* with *value* and wake whoever asked. False if stale."""
        with self._lock:
            if self._question is not q:
                return False
            self._question = None
            self._focus = "line"
            q.result = value
        q.event.set()
        self._changed()
        return True

    def _cancel_question(self) -> None:
        """Answer any pending question with nothing, so no tool hangs on it."""
        q = self._question
        if q is not None:
            self._resolve(q, None)

    def _set_focus(self, focus: str) -> None:
        """Move the focus, redrawing only if it actually moved."""
        with self._lock:
            if self._focus == focus:
                return
            self._focus = focus
        self._changed()

    def _toggle_focus(self) -> None:
        """Tab: hand the keyboard between the steering line and a question."""
        with self._lock:
            if self._question is None:
                return
            self._focus = "line" if self._focus == "ask" else "ask"
            # A string answer is typed at the end of whatever is already there.
            if self._focus == "ask" and self._question.kind == "string":
                self._question.pos = len(self._question.buffer)
        self._changed()

    # ── lifecycle ──────────────────────────────────────────────────────────

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

    def _disarm(self) -> None:
        """Stop the reader thread and restore the tty, keeping what is typed.

        The read loop reads one fd in cbreak mode, so anything that wants the
        tty back (an ``ask_*`` question, a subprocess) has to disarm it first.
        Nothing is discarded here: the buffer, its caret and the queue are the
        user's, not the reader's.
        """
        self._cancel_question()
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

    def stop(self) -> None:
        """Disarm the reader, restore the tty, and drop the partial line."""
        self._disarm()
        with self._lock:
            self._buffer.clear()
            self._pos = 0
            self._focus = "line"

    def suspend(self) -> None:
        """Disarm temporarily, keeping the partial line exactly as it is.

        A steered turn can be interrupted by a subprocess taking over the
        terminal (``run_subprocess``, an editor). That needs the tty in its
        normal state, so the reader has to go -- but the user is
        *mid-sentence*, not finished. Discarding the line here is what made
        steering input vanish when a tool took the terminal: the buffer was
        dropped on the way in and the row came back empty once the tool had
        run. Only Enter submits and only Ctrl+U clears; an interruption is
        neither.

        Questions the agent asks (``ask_confirm``/``ask_string``) no longer
        come through here at all: they are answered on the footer through
        :meth:`ask_question`, which keeps the reader armed.

        Paired with :meth:`resume`. :meth:`stop` is the other half of that
        distinction: it is for the end of a turn, when the prompt owns the
        terminal again and the partial line has no home left.
        """
        self._disarm()

    def resume(self) -> bool:
        """Re-arm after a :meth:`suspend`, keeping the line and the queue."""
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
                # A lone Escape is indistinguishable from the start of a
                # sequence until something follows it. Nothing has for a whole
                # timeout, so it was the key itself: end the sequence, rather
                # than swallowing the next keystroke an arbitrary time later.
                if self._esc_mode:
                    self._esc_mode = 0
                    self._esc_buf = b""
                    # While the focus is on a question, a lone Escape is the
                    # way back to the steering line.
                    self._set_focus("line")
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
        """Apply a single input byte to whichever field has the focus.

        Escape sequences are consumed whole, so an arrow key or a function key
        never leaks its bytes into the line as stray characters. The few that
        name a caret movement are decoded rather than dropped
        (:func:`_nav_action`); everything else -- F-keys, PageUp, any unknown
        sequence -- is swallowed exactly as before.

        The two fields are the steering line (the default) and a question the
        agent is waiting on. Tab is the only thing that moves the focus
        between them, and it is free here: it was an ignored control byte.
        """
        if self._esc_mode:
            if self._esc_mode == 1:
                if byte in (ord("["), ord("O")):
                    self._esc_mode = 2
                    self._esc_buf = bytes([byte])
                else:
                    # Not a sequence after all: ``ESC <key>``, or a lone
                    # Escape the read loop has not timed out yet. Drop the
                    # ESC and let the byte do its ordinary job -- treating it
                    # as part of an escape swallowed whatever character
                    # followed a stray Escape press. On a focused question
                    # that ESC was the way out.
                    self._esc_mode = 0
                    self._esc_buf = b""
                    self._set_focus("line")
                    self._handle_byte(byte)
                return
            self._esc_buf += bytes([byte])
            if byte not in _CSI_FINAL:
                return
            sequence = (b"\x1b" + self._esc_buf).decode("ascii", "replace")
            self._esc_mode = 0
            self._esc_buf = b""
            if is_ctrl_c_sequence(sequence):
                self._interrupt()
                return
            if _is_tab_sequence(sequence):
                self._toggle_focus()
                return
            action = _nav_action(sequence)
            if action is not None:
                self._edit(action)
            return
        if byte == 0x09:  # Tab: the one key that moves the focus
            self._toggle_focus()
            return
        if byte == 0x1B:
            self._esc_mode = 1
            return
        if self._focus == "ask":
            self._handle_ask_byte(byte)
            return
        if byte in (0x0D, 0x0A):  # Enter
            text = self.line().strip()
            with self._lock:
                self._buffer.clear()
                self._pos = 0
                if text:
                    self._queue.append(text)
            self._changed()
            if text:
                self._safe(self._on_submit, text)
            return
        if byte in (0x03, 0x1A, 0x1C):
            # Ctrl+C / Ctrl+\ / Ctrl+Z. ISIG is left enabled, so the kernel
            # turns Ctrl+C into SIGINT (and Ctrl+\ into SIGQUIT) before these
            # bytes ever reach us. Never buffer them.
            return
        # Line editing, in the readline spelling where one exists. All of it
        # goes through `_edit` so that "did anything move" is decided in one
        # place -- a no-op must not redraw the footer row.
        action = _EDIT_BYTES.get(byte)
        if action is not None:
            self._edit(action)
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
                self._buffer.insert(self._pos, char)
                self._pos += 1
                changed = True
            else:
                changed = False
        if changed:
            self._changed()

    # ── line editing ─────────────────────────────────────────────────────

    # ── asking the user without stealing the keyboard ──────────────────────

    def _handle_ask_byte(self, byte: int) -> None:
        """A keystroke typed while the pending question has the focus."""
        q = self._question
        if q is None:
            return
        if byte in (0x03, 0x1A, 0x1C):
            return  # Ctrl+C & co: never an answer, never buffered
        if q.kind == "confirm":
            if byte in (0x79, 0x59):  # y / Y
                self._resolve(q, "yes")
            elif byte in (0x6E, 0x4E):  # n / N
                self._resolve(q, "no")
            elif byte in (0x0D, 0x0A):  # Enter: the default answer
                self._resolve(q, "yes" if q.default else "no")
            return
        if byte in (0x0D, 0x0A):  # Enter: submit the typed answer
            with self._lock:
                value = "".join(q.buffer)
            self._resolve(q, value if value.strip() else str(q.default))
            return
        action = _EDIT_BYTES.get(byte)
        if action is not None:
            self._edit_answer(action)
            return
        if byte < 0x20 or byte == 0x7F:
            return
        try:
            char = bytes([byte]).decode("utf-8")
        except UnicodeDecodeError:
            return
        if not char.isprintable():
            return
        with self._lock:
            if sum(len(c) for c in q.buffer) < self._max_len:
                q.buffer.insert(q.pos, char)
                q.pos += 1
                changed = True
            else:
                changed = False
        if changed:
            self._changed()

    # ── line editing ───────────────────────────────────────────────────────

    def _edit(self, action: str) -> None:
        """Apply a line-editing *action*, notifying only if something moved.

        The notification is what redraws the footer row, and that row shows
        the caret as well as the text -- so a pure caret move notifies too.
        A genuine no-op (Left at the start of the line, Backspace on an empty
        one) does not, which is what keeps a held-down key from redrawing the
        row a few hundred times a second for nothing.

        The action lands on whichever field has the focus, so the same keys
        edit the steering line and a question's answer.
        """
        if self._focus == "ask":
            self._edit_answer(action)
            return
        with self._lock:
            before = (list(self._buffer), self._pos)
            self._pos = _apply_action(action, self._buffer, self._pos)
            moved = (self._buffer, self._pos) != before
        if moved:
            self._changed()

    def _edit_answer(self, action: str) -> None:
        """The same editor, applied to the pending question's answer."""
        with self._lock:
            q = self._question
            if q is None:
                return
            before = (list(q.buffer), q.pos)
            q.pos = _apply_action(action, q.buffer, q.pos)
            moved = (q.buffer, q.pos) != before
        if moved:
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
        """Notify the caller that the line or the caret moved.

        The two are published together, and the frontend draws both, so a
        caret move is as much a change to the row as an inserted character.
        """
        if self._on_change is not None:
            self._safe(self._on_change)

    @staticmethod
    def _safe(func, *args) -> None:
        """Call a callback, swallowing any error from the reader thread."""
        try:
            func(*args)
        except Exception:
            pass