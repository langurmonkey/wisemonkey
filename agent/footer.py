"""Sticky footer for the classic REPL, based on scroll regions (DECSTBM).

During an assistant turn the prompt_toolkit prompt is off-screen and the
response is streamed to stdout. This module reserves the bottom
``FOOTER_LINES`` rows of the terminal for a persistent status line (model,
session, memory usage, ...), an optional steering input line, and a key-hint
bar, so they stay visible while the response scrolls.

The mechanism is the terminal's native **scroll region** (``DECSTBM``,
``\\x1b[<top>;<bottom>r``), the same one vim/less use:

1. ``start()`` sets the scroll region to the top ``H - FOOTER_LINES`` rows.
   The bottom ``FOOTER_LINES`` rows are then physically excluded from
   scrolling: no matter how much output is printed, they stay put.
2. The rows are drawn with :meth:`Footer.update_status`, which briefly moves
   the cursor there and back. No cursor tracking, no guarded writes, no
   output interception — all existing output code works unchanged.
3. ``stop()`` resets the scroll region (``\\x1b[r``), blanks the reserved
   rows, and leaves the cursor at the bottom of the screen so the
   prompt_toolkit prompt renders normally.

Concurrency
-----------
Once the footer is armed, fd 1 has **two** independent writers: the footer
itself and whatever is streaming the response (``mdstream``/Rich, and any
tool output). Without serialization a status redraw can be split by a
streamed chunk, leaving the footer garbled. ``start()`` therefore installs a
locking proxy over ``sys.stdout`` for the duration of the turn, so *every*
write to fd 1 is atomic with respect to the others. ``stop()`` restores the
original object. Rich resolves ``sys.stdout`` lazily (``Console.file`` is a
property), so the proxy is picked up without recreating the console.

The footer also runs its own spinner animation thread while a stage is
active (see :meth:`Footer.set_spinner`). That is a *third* writer, so its
redraws go through the same lock, and it draws only within the reserved
rows — it never touches the scroll region.

Optionally, with ``agent.footer_debug_bytes: true``, every byte the footer
writes is also appended (repr'd) to ``$XDG_STATE_HOME/wisemonkey/footer.log``
so terminal rendering problems can be diagnosed after the fact.

The footer is only armed during a turn. It degrades to a no-op when stdout
is not a TTY (e.g. output is redirected), so piped/redirected sessions are
unaffected.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time

from agent.palette import PALETTE

# Number of lines reserved for the footer:
# separator + status + steering input + key hints.
FOOTER_LINES = 4

# Braille frames and tick interval for the footer's own spinner. This is the
# same glyph sequence Rich uses for its "dots" spinner; the footer animates
# it itself because starting a Rich `Live` display would wrap sys.stdout in a
# FileProxy, which turns the footer's cursor addressing into literal text
# (see :func:`real_stream`).
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.08

# Label of the reasoning indicator on the key-hints row. It carries an
# animation frame in front of it exactly like the stage spinner, so the glyph
# is part of the label (the frame goes before it, not in place of it).
THINKING_LABEL = "💡 Thinking..."

# Sentinel for "the transient row has not been drawn yet". Distinct from "",
# which means "drawn, and empty": the spinner thread relies on that
# difference to decide whether a tick actually changes anything.
_UNSET = "\x00unset\x00"

# ANSI SGR sequences for the footer styling, generated from the shared
# palette so the footer cannot drift from the Rich theme above it. It writes
# straight to the terminal and never goes through Rich, which is exactly why
# it used to need its own hand-written colours.
_SGR = PALETTE.footer_sgr()
_DIM = _SGR["dim"]
_RESET = _SGR["reset"]
_ACCENT = _SGR["accent"]
_LABEL = _SGR["label"]
_WHITE = _SGR["text"]
_KEY = _SGR["key"]
_BAR_BG = _SGR["bar_bg"]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Serializes every write to stdout while the footer is armed. Module level so
# that anything else needing to coordinate (tests, diagnostics) can share it.
WRITE_LOCK = threading.RLock()


def visible_len(text: str) -> int:
    """Length of *text* ignoring ANSI escape sequences."""
    return len(_ANSI_RE.sub("", text))


def clip_ansi(text: str, width: int, ellipsis: str = "…") -> str:
    """Clip *text* (which may contain SGR codes) to *width* visible columns.

    The styling is preserved up to the cut point and a final reset is
    appended so no colour bleeds into the rest of the line.
    """
    if width <= 0:
        return ""
    if visible_len(text) <= width:
        return text
    if width <= len(ellipsis):
        return _ANSI_RE.sub("", text)[:width]

    out: list[str] = []
    shown = 0
    pos = 0
    styled = False
    while pos < len(text) and shown < width - len(ellipsis):
        match = _ANSI_RE.match(text, pos)
        if match:
            out.append(match.group())
            styled = True
            pos = match.end()
            continue
        out.append(text[pos])
        pos += 1
        shown += 1
    out.append(ellipsis)
    out.append(_RESET if styled else "")
    return "".join(out)


class _LockedWriter:
    """A stdout proxy that serializes writes and can log raw bytes.

    Proxies the handful of stream attributes anything may touch (``fileno``,
    ``isatty``, ``encoding``, ``buffer``, ...). Every ``write`` is performed
    under :data:`WRITE_LOCK`, so footer redraws and streamed chunks never
    interleave mid-escape-sequence.
    """

    def __init__(self, stream, log_path: str | None = None) -> None:
        self._stream = stream
        self._log_path = log_path

    @property
    def rich_proxied_file(self):
        """The object we wrap.

        Rich's ``FileProxy``/``Console.file`` unwrap this attribute to recover
        the *real* stream, so a ``FileProxy`` installed on top of this proxy
        (by a ``console.status()`` spinner) still writes straight to the
        terminal, under our lock.
        """
        return self._stream

    # Proxying

    def __getattr__(self, name: str):
        return getattr(self._stream, name)

    def write(self, text: str):
        with WRITE_LOCK:
            if self._log_path is not None:
                try:
                    with open(self._log_path, "a", encoding="utf-8") as log:
                        log.write(repr(text) + "\n")
                except OSError:
                    pass
            return self._stream.write(text)

    def writelines(self, lines) -> None:
        with WRITE_LOCK:
            for line in lines:
                self._stream.write(line)

    def flush(self) -> None:
        with WRITE_LOCK:
            self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def real_stream(stream=None):
    """Unwrap *stream* down to the stream that actually reaches the terminal.

    Rich's ``console.status()`` spinner wraps ``sys.stdout`` in a
    ``FileProxy``, which *interprets* ANSI escapes: text written to it is
    decoded, markup-parsed and re-rendered. The footer's cursor-addressing
    sequences must never go through that path — they would be decoded and
    printed as literal ``[24;1H``-style text. ``FileProxy`` exposes the
    stream it wraps as ``rich_proxied_file``, so skip over it.

    Our own :class:`_LockedWriter` is deliberately *not* unwrapped: the
    footer writes through it so that the shared :data:`WRITE_LOCK` still
    serializes footer redraws against streamed output.
    """
    stream = sys.stdout if stream is None else stream
    seen = 0
    while seen < 10 and not isinstance(stream, _LockedWriter):
        nxt = getattr(stream, "rich_proxied_file", None)
        if nxt is None or nxt is stream:
            break
        stream = nxt
        seen += 1
    return stream


def _debug_log_path() -> str:
    """Return the path of the footer byte log."""
    state = os.environ.get("XDG_STATE_HOME") or os.path.expanduser(
        "~/.local/state"
    )
    return os.path.join(state, "wisemonkey", "footer.log")


class Footer:
    """Reserve the bottom rows of the terminal for a sticky status line."""

    def __init__(self, debug_bytes: bool = False) -> None:
        self._active = False
        self._term_height = 24
        self._term_width = 80
        self._status_text = ""
        # `_UNSET` marks "the transient row has not been drawn yet", which is
        # distinct from "" ("drawn, and empty").
        self._input_text: str = _UNSET
        # Whether the "💡 Thinking..." indicator is shown on the key-hints row.
        # Set via `set_thinking`; `start` clears it, since a new turn is not
        # reasoning.
        self._thinking = False
        self._debug_bytes = debug_bytes
        self._saved_stdout = None
        # Spinner state. `_spinner_text` is the stage label ("" = idle);
        # `_frame` is the current braille glyph index.
        self._spinner_text = ""
        self._frame = 0
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None
        # The status line and the steering input line are redrawn from the
        # main thread, from the steer reader thread, and from the spinner
        # thread, so they must be serialized on top of the global stdout
        # lock.
        self._lock = threading.RLock()

    # Terminal helpers

    @staticmethod
    def _is_tty() -> bool:
        return sys.stdout.isatty()

    def _read_terminal_size(self) -> tuple[int, int]:
        try:
            size = os.get_terminal_size(sys.stdout.fileno())
            return size.columns, size.lines
        except (OSError, ValueError):
            size = shutil.get_terminal_size((80, 24))
            return size.columns, size.lines

    # Lifecycle

    def clear_prompt_zone(self) -> None:
        """Push the prompt box above the footer zone before the footer arms.

        After prompt_toolkit accepts a tall prompt, the cursor sits on the
        last line of the prompt box. If that line is inside the bottom
        ``FOOTER_LINES`` rows, ``start()`` will blank it and streamed output
        will overwrite the bottom of the box. Emitting ``FOOTER_LINES``
        newlines scrolls the box fully clear of the reserved rows.

        No-op when stdout is not a TTY.
        """
        if not self._is_tty():
            return
        out = real_stream()
        try:
            out.write("\n" * FOOTER_LINES)
            out.flush()
        except Exception:
            pass

    def start(self) -> None:
        """Arm the footer: set the scroll region and reserve the bottom rows.

        No-op if stdout is not a TTY or the terminal is too small.
        """
        # A new turn means no reasoning is in progress: clear any indicator
        # the previous turn left behind (it may have ended mid-reasoning).
        self._thinking = False
        if self._active or not self._is_tty():
            return
        self._term_width, self._term_height = self._read_terminal_size()
        if self._term_height <= FOOTER_LINES + 1:
            # Terminal too small to reserve rows; disable the footer.
            return

        # Serialize all writes to fd 1 for the duration of the turn, and
        # optionally log them.
        self._saved_stdout = sys.stdout
        if self._debug_bytes:
            path = _debug_log_path()
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            except OSError:
                path = None
        else:
            path = None
        sys.stdout = _LockedWriter(self._saved_stdout, path)

        footer_top = self._term_height - FOOTER_LINES + 1
        out = real_stream()
        try:
            # Set the scroll region to the rows above the footer zone.
            out.write(f"\x1b[1;{footer_top - 1}r")
            self._blank_footer_rows(out)
            # Leave the cursor at the bottom of the scroll region, where
            # streamed output expects to continue.
            out.write(f"\x1b[{footer_top - 1};1H")
            out.flush()
        except Exception:
            self._restore_stdout()
            return
        self._active = True

    def _blank_footer_rows(self, out) -> None:
        """Erase each reserved row without moving the cursor off the screen.

        The obvious spelling is ``CUP`` to the first row followed by
        ``\\x1b[2K\\r\\n`` per row, but the final newline lands *below the last
        screen row* and makes the terminal scroll: with the scroll region reset
        (which is the case in :meth:`stop`) the whole visible screen shifts up
        one line, and the REPL's turn header is the line that gets eaten. This
        is why "Prompt processed" occasionally appears to overwrite the
        banner above it -- the two are unrelated, and the scroll happens
        first.

        Addressing each row in turn keeps every write inside the screen, so
        blanking is idempotent and nothing above the footer moves.
        """
        top = self._term_height - FOOTER_LINES + 1
        for row in range(top, top + FOOTER_LINES):
            out.write(f"\x1b[{row};1H\x1b[2K")

    def stop(self) -> None:
        """Disarm the footer: reset the scroll region and clean up.

        The reserved rows are blanked and the cursor is left at the bottom
        of the screen, so the prompt_toolkit prompt renders normally.
        """
        # Stop any animation *before* blanking the rows, so the spinner
        # thread cannot redraw into a torn-down footer. Clearing the
        # reasoning indicator first is what lets `set_spinner("")` stop the
        # thread: on its own it leaves the animation running for a
        # still-active indicator.
        self._thinking = False
        self.set_spinner("")
        if not self._active:
            return
        self._active = False
        out = real_stream()
        try:
            # Reset the scroll region to the full screen.
            out.write("\x1b[r")
            self._blank_footer_rows(out)
            # Leave the cursor at the bottom of the screen.
            out.write(f"\x1b[{self._term_height};1H")
            out.flush()
        except Exception:
            pass
        finally:
            self._restore_stdout()
        self._status_text = ""
        self._input_text = _UNSET

    def _restore_stdout(self) -> None:
        """Put the original stdout back in place."""
        if self._saved_stdout is not None:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            sys.stdout = self._saved_stdout
            self._saved_stdout = None

    @property
    def active(self) -> bool:
        return self._active

    # Spinner

    def set_spinner(self, text: str) -> None:
        """Show an animated spinner with label *text*; "" clears it.

        The footer animates this itself rather than handing the job to a
        Rich ``Live`` display: ``Live`` replaces ``sys.stdout`` with a
        ``FileProxy``, which decodes ANSI escapes and would turn the
        footer's cursor addressing into literal text on screen (see
        :func:`real_stream`).

        The animation is a daemon thread that ticks a braille frame and
        redraws. It shares the footer's lock and the global
        :data:`WRITE_LOCK`, and it writes only inside the reserved rows, so
        it cannot disturb the streaming response.

        The spinner is drawn on the right of the *key-hints* row (the
        bottom one), left of the reasoning indicator. That row belongs to
        the footer alone, so a typed steering line -- which lives on the
        row above it -- is never overwritten by an animation frame.
        """
        text = text or ""
        with self._lock:
            if text == self._spinner_text:
                return
            self._spinner_text = text
            if not text:
                # Stop the animation only when nothing else needs a tick: a
                # running reasoning indicator keeps animating on its own.
                if not self._thinking:
                    self._stop_spinner_thread()
                self._frame = 0
                # Force a redraw: the hints row may currently hold a spinner
                # frame that has just gone away. `_redraw` no-ops with nothing
                # animating, so the clearing redraw is issued explicitly.
                self._input_text = _UNSET
                self.update_status(self._status_text, "", force=True)
                return
            if self._active and self._spinner_thread is None:
                self._start_spinner_thread()

        # Redraw immediately so the label appears without waiting a tick.
        self._redraw()

    def set_thinking(self, active: bool) -> None:
        """Show or hide the reasoning indicator on the key-hints row.

        The indicator animates like the stage spinner — its own frame, its
        own label — and sits right of the key bindings, so the model's
        reasoning state is visible without a chat-output marker. It drives
        the animation thread by itself, so reasoning is still animated when
        no stage spinner is showing. A no-op when the state is unchanged.
        """
        with self._lock:
            if active == self._thinking:
                return
            self._thinking = active
            if active:
                if self._active and self._spinner_thread is None:
                    self._start_spinner_thread()
            elif not self._spinner_text:
                self._stop_spinner_thread()
        # The hints row changed even though status and transient did not.
        self.update_status(self._status_text, self._transient(), force=True)

    def _transient(self) -> str:
        """The current transient-row content ("" when it was never drawn)."""
        return "" if self._input_text == _UNSET else self._input_text

    def _start_spinner_thread(self) -> None:
        stop = threading.Event()
        thread = threading.Thread(
            target=self._spin_loop, args=(stop,), daemon=True,
            name="footer-spinner",
        )
        self._spinner_stop = stop
        self._spinner_thread = thread
        thread.start()

    def _stop_spinner_thread(self) -> None:
        thread = self._spinner_thread
        stop = self._spinner_stop
        self._spinner_thread = None
        self._spinner_stop = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            # Bounded join: the loop wakes at least every SPINNER_INTERVAL.
            thread.join(timeout=SPINNER_INTERVAL * 10)

    def _animating(self) -> bool:
        """Whether anything on the key-hints row still needs a tick."""
        return bool(self._spinner_text or self._thinking)

    def _spin_loop(self, stop: threading.Event) -> None:
        while not stop.wait(SPINNER_INTERVAL):
            with self._lock:
                if not self._active or not self._animating():
                    return
                self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
            self._redraw()

    def _indicator(self, label: str, frame: str) -> str:
        """One animated indicator: frame, then *label*, on the bar background."""
        return (
            f"{_BAR_BG} {_ACCENT}{frame}{_RESET}"
            f"{_BAR_BG} {_LABEL}{label}{_RESET}"
        )

    def _hints_right(self) -> str:
        """Right-hand content of the key-hints row, or "".

        The stage spinner and the reasoning indicator, each with its own
        animation frame, styled to sit on the bar background. Both are
        footer-owned, so neither has to defer to typed text.
        """
        frame = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        parts: list[str] = []
        if self._spinner_text:
            parts.append(self._indicator(self._spinner_text, frame))
        if self._thinking:
            parts.append(self._indicator(THINKING_LABEL, frame))
        return "".join(parts)

    # Footer rendering

    def _redraw(self) -> None:
        """Advance one animation frame on the key-hints row.

        Does nothing when neither indicator is showing. The hints row is
        written only by the footer, so there is no other writer to defer to:
        the current transient-row content is passed back unchanged.
        """
        with self._lock:
            if not self._animating():
                return
        self.update_status(self._status_text, self._transient(), force=True)

    def update_status(
        self,
        status_line: str,
        input_line: str = "",
        force: bool = False,
    ) -> None:
        """Render *status_line* and *input_line* in the reserved rows.

        Layout (top to bottom)::

            ────...──── separator
            ⇒ model | session:x | Mem…    status
            ⤷ what you are typing          steering input / queued follow-up
            Ctrl+C: cancel turn | …       key hints  ⠋ ⏳ stage… ⠋ 💡 Thinking…

        The third row is the transient one: it shows *input_line* when the
        caller supplied one (a partially typed steering line, or a queued
        follow-up), and is blank otherwise. The stage spinner and the
        reasoning indicator live on the right of the key-hints row instead,
        so an animation frame never competes with what the user is typing.

        The terminal size is re-read on every redraw so a resize mid-turn
        cannot leave stale geometry behind. The cursor is moved back to the
        bottom of the scroll region afterwards. No-op when the footer is not
        armed, or when neither line changed (pass *force* to bypass that,
        which the spinner does on every tick).
        """
        if not self._active:
            return
        with self._lock:
            transient = input_line
            if not force and (
                status_line == self._status_text
                and transient == self._input_text
            ):
                return
            self._status_text = status_line
            self._input_text = transient
            try:
                # Re-read the size: the window may have been resized mid-turn.
                self._term_width, self._term_height = self._read_terminal_size()
                if self._term_height <= FOOTER_LINES + 1:
                    return
                footer_top = self._term_height - FOOTER_LINES + 1
                width = self._term_width

                # The content cursor is always at the bottom of the scroll
                # region after normal output, so we simply move back there
                # after drawing; no save/restore is needed.
                sep = _DIM + "─" * width + _RESET

                # Hint line: advertised key bindings for the current state.
                # The dark gray bar extends the full terminal width: draw the
                # hint, then the right-hand indicators, then pad with
                # background-colored spaces.
                hint_plain = " Ctrl+C: cancel turn   |   type + ↵ to steer "
                hint = (
                    f" {_KEY}Ctrl{_RESET}{_BAR_BG}+{_KEY}C{_RESET}{_BAR_BG}:{_RESET}"
                    f"{_BAR_BG} cancel turn   {_DIM}|{_RESET}{_BAR_BG}   type + "
                    f"{_KEY}↵{_RESET}{_BAR_BG} to steer {_RESET}"
                )
                # Two spaces of right padding for the *content*: the spinner
                # and the reasoning indicator end two columns short of the
                # edge. The gray bar itself still fills the row to the very
                # end (the trailing background spaces below).
                right_width = max(0, width - len(hint_plain) - 2)
                right = clip_ansi(self._hints_right(), right_width)
                pad = " " * max(0, right_width - visible_len(right))

                # Never let a line wrap into the row below it.
                status_line = clip_ansi(status_line, width)
                transient = clip_ansi(transient, width)

                # A Rich spinner may have wrapped sys.stdout in a
                # FileProxy, which would decode and print these sequences as
                # literal text. Write to the real stream instead.
                out = real_stream()
                with WRITE_LOCK:
                    out.write(f"\x1b[{footer_top};1H")
                    out.write(f"\x1b[2K{sep}")
                    out.write(f"\x1b[{footer_top + 1};1H")
                    out.write(f"\x1b[2K{status_line}")
                    out.write(f"\x1b[{footer_top + 2};1H")
                    out.write(f"\x1b[2K{transient}")
                    out.write(f"\x1b[{footer_top + 3};1H")
                    # `right` ends with its own reset; re-apply the bar
                    # background for the two trailing spaces so the bar
                    # reaches the window edge, not the content.
                    out.write(
                        f"\x1b[2K{_BAR_BG}{hint}{_BAR_BG}{pad}{right}"
                        f"{_BAR_BG}  {_RESET}"
                    )
                    # Move the cursor back to the bottom of the scroll region.
                    out.write(f"\x1b[{footer_top - 1};1H")
                    out.flush()
            except Exception:
                pass

    # Rich's Console may call these if the footer is ever set as its file.

    def write(self, text: str) -> None:
        sys.stdout.write(text)

    def flush(self) -> None:
        sys.stdout.flush()

    def isatty(self) -> bool:
        return sys.stdout.isatty()
