"""Sticky footer for the classic REPL, based on scroll regions (DECSTBM).

During an assistant turn the prompt_toolkit prompt is off-screen and the
response is streamed to stdout. This module reserves the bottom
``FOOTER_LINES`` rows of the terminal for a persistent status line (model,
session, memory usage, ...) so it stays visible while the response scrolls.

The mechanism is the terminal's native **scroll region** (``DECSTBM``,
``\\x1b[<top>;<bottom>r``), the same one vim/less use:

1. ``start()`` sets the scroll region to the top ``H - FOOTER_LINES`` rows.
   The bottom ``FOOTER_LINES`` rows are then physically excluded from
   scrolling: no matter how much output is printed, they stay put.
2. The status line is drawn into the reserved rows with
   :meth:`Footer.update_status`, which briefly moves the cursor there and
   back. No cursor tracking, no guarded writes, no output interception —
   all existing output code works unchanged.
3. ``stop()`` resets the scroll region (``\\x1b[r``), blanks the reserved
   rows, and leaves the cursor at the bottom of the screen so the
   prompt_toolkit prompt renders normally.

The footer is only armed during a turn. It degrades to a no-op when stdout
is not a TTY (e.g. output is redirected), so piped/redirected sessions are
unaffected.
"""

from __future__ import annotations

import os
import shutil
import sys

# Number of lines reserved for the footer (separator + status line).
FOOTER_LINES = 2

# ANSI SGR sequences for the footer styling.
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
_ACCENT = "\x1b[36m"  # cyan
_LABEL = "\x1b[90m"  # bright black (gray)


class Footer:
    """Reserve the bottom rows of the terminal for a sticky status line."""

    def __init__(self) -> None:
        self._active = False
        self._term_height = 24
        self._term_width = 80
        self._status_text = ""

    # ── terminal helpers ────────────────────────────────────────────────

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

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Arm the footer: set the scroll region and reserve the bottom rows.

        No-op if stdout is not a TTY.
        """
        if self._active or not self._is_tty():
            return
        self._term_width, self._term_height = self._read_terminal_size()
        if self._term_height <= FOOTER_LINES + 1:
            # Terminal too small to reserve rows; disable the footer.
            return
        footer_top = self._term_height - FOOTER_LINES + 1
        try:
            # Set the scroll region to the rows above the footer zone.
            sys.stdout.write(f"\x1b[1;{footer_top - 1}r")
            # Move into the footer zone and blank it.
            sys.stdout.write(f"\x1b[{footer_top};1H")
            for _ in range(FOOTER_LINES):
                sys.stdout.write("\x1b[2K\r\n")
            sys.stdout.flush()
        except Exception:
            return
        self._active = True

    def stop(self) -> None:
        """Disarm the footer: reset the scroll region and clean up.

        The reserved rows are blanked and the cursor is left at the bottom
        of the screen, so the prompt_toolkit prompt renders normally.
        """
        if not self._active:
            return
        self._active = False
        try:
            # Reset the scroll region to the full screen.
            sys.stdout.write("\x1b[r")
            # Move to the footer zone and blank it.
            footer_top = self._term_height - FOOTER_LINES + 1
            sys.stdout.write(f"\x1b[{footer_top};1H")
            for _ in range(FOOTER_LINES):
                sys.stdout.write("\x1b[2K\r\n")
            # Leave the cursor at the bottom of the screen.
            sys.stdout.write(f"\x1b[{self._term_height};1H")
            sys.stdout.flush()
        except Exception:
            pass
        self._status_text = ""

    @property
    def active(self) -> bool:
        return self._active

    # ── footer rendering ────────────────────────────────────────────────

    def update_status(self, status_line: str) -> None:
        """Render *status_line* in the reserved rows.

        The cursor is briefly moved into the footer zone, the line is
        written, and the cursor is moved back to the content area (bottom
        of the scroll region). No-op when the footer is not armed.
        """
        if not self._active:
            return
        if status_line == self._status_text:
            return
        self._status_text = status_line
        try:
            footer_top = self._term_height - FOOTER_LINES + 1
            # Save nothing: the content cursor is always at the bottom of
            # the scroll region after normal output, so we simply move back
            # there after drawing.
            sep = _DIM + "─" * self._term_width + _RESET
            sys.stdout.write(f"\x1b[{footer_top};1H")
            sys.stdout.write(f"\x1b[2K{sep}")
            sys.stdout.write(f"\x1b[{footer_top + 1};1H")
            sys.stdout.write(f"\x1b[2K{status_line}")
            # Move the cursor back to the bottom of the scroll region.
            sys.stdout.write(f"\x1b[{footer_top - 1};1H")
            sys.stdout.flush()
        except Exception:
            pass

    # Rich's Console may call these if the footer is ever set as its file.
    def write(self, text: str) -> None:
        sys.stdout.write(text)

    def flush(self) -> None:
        sys.stdout.flush()

    def isatty(self) -> bool:
        return sys.stdout.isatty()
