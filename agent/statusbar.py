"""Pinned bottom status bar for the REPL during assistant turns.

Uses the terminal's ANSI scroll region (``ESC[1;{H-n}r``) to reserve the
bottom *n* rows of the terminal. While the region is active, normal output
scrolls only in the upper region and the reserved rows stay pinned, so a
status line (model, session, context usage, turn state) remains visible
while the model streams.

Everything degrades gracefully: if the terminal is not ANSI-capable or the
feature is disabled in config, the bar is simply not drawn.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

# ANSI escape sequences
_SCROLL_REGION = "\x1b[{top};{bottom}r"
_RESET_REGION = "\x1b[r"
_CURSOR_TO_ROW = "\x1b[{row};1H"
_CLEAR_LINE = "\x1b[2K"
_SAVE_CURSOR = "\x1b[s"
_RESTORE_CURSOR = "\x1b[u"

_BAR_ROWS = 2  # reserved rows: 1 separator + 1 status line


class TurnStatusBar:
    """Pin a status line to the bottom of the terminal during a turn.

    Usage::

        bar = TurnStatusBar()
        if bar.available():
            bar.start(model=..., session=..., ctx_total=..., ctx_max=...)
            try:
                ... run the turn, printing normally ...
            finally:
                bar.stop()

    While active, all other terminal output must go through the normal
    console (it scrolls inside the restricted region). Anything that takes
    over the terminal (shell commands, pagers, editors) should call
    :meth:`suspend` / :meth:`resume` around itself.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._active = False
        self._rows = _BAR_ROWS
        self._start_time: float | None = None
        self._base: dict = {}

    # ── capability ───────────────────────────────────────────────────────────

    def available(self) -> bool:
        """Whether a pinned bar can be drawn on this terminal."""
        if not self.enabled:
            return False
        if not sys.stdout.isatty():
            return False
        if os.environ.get("TERM", "") in ("", "dumb"):
            return False
        return shutil.get_terminal_size().lines > self._rows + 4

    @property
    def rows(self) -> int:
        return self._rows

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, **info) -> None:
        """Reserve the bottom rows and draw the initial status."""
        if not self.available():
            return
        rows = shutil.get_terminal_size().lines
        # Restrict scrolling to everything above the reserved rows.
        sys.stdout.write(_SCROLL_REGION.format(top=1, bottom=rows - self._rows))
        self._active = True
        self._start_time = time.time()
        self._draw(**info)

    def update(self, **info) -> None:
        """Redraw the status line with updated values."""
        if self._active:
            self._draw(**info)

    def stop(self) -> None:
        """Release the scroll region and clear the bar."""
        if not self._active:
            return
        sys.stdout.write(_RESET_REGION.format())
        # Move to the reserved rows and clear them so nothing is left behind.
        rows = shutil.get_terminal_size().lines
        sys.stdout.write(_CURSOR_TO_ROW.format(row=rows - self._rows + 1))
        for _ in range(self._rows):
            sys.stdout.write(_CLEAR_LINE + "\n")
        sys.stdout.write(_RESET_REGION.format())
        sys.stdout.flush()
        self._active = False

    def suspend(self) -> None:
        """Temporarily release the region (e.g. before a subprocess)."""
        if self._active:
            sys.stdout.write(_RESET_REGION.format())
            sys.stdout.flush()

    def resume(self, **info) -> None:
        """Re-establish the region after a suspend."""
        if self._active:
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(_SCROLL_REGION.format(top=1, bottom=rows - self._rows))
            self._draw(**info)

    # ── drawing ──────────────────────────────────────────────────────────────

    def _draw(self, **info) -> None:
        rows = shutil.get_terminal_size().lines
        status_row = rows - 1  # last row of the scrollable area + 1
        sep_row = status_row - 1

        model = info.get("model", "?")
        session = info.get("session", "?")
        ctx_total = info.get("ctx_total", 0)
        ctx_max = info.get("ctx_max", 1)
        rate = (ctx_total / ctx_max * 100) if ctx_max else 0.0
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        state = info.get("state", "thinking")

        line = (
            f" {model} │ {session} │ ctx {ctx_total} tks ({rate:.1f}%) │ "
            f"{elapsed:.0f}s │ {state}"
        )
        cols = shutil.get_terminal_size().columns
        line = line[: cols - 1].ljust(cols)

        out = _SAVE_CURSOR
        # Dim separator line
        sys.stdout.write(_CURSOR_TO_ROW.format(row=sep_row))
        sys.stdout.write(_CLEAR_LINE + "\x1b[38;5;240m" + "─" * cols + "\x1b[0m")
        # Status line
        sys.stdout.write(_CURSOR_TO_ROW.format(row=status_row))
        sys.stdout.write(_CLEAR_LINE + "\x1b[2m" + line + "\x1b[0m")
        sys.stdout.write(_RESTORE_CURSOR)
        sys.stdout.flush()