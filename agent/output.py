"""Concrete OutputAdapter implementations for classic REPL and TUI modes."""

from __future__ import annotations

import threading
from typing import Protocol

from rich.console import Console
from rich.prompt import Prompt as RichPrompt, FloatPrompt, Confirm
from rich.rule import Rule
from rich.align import AlignMethod
from rich.text import Text as RichText

from prompt_toolkit.shortcuts import choice
from prompt_toolkit.formatted_text import HTML

from textual.widgets import RichLog

from agent.console import console, theme_dict
from agent.utils import term_width

# OutputAdapter abstraction layer
_active_output: OutputAdapter | None = None

def set_output(ui: OutputAdapter | None) -> None:
    """Set the active PromptUi instance for tools to use."""
    global _active_output
    _active_output = ui

def get_output() -> OutputAdapter:
    """Return the active PromptUi instance, or None if not set."""
    if _active_output:
        return _active_output

    raise RuntimeError("Output adapter can't be None")

class OutputAdapter(Protocol):
    """Abstract interface for output adapters.

    Implementations provide the same prompting primitives that ``rich.prompt``
    and ``prompt_toolkit`` offer, but routed through whichever UI is active
    (classic REPL or full-screen TUI).
    """

    def print(self, text: str, end='\n', indent: int = 0) -> None:
        """Print plain or markup text."""

    def print_rich(self, renderable) -> None:
        """Print a Rich renderable."""

    def newline(self) -> None:
        """Print a blank line."""

    def rule(self, style: str = "dim", title: str = "") -> None:
        """Print a horizontal rule line."""

    def info(self, text: str, indent: int = 0) -> None:
        """Print an info line."""

    def ok(self, text: str, indent: int = 0) -> None:
        """Print a success message."""

    def err(self, text: str, indent: int = 0) -> None:
        """Print an error message."""

    def ask_string(self, message: str, default: str = "") -> str:
        """Prompt the user for a free-form string."""

    def ask_float(self, message: str, default: float = 0.0) -> float:
        """Prompt the user for a floating-point number."""

    def ask_choice(
        self,
        message: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        """Present a list of *options* as ``(value, label)`` tuples and return
        the selected *value*."""

    def ask_confirm(self, message: str, default: bool = False) -> bool:
        """Ask a yes/no question and return the boolean answer."""

    def run_subprocess(self, cmd: list[str]):
        """Run an external program with full terminal control.
        Default: just run it directly (fine for plain-terminal contexts)."""
        import subprocess
        return subprocess.run(cmd)

class RichOutputAdapter(OutputAdapter):
    """OutputAdapter backed by rich.prompt (for classic REPL mode).

    Uses its own ``Console`` instance (with the ``monkee_theme``) so that
    styled output is **not** affected by the ``no_color = True`` / file
    redirection that ``Core._tool_calls`` applies to the shared
    ``agent_console.console`` while capturing tool results.
    """

    def __init__(self) -> None:
        from agent.console import monkee_theme
        self._console = Console(theme=monkee_theme, color_system="truecolor")

    def print(self, text: str, end='\n', indent: int = 0) -> None:
        self._console.print(f"{' ' * indent}{text}", end=end)

    def print_rich(self, renderable) -> None:
        self._console.print(renderable)

    def newline(self) -> None:
        self._console.print()

    def info(self, text: str, indent: int = 0) -> None:
        self._console.print(f"{' ' * indent}[info]⇒[/info] {text}")

    def err(self, text:str, indent: int = 0):
        self._console.print(f"{' ' * indent}[err]⨯[/err] {text}")

    def ok(self, text, indent: int = 0):
        self._console.print(f"{' ' * indent}[ok]✓[/ok] {text}")

    def rule(self, style: str = "dim", title: str = "") -> None:
        self._console.rule(style=style)

    def ask_string(self, message: str, default: str = "") -> str:
        return RichPrompt.ask(message, default=default, console=console)

    def ask_float(self, message: str, default: float = 0.0) -> float:
        return FloatPrompt.ask(message, default=default, console=console)

    def ask_choice(
        self,
        message: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        return choice(
            message=message,
            options=options,
            default=default,
            bottom_toolbar=HTML(
                " <b>↑</b>/<b>↓</b>: select | <b>Enter</b>: accept"
            ),
        )

    def ask_confirm(self, message: str, default: bool = False) -> bool:
        return Confirm.ask(message, default=default, console=console)


class TuiOutputAdapter(OutputAdapter):
    """OutputAdapter backed by the Textual TUI (for --tui mode).

    Each method writes a prompt to the output log, switches the status bar
    to an input mode, and blocks until the user submits a response via the
    TextArea.  A threading.Event is used to wait for the response without
    blocking the Textual event loop.

    IMPORTANT: All widget-access methods are dispatched to the main thread
    via ``call_from_thread`` because ``TuiOutputAdapter`` is typically used from
    a worker thread (the turn-execution thread).
    """

    # ---------------------------------------------------------------------------
    # Style-name resolution  (abstract -> concrete colour tags)
    # ---------------------------------------------------------------------------
    # The ``monkee_theme`` in agent/console.py defines abstract style names
    # (``title``, ``accent``, …).  Textual's RichLog natively supports Rich
    # markup but does *not* load custom themes, so we replace abstract tags
    # with concrete colour tags before writing to the log.
    # ---------------------------------------------------------------------------
    _STYLE_MAP = {}
    for key in theme_dict.keys():
        _STYLE_MAP[f"[{key}]"] = f"[{theme_dict[key]}]"
        _STYLE_MAP[f"[/{key}]"] = f"[/{theme_dict[key]}]"

    _STYLE_KEYS = sorted(_STYLE_MAP.keys(), key=len, reverse=True)

    def _concretize(self, text: str) -> str:
        """Replace abstract style tags with concrete colour tags."""
        for k in self._STYLE_KEYS:
            text = text.replace(k, self._STYLE_MAP[k])
        return text


    def __init__(self, app):
        """
        Parameters
        ----------
        app : WisemonkeyTui
            The running Textual application instance.
        """
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from agent.tui import WisemonkeyTui
        self._app: "WisemonkeyTui" = app
        self._app_thread_id = app._thread_id
        self._pending_event = None
        self._pending_result = None
        self._rc: Console | None = None  # lazy init

    def _on_app_thread(self) -> bool:
        return threading.get_ident() == self._app_thread_id

    def _request(self, message: str) -> str:
        """Display *message* in the output log and wait for the user to type
        a response in the input area and press Enter.

        Must be called from a worker thread (not the Textual main thread).
        Widget access is dispatched to the main thread via ``call_from_thread``;
        the worker thread then blocks on ``_pending_event.wait()``.
        """
        import threading
        self._pending_event = threading.Event()
        self._pending_result = None

        # Dispatch UI setup to the main thread
        self._app.call_from_thread(self._request_ui, message)

        # Block worker thread until the user submits a response
        self._pending_event.wait()
        self._pending_event = None
        return self._pending_result or ""

    def _request_ui(self, message: str) -> None:
        """Set up the UI for a prompt request — runs on the Textual main thread."""
        from textual.widgets import Static, TextArea

        self._write(f"[bold]{message}[/bold]")
        self._write("")
        self._app.query_one("#status-bar", Static).update(
            " ↳ Type your response below and press Enter"
        )
        inp = self._app.query_one("#input", TextArea)
        inp.focus()

    def _submit(self, text: str) -> None:
        """Called by the TUI when the user submits input while a prompt is
        pending."""
        self._pending_result = text
        if self._pending_event:
            self._pending_event.set()


    def _get_rich_console(self) -> Console:
        """Return a themed Rich console set to the terminal width."""
        if self._rc is None:
            from io import StringIO
            from agent.console import monkee_theme
            w = term_width() - 4
            self._rc_buf = StringIO()
            self._rc = Console(
                theme=monkee_theme, file=self._rc_buf, width=w,
                force_terminal=True, color_system="truecolor",
            )
        return self._rc

    def _render_to_text(self, renderable) -> RichText:
        """Render a Rich renderable -> themed ANSI -> Rich Text."""
        rc = self._get_rich_console()
        self._rc_buf.truncate(0)
        self._rc_buf.seek(0)
        rc.print(renderable)
        return RichText.from_ansi(self._rc_buf.getvalue())

    def _write(self, text: str) -> None:
        """Append plain or markup text to the output log."""
        text = self._concretize(text)
        self._app.query_one("#output", RichLog).write(text)

    def _newline(self) -> None:
        self._app.query_one("#output", RichLog).write("\n")

    def _write_rich(self, renderable) -> None:
        """Append a Rich renderable to the output log."""
        self._app.query_one("#output", RichLog).write(renderable)

    # OutputAdapter interface

    def print(self, text: str = "", end='\n', indent: int = 0) -> None:
        """Print *text* to the RichLog output on the main thread."""
        txt = f"{' ' * indent}{text}"
        if self._on_app_thread():
            self._write(txt)
        else:
            self._app.call_from_thread(self._write, txt)

    def newline(self) -> None:
        """Print a blank line to the RichLog output on the main thread."""
        if self._on_app_thread():
            self._newline()
        else:
            self._app.call_from_thread(self._newline)

    def print_rich(self, renderable) -> None:
        """Print rich text"""
        txt = self._render_to_text(renderable)
        if self._on_app_thread():
            self._write_rich(txt)
        else:
            self._app.call_from_thread(self._write_rich, txt)

    def rule(self, style: str = "dim", title: str = "", align: AlignMethod = "center") -> None:
        rule = self._render_to_text(Rule(style=style, title=title, align=align))
        if self._on_app_thread():
            self._write_rich(rule)
        else:
            self._app.call_from_thread(self._write_rich, rule)

    def info(self, text: str, indent: int = 0) -> None:
        self.print(f"{' ' * indent}[bold deep_sky_blue3]⇨[/bold deep_sky_blue3] {text}")

    def err(self, text:str, indent: int = 0):
        self.print(f"{' ' * indent}[err]⨯[/err] {text}")

    def ok(self, text, indent: int = 0):
        self.print(f"{' ' * indent}[ok]✓[/ok] {text}")

    def ask_string(self, message: str, default: str = "") -> str:
        result = self._request(f"{message} [{default}]")
        return result if result else default

    def ask_float(self, message: str, default: float = 0.0) -> float:
        raw = self._request(f"{message} [{default}]")
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            self.print(f"[red]Invalid number, using default: {default}[/]")
            return default

    def ask_choice(
        self,
        message: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        # Display options via the main thread and set autocomplete
        suggestions = [name for (name, _) in options]

        def _display() -> None:
            lines = [f"[bold]{message}[/bold]"]
            for value, label in options:
                marker = "●" if value == default else "○"
                if value == label:
                    lines.append(f"  {marker} [accent]{value}[/accent]")
                else:
                    lines.append(f"  {marker} [accent]{value}[/accent] — {label}")
            self._write("\n".join(lines))
            self._write("\n")
            self._app.set_special_suggestions(suggestions)

        if self._on_app_thread():
            _display()
        else:
            self._app.call_from_thread(_display)

        raw = self._request(
            f"Enter your choice from the list above (default: [accent]{default}[/accent])"
        )

        # Reset special autocomplete on the main thread
        if self._on_app_thread():
            self._app.set_special_suggestions(None)
        else:
            self._app.call_from_thread(self._app.set_special_suggestions, None)

        if not raw and default is not None:
            return default
        # Allow typing either the value or the label
        values = {val: val for val, _ in options}
        labels = {label.lower(): val for val, label in options}
        if raw in values:
            return values[raw]
        if raw.lower() in labels:
            return labels[raw.lower()]
        if default is not None:
            self._app.call_from_thread(
                self._write, f"[red]Invalid choice, using default: {default}[/]"
            )
            return default
        return raw

    def ask_confirm(self, message: str, default: bool = False) -> bool:
        raw = self._request(f"{message} \\[{'Y/n' if default else 'y/N'}]")
        if not raw:
            return default
        return raw.lower() in ("y", "yes", "true", "1")

    def run_subprocess(self, cmd: list[str]):
        import subprocess

        def _do():
            with self._app.suspend():
                return subprocess.run(cmd)

        if self._on_app_thread():
            return _do()
        else:
            return self._app.call_from_thread(_do)
