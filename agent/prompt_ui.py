"""Concrete PromptUi implementations for classic REPL and TUI modes."""

from __future__ import annotations

from typing import Protocol

from rich.prompt import Prompt as RichPrompt, FloatPrompt, Confirm
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.formatted_text import HTML

from agent.console import console

# Prompt UI abstraction layer

class PromptUi(Protocol):
    """Abstract interface for user prompting.

    Implementations provide the same prompting primitives that ``rich.prompt``
    and ``prompt_toolkit`` offer, but routed through whichever UI is active
    (classic REPL or full-screen TUI).
    """

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

class RichPromptUi(PromptUi):
    """PromptUi backed by rich.prompt (for classic REPL mode)."""

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


class TuiPromptUi(PromptUi):
    """PromptUi backed by the Textual TUI (for --tui mode).

    Each method writes a prompt to the output log, switches the status bar
    to an input mode, and blocks until the user submits a response via the
    TextArea.  A threading.Event is used to wait for the response without
    blocking the Textual event loop.
    """

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
        self._pending_event = None
        self._pending_result = None

    # -- internal helpers ---------------------------------------------------

    def _request(self, message: str) -> str:
        """Display *message* in the output log and wait for the user to type
        a response in the input area and press Enter."""
        import threading
        self._pending_event = threading.Event()
        self._pending_result = None
        # Show the prompt in the output area
        self._app._write(f"[bold]{message}[/bold]")
        # Update status bar so the user knows we're waiting
        self._app._write("")
        from textual.widgets import Static
        self._app.query_one("#status-bar", Static).update(
            " ↳ Type your response below and press Enter"
        )
        # Focus the input
        from textual.widgets import TextArea
        inp = self._app.query_one("#input", TextArea)
        inp.focus()
        # Block until _on_input_submitted fires
        self._pending_event.wait()
        self._pending_event = None
        return self._pending_result or ""

    def _submit(self, text: str) -> None:
        """Called by the TUI when the user submits input while a prompt is
        pending."""
        self._pending_result = text
        if self._pending_event:
            self._pending_event.set()

    # -- PromptUi interface -------------------------------------------------

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
            self._app._write(f"[red]Invalid number, using default: {default}[/]")
            return default

    def ask_choice(
        self,
        message: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        # Display options in the output log
        lines = [f"[bold]{message}[/bold]"]
        for value, label in options:
            marker = "●" if value == default else "○"
            if value == label:
                lines.append(f"  {marker} [accent]{value}[/accent]")
            else:
                lines.append(f"  {marker} [accent]{value}[/accent] — {label}")
        self._app._write("\n".join(lines))
        self._app._write("\n")
        raw = self._request(f"Enter your choice from the list above (default: [accent]{default}[/accent])")
        if not raw and default is not None:
            return default
        # Allow typing either the value or the label
        values = {v: v for v, _ in options}
        labels = {l.lower(): v for v, l in options}
        if raw in values:
            return values[raw]
        if raw.lower() in labels:
            return labels[raw.lower()]
        if default is not None:
            self._app._write(f"[red]Invalid choice, using default: {default}[/]")
            return default
        return raw

    def ask_confirm(self, message: str, default: bool = False) -> bool:
        raw = self._request(f"{message} [{'Y/n' if default else 'y/N'}]")
        if not raw:
            return default
        return raw.lower() in ("y", "yes", "true", "1")

    def run_subprocess(self, cmd: list[str]):
        import subprocess

        def _do():
            with self._app.suspend():
                return subprocess.run(cmd)

        # We're called from the worker thread running the command;
        # call_from_thread hops to the main thread and blocks until done.
        return self._app.call_from_thread(_do)
