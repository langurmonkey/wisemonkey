"""Concrete OutputAdapter implementations for classic REPL mode."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Protocol

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import choice
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt
from rich.prompt import Prompt as RichPrompt

from agent.console import console
from agent.footer import Footer
from agent.keys import ensure_readline, legacy_key_encoding, readline_prompt
from agent.steer import SteerInput

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


def get_output_or_ipc() -> OutputAdapter:
    """Return the active output adapter, falling back to a loopback IPC one.

    Code running outside a configured frontend (scripts, headless turns,
    background threads) gets an :class:`IpcOutputAdapter` wired to a private
    loopback transport instead of crashing or scribbling on a shared console.
    The adapter drains its own events, so writes are effectively no-ops
    visually but remain observable (e.g. via ``drain()`` in tests or tools).
    """
    if _active_output:
        return _active_output
    return IpcOutputAdapter()


def incomplete_message(reason: str = "") -> str:
    """The one wording for a fragmented answer, shared by every frontend.

    ``reason`` is ``"length"`` when the provider stopped at the output-token
    limit and ``"stream_ended"`` (or empty) when the stream ended with no
    terminal marker at all -- a connection cut at a chunk boundary, which
    raises nothing and so used to be shown as a finished answer. The two
    cases need different words: one is a budget, the other a fault.
    """
    if reason == "length":
        return (
            "⚠  Response incomplete: the model hit the output token limit. "
            "The answer is cut off; ask it to continue."
        )
    return (
        "⚠  Response incomplete: the stream ended without a stop marker "
        "(the connection was cut). The stored answer is a fragment; re-run "
        "or ask it to continue."
    )


#: Tool-result kinds that did not come from the tool itself. Two shapes reach
#: `tool_result_callback` this way, and both need the frame below: an external
#: hook *denying* a call, and the user *declining* after a hook asked for
#: review. Everything else is an ordinary tool result.
HOOK_BLOCK_KIND = "hook_block"
HOOK_DECLINED_KIND = "hook_ask_declined"


def hook_notice(kind: str, hook: str, tool_name: str, reason: str):
    """A framed notice for a tool call an external hook stopped.

    The frame is the point. The text inside is the *guard's*, and on a bare
    `✗ Tool run_command failed: ...` line it read exactly like the agent
    reporting a failure of its own -- so a refused call looked like the model
    speaking, which is the confusion this exists to prevent. The title and the
    first line name who spoke.

    The body is built as ``Text``, not markup: a guard's reason is arbitrary
    text and its own brackets or colour codes must not be reinterpreted as
    Rich markup (an unknown tag is dropped silently, mangling the message).
    """
    from rich.panel import Panel
    from rich.text import Text

    name = hook or "a hook"
    if kind == HOOK_DECLINED_KIND:
        title = f"✋ Declined after {name} asked for review"
        lead = f"{tool_name} was not run: you declined."
        style = "warn"
    else:
        title = f"⛔ Blocked by {name}"
        lead = f"{tool_name} was not run: {name} refused it."
        style = "error"

    body = Text()
    body.append(lead + "\n", style="weak")
    body.append(f"The text below is {name}'s own output, not the model's.\n\n", style="weak")
    body.append(reason.strip() or "(no reason given)")
    return Panel(
        body,
        title=Text(title, style=style),
        title_align="left",
        border_style=style,
    )


# ── readline-aware Rich prompts ──────────────────────────────────────────────
#
# Rich prints the prompt itself and then calls `input()` with an *empty*
# prompt. readline therefore believes the line buffer begins at column 1, so
# any redraw — Backspace above all, and equally an insert in the middle of the
# line or any character typed after a Left — rewrites from column 1 and erases
# the question that was already on screen. The symptom is precise: press
# Backspace once and the whole question vanishes, leaving only the typed text.
#
# `readline_prompt()` hands readline the prompt so it knows where the buffer
# starts. These three subclasses are the only difference from the Rich
# originals; `ask_choice` uses prompt_toolkit, which renders and reads the
# line itself and never had the problem.


class _ReadlinePromptMixin:
    """Give a Rich prompt class a line editor that knows the prompt exists."""

    @classmethod
    def get_input(cls, console, prompt, password, stream=None) -> str:
        if password or stream is not None:
            # Password input goes through getpass, and an explicit stream means
            # the caller is reading lines itself; neither wants readline.
            return RichPrompt.get_input(console, prompt, password, stream)
        return input(readline_prompt(console, prompt))


class _ReadlinePrompt(_ReadlinePromptMixin, RichPrompt):
    pass


class _ReadlineFloatPrompt(_ReadlinePromptMixin, FloatPrompt):
    pass


class _ReadlineConfirm(_ReadlinePromptMixin, Confirm):
    pass

class OutputAdapter(Protocol):
    """Abstract interface for output adapters.

    Implementations provide the same prompting primitives that ``rich.prompt``
    and ``prompt_toolkit`` offer, but routed through whichever UI is active.
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

    # ── sticky footer + mid-turn steering ─────────────────────────────────────
    # Frontends that own a plain terminal (the classic REPL) implement these;
    # full-screen frontends inherit these no-ops.

    def footer_clear_prompt(self) -> None:
        """Scroll the prompt box clear of the footer zone before arming."""

    def footer_start(self) -> None:
        """Reserve the bottom terminal rows for a sticky status line."""

    def footer_stop(self) -> None:
        """Release the reserved rows."""

    def footer_update(self, status_line: str, input_line: str = "") -> None:
        """Update the sticky status line and the input line above the hints."""

    def footer_active(self) -> bool:
        """Whether a sticky footer is currently reserving terminal rows."""
        return False

    def footer_spinner(self, text: str) -> None:
        """Show an animated spinner with label *text* on the footer.

        The footer animates it itself (see :meth:`agent.footer.Footer.set_spinner`);
        a no-op where there is no sticky footer.
        """
        return

    def footer_thinking(self, active: bool) -> None:
        """Show or hide the model's reasoning indicator on the footer.

        No-op where the footer has no reasoning indicator.
        """
        return

    def steer_start(
        self,
        on_submit: Callable[[str], None],
        on_change: Callable[[], None] | None = None,
    ) -> bool:
        """Capture mid-turn keystrokes. Returns True when steering works."""
        return False

    def steer_stop(self) -> None:
        """Stop capturing mid-turn keystrokes."""

    def steer_take(self) -> "str | None":
        """Pop the oldest queued steering line, if any."""
        return None

    def steer_ack(self, text: str) -> bool:
        """Drop the queued line *text* once the server confirms delivery."""
        return False

    def steer_take_all(self) -> "list[str]":
        """Pop and return every queued steering line, oldest first."""
        return []

    def steer_line(self) -> str:
        """Return the steering line currently being typed."""
        return ""

    def steer_caret(self) -> int:
        """Return the caret's index within :meth:`steer_line`.

        Separate from the line because the frontend draws a block cursor
        *inside* the text rather than always at its end: an arrow key moves
        the caret, and the row has to follow it.
        """
        return 0

    def steer_active(self) -> bool:
        """Whether mid-turn keystrokes are being captured *right now*.

        False where steering is unavailable (no tty, disabled by config) and
        while the reader is paused -- an ``ask_*`` question or a subprocess
        owning the terminal. That is what lets a frontend tell a row that
        accepts typing from one that would swallow it.
        """
        return False

    def steer_pending(self) -> "str | None":
        """Return the oldest queued steering line without consuming it."""
        return None

    @contextmanager
    def steer_paused(self):
        """Temporarily hand the tty back to the rest of the REPL."""
        yield


class RichOutputAdapter(OutputAdapter):
    """OutputAdapter backed by rich.prompt (for classic REPL mode).

    Uses its own ``Console`` instance (with the ``monkee_theme``) so that
    styled output is **not** affected by the ``no_color = True`` / file
    redirection that ``Core._tool_calls`` applies to the shared
    ``agent_console.console`` while capturing tool results.
    """

    def __init__(self) -> None:
        # `agent.footer_debug_bytes`: append every byte written to the
        # footer to $XDG_STATE_HOME/wisemonkey/footer.log, for diagnosing
        # terminal rendering problems.
        from agent.config import Config
        from agent.console import monkee_theme

        debug = bool(Config().get("agent.footer_debug_bytes", False))
        self._footer = Footer(debug_bytes=debug)
        self._console = Console(theme=monkee_theme, color_system="truecolor")
        self._steer: SteerInput | None = None
        self._steer_on_change: "Callable[[], None] | None" = None
        self._footer_status_cache: tuple[str, str] = ("", "")

    # ── sticky footer (active only during an assistant turn) ────────────

    def footer_clear_prompt(self) -> None:
        """Scroll the prompt box above the footer zone before arming it."""
        self._footer.clear_prompt_zone()

    def footer_start(self) -> None:
        """Reserve the bottom terminal rows for the sticky status line."""
        self._footer.start()

    def footer_stop(self) -> None:
        """Release the reserved rows and restore the normal scroll region."""
        self._footer.stop()

    def footer_update(self, status_line: str, input_line: str = "") -> None:
        """Update the sticky status line (no-op when the footer is off)."""
        self._footer_status_cache = (status_line, input_line)
        self._footer.update_status(status_line, input_line)

    def footer_active(self) -> bool:
        """Whether the sticky footer is currently armed."""
        return self._footer.active

    def footer_spinner(self, text: str) -> None:
        """Start/stop the footer's own spinner animation."""
        self._footer.set_spinner(text)

    def footer_thinking(self, active: bool) -> None:
        """Show/hide the footer's reasoning indicator (key-hints row)."""
        self._footer.set_thinking(active)

    # ── mid-turn steering (classic REPL only) ─────────────────────────────────

    def steer_start(
        self,
        on_submit: "Callable[[str], None]",
        on_change: "Callable[[], None] | None" = None,
    ) -> bool:
        """Capture mid-turn keystrokes so the user can steer a running turn.

        The tty is switched to cbreak mode; submitted lines are handed to
        *on_submit* from a reader thread. Returns False when there is no
        interactive terminal, in which case steering is simply unavailable.
        """
        self.steer_stop()
        self._steer = SteerInput(on_submit, on_change)
        # Kept separately so `steer_paused` can redraw the row when it re-arms
        # (see below); the reader thread is gone by then.
        self._steer_on_change = on_change
        return self._steer.start()

    def steer_stop(self) -> None:
        """Stop capturing keystrokes and restore the tty.

        Queued lines are kept so the caller can carry them into the prompt
        once the turn is over (:meth:`steer_start` is what discards them).
        """
        if self._steer is not None:
            self._steer.stop()

    def steer_take(self) -> "str | None":
        """Pop the oldest queued steering line, if any."""
        return self._steer.take() if self._steer else None

    def steer_ack(self, text: str) -> bool:
        """Drop the queued line *text* once the server confirms delivery.

        Returns True if the line was still queued (the normal case).
        """
        return self._steer.ack(text) if self._steer else False

    def steer_take_all(self) -> "list[str]":
        """Pop and return every queued steering line, oldest first."""
        return self._steer.take_all() if self._steer else []

    def steer_line(self) -> str:
        """Return the steering line currently being typed."""
        return self._steer.line() if self._steer else ""

    def steer_caret(self) -> int:
        """Return the caret's index within :meth:`steer_line`."""
        return self._steer.caret() if self._steer else 0

    def steer_active(self) -> bool:
        """Whether the reader thread is capturing keystrokes right now."""
        return bool(self._steer is not None and self._steer.armed)

    def steer_pending(self) -> "str | None":
        """Return the oldest queued steering line without consuming it."""
        return self._steer.pending() if self._steer else None

    @contextmanager
    def steer_paused(self):
        """Temporarily hand the tty back (agent questions, subprocesses).

        The reader thread is *suspended* so the cooked tty the ask_*/
        run_subprocess helpers expect works. Nothing the user has typed is
        discarded, and that is the point: a tool asking for confirmation --
        or an editor, or a subprocess -- used to clear the half-written
        steering line, which then reappeared empty once the tool had run.
        Enter submits and Ctrl+U clears; a question asked mid-turn does
        neither, so the line, its caret and the queued follow-ups all come
        back on :meth:`resume`.

        It is re-armed **only if it was armed when we entered**. Between turns
        the agent calls :meth:`steer_stop`, so at the moment the user runs
        ``/config edit`` or ``/edit`` the reader is already disarmed and the
        prompt owns the tty. Blindly re-arming it here would leave a second
        reader thread on the same fd: prompt_toolkit and ``SteerInput`` would
        race for every byte, the tty would be left in cbreak mode after the
        editor exited, and roughly one keystroke in ten would reach the
        prompt -- the other nine going into the discarded steering buffer.
        """
        steer = self._steer
        if steer is None:
            yield
            return
        was_armed = steer.armed
        steer.suspend()
        try:
            yield
        finally:
            if was_armed:
                steer.resume()
                # The row drops its caret while the reader is suspended (it
                # would be a lie -- keystrokes were going to the question).
                # Redraw so it reappears now, rather than at the next
                # streamed chunk.
                on_change = getattr(self, "_steer_on_change", None)
                if on_change is not None:
                    on_change()

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
        self._console.rule(title=title, style=style)
# A turn may ask the user something (confirmation, a question) or hand the
    # terminal to a subprocess. Both need a normal tty, so mid-turn steering
    # is suspended for the duration.
    #
    # Three things are needed for these three prompts:
    #
    # 1. `ensure_readline()`. `rich.prompt` reads through the builtin
    #    `input()`, and CPython only gives that a line editor when the
    #    `readline` module has been imported -- it does not import it for a
    #    script. Without it the read is bare, the kernel echoes every byte it
    #    gets, and pressing Right types `^[[C` into the line instead of moving
    #    the cursor.
    # 2. `legacy_key_encoding`. The disambiguate flag re-encodes every
    #    functional key, so a Left arrow arrives as `CSI 1;129D`, which
    #    readline does not know.
    # 3. `_ReadlinePrompt`. Rich prints the question and then calls `input()`
    #    with an *empty* prompt, so readline believes the line buffer starts at
    #    column 1. Every redraw -- Backspace above all -- then rewrites from
    #    column 1 and erases the question, leaving only the typed text. The
    #    subclasses below hand readline the prompt instead; see
    #    `agent.keys.readline_prompt`.
    #
    # `ask_choice` is prompt_toolkit: it renders and reads the line itself, so
    # it needs none of the three and keeps Shift+Enter.
    def ask_string(self, message: str, default: str = "") -> str:
        ensure_readline()
        with self.steer_paused(), legacy_key_encoding():
            return _ReadlinePrompt.ask(message, default=default, console=console)

    def ask_float(self, message: str, default: float = 0.0) -> float:
        ensure_readline()
        with self.steer_paused(), legacy_key_encoding():
            return _ReadlineFloatPrompt.ask(
                message, default=default, console=console
            )

    def ask_choice(
        self,
        message: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        with self.steer_paused():
            return choice(
                message=message,
                options=options,
                default=default,
                bottom_toolbar=HTML(
                    " <b>↑</b>/<b>↓</b>: select | <b>Enter</b>: accept"
                ),
            )

    def ask_confirm(self, message: str, default: bool = False) -> bool:
        ensure_readline()
        with self.steer_paused(), legacy_key_encoding():
            return _ReadlineConfirm.ask(message, default=default, console=console)

    def run_subprocess(self, cmd: list[str]):
        """Run a program with full terminal control.

        Steering, the scroll region and the keyboard protocol all interfere
        with a child process, so they are torn down around the run and restored
        afterwards. The protocol matters for `$EDITOR`: the flag re-encodes
        every functional key, and a program that never asked for it -- which is
        every program that does not implement the kitty protocol -- reads the
        result as escape characters typed into its input.
        """
        import subprocess

        with self.steer_paused(), legacy_key_encoding():
            active = self._footer.active
            if active:
                self._footer.stop()
            try:
                return subprocess.run(cmd)
            finally:
                if active:
                    self._footer.start()
                    self._footer.update_status(*self._footer_status_cache)


class IpcOutputAdapter(OutputAdapter):
    """OutputAdapter that routes UI interactions through IPC payloads.

    Print-style methods are encoded as :class:`agent.ipc.OutputPayload`
    events on a transport (a loopback pair by default, later a real socket).
    Interactive requests (``ask_*``) are emitted as server requests and, in
    non-interactive contexts, return safe defaults immediately.
    """

    def __init__(self, transport=None, ask_handler=None) -> None:
        """Create the adapter.

        Parameters:
            transport: Transport  - Where OUTPUT events are sent. Defaults to
                                    a private loopback pair.
            ask_handler: callable - Optional ``(name, payload) -> value`` used
                                    to answer interactive ``ask_*`` requests
                                    (e.g. the server's ``ask_client`` RPC
                                    relay). When None, requests are answered
                                    with safe non-interactive defaults.
        """
        from agent.ipc import loopback_pair

        if transport is None:
            transport, _peer = loopback_pair()
        self.transport = transport
        self.ask_handler = ask_handler
        self._interactive = ask_handler is not None
    def _send(self, payload) -> None:
        from agent.ipc import Event, Message

        try:
            self.transport.send(Message.event(Event.OUTPUT, payload))
        except Exception:
            pass

    def _output(self, fmt: str, text: str, level: str = "normal", indent: int = 0, **extra) -> None:
        from agent.ipc import OutputPayload

        self._send(OutputPayload(format=fmt, text=text, level=level, indent=indent, **extra))

    # ── print-style ─────────────────────────────────────────────────────────
    def print(self, text: str = "", end: str = "\n", indent: int = 0) -> None:
        self._output("text", text, end=end, indent=indent)

    def print_rich(self, renderable) -> None:
        self._output("text", str(renderable))

    def newline(self) -> None:
        self._output("text", "")

    def rule(self, style: str = "dim", title: str = "") -> None:
        self._output("rule", "", style=style, title=title)

    def info(self, text: str, indent: int = 0) -> None:
        self._output("text", text, level="info", indent=indent)

    def err(self, text: str, indent: int = 0) -> None:
        self._output("text", text, level="err", indent=indent)

    def ok(self, text: str, indent: int = 0) -> None:
        self._output("text", text, level="ok", indent=indent)

    # ── interactive (non-interactive defaults in phase 1) ──────────────────
    def _deny(self, message: str) -> bool:
        self._output("text", f"[denied] {message}", level="warn")
        return False

    def _ask(self, name, payload):
        """Route an interactive request to the handler, if configured."""
        from agent.ipc import ServerRequest

        if self.ask_handler is None:
            return None
        return self.ask_handler(ServerRequest(name), payload)

    def ask_string(self, message: str, default: str = "") -> str:
        from agent.ipc import AskStringPayload, ServerRequest

        value = self._ask(ServerRequest.ASK_STRING, AskStringPayload(message=message, default=default))
        if value is not None:
            return str(value)
        self._output("text", f"[ask] {message}", level="warn")
        return default

    def ask_float(self, message: str, default: float = 0.0) -> float:
        from agent.ipc import AskFloatPayload, ServerRequest

        value = self._ask(ServerRequest.ASK_FLOAT, AskFloatPayload(message=message, default=default))
        if value is not None:
            return float(value)
        self._output("text", f"[ask] {message}", level="warn")
        return default

    def ask_choice(
        self,
        message: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        from agent.ipc import AskChoicePayload, ServerRequest

        payload = AskChoicePayload(
            message=message,
            options=[[v, l] for v, l in options],
            default=default or "",
        )
        value = self._ask(ServerRequest.ASK_CHOICE, payload)
        if value is not None:
            return str(value)
        self._output("text", f"[ask] {message}", level="warn")
        return options[0][0] if options else ""

    def ask_confirm(self, message: str, default: bool = False) -> bool:
        from agent.ipc import ConfirmPayload, ServerRequest

        value = self._ask(ServerRequest.CONFIRM, ConfirmPayload(message=message, default=default))
        if value is not None:
            return bool(value)
        self._output("text", f"[ask] {message}", level="warn")
        return default

    def footer_clear_prompt(self) -> None:
        """No-op: no terminal is owned in an IPC context."""

    def footer_start(self) -> None:
        """No-op: no terminal is owned in an IPC context."""

    def footer_stop(self) -> None:
        """No-op: nothing was reserved."""

    def footer_update(self, status_line: str, input_line: str = "") -> None:
        """No-op: there is no footer to update."""

    def steer_start(
        self,
        on_submit: Callable[[str], None],
        on_change: Callable[[], None] | None = None,
    ) -> bool:
        """Mid-turn steering needs a terminal; unavailable over IPC."""
        return False

    def steer_stop(self) -> None:
        """No-op: nothing was captured."""

    def run_subprocess(self, cmd: list[str]):
        from agent.ipc import RunSubprocessPayload, ServerRequest

        value = self._ask(
            ServerRequest.RUN_SUBPROCESS,
            RunSubprocessPayload(cmd=list(cmd), cwd=""),
        )
        if value is not None:
            return value
        self._output("text", f"[subprocess denied in non-interactive context] {' '.join(cmd)}", level="err")
        raise RuntimeError("run_subprocess is not available in non-interactive IPC context")

    # ── observability ───────────────────────────────────────────────────────
    def drain(self, timeout: float = 0.0):
        """Consume pending events (useful for tests and recording)."""
        out = []
        while True:
            try:
                msg = self.transport.recv(timeout=timeout)
            except Exception:
                break
            if msg is None:
                break
            out.append(msg)
        return out
