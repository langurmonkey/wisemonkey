"""
Textual TUI for Wisemonkey.

Provides a full-screen terminal UI with:
  - Title bar (Header) at the top
  - Scrollable output area (RichLog) in the middle
  - Status bar above the input
  - Text input at the bottom

Coexists with the terminal-based agent (agent/agent.py).
Launch with: wisemonkey --tui <session>
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text as RichText

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import TextArea, Header, RichLog, Static, Footer
from textual import work
from textual.binding import Binding
from textual.timer import Timer

from agent.core import Core, Stage, TurnCancelled
from agent.commands import registry
from agent.history import History
from agent.prompt_ui import TuiPromptUi as _TuiPromptUi
from agent.startup import startup_info, StartupOutput
from agent.console import theme_dict
from agent.utils import term_width
from textual.events import Paste as PasteEvent

# Number of characters above which the paste action creates a file
PASTE_THRESHOLD = 20

# ---------------------------------------------------------------------------
# A TextArea that handles history on up/down when cursor is at boundaries
# ---------------------------------------------------------------------------

class _SubmitTextArea(TextArea):
    """TextArea that handles history on up/down, Ctrl+C (clear/double-tap
    quit), and paste threshold."""

    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("shift+enter", "newline", "New line"),
        Binding("ctrl+c", "clear", "Clear"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_ctrl_c_time: float = 0.0

    @property
    def _wm_app(self) -> WisemonkeyTui:
        """Narrow ``self.app`` to ``WisemonkeyTui`` for type-checking."""
        from typing import cast as _cast
        return _cast(WisemonkeyTui, self.app)

    def action_submit(self) -> None:
        """Forward submit to the parent app."""
        self._wm_app.action_submit_text()

    def action_newline(self) -> None:
        """Insert a newline."""
        self._wm_app.action_newline()

    def action_cursor_up(self, select: bool = False) -> None:
        """Override: navigate history when at very start of first line."""
        row, col = self.cursor_location
        if row == 0 and col == 0:
            self._wm_app.action_history_up()
        else:
            super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        """Override: navigate history when at very end of last line."""
        row, col = self.cursor_location
        lines = self.text.split("\n")
        if row == len(lines) - 1 and col == len(lines[-1]):
            self._wm_app.action_history_down()
        else:
            super().action_cursor_down(select)

    def action_clear(self) -> None:
        """Clear text"""
        self.clear()

    async def _on_paste(self, event: PasteEvent) -> None:
        """Handle paste: save large pastes (>PASTE_THRESHOLD chars) to a file
        instead of inserting them into the buffer."""
        if self.read_only:
            return

        event.stop()
        event.prevent_default()

        text = event.text
        start, end = self.selection

        if len(text) <= PASTE_THRESHOLD:
            # Small paste: insert normally at the current selection.
            self.replace(text, start, end, maintain_selection_offset=False)
            return

        core = self._wm_app.core
        if not core:
            return

        file_path = core.memory.create_pasted_file(text)

        # If the cursor isn't already at the start of a line, lead with a
        # newline so the reference sits on its own line.
        _, col = start
        prefix = "\n" if col != 0 else ""
        reference = f"{prefix}*Pasted file: {file_path}*\n"

        # Insert ONLY the reference, at exactly the spot the pasted text
        # would have gone — never touch self.text wholesale.
        self.replace(reference, start, end, maintain_selection_offset=False)

    def key_control_c(self) -> None:
        """Ctrl+C: first press clears input, second press (within 1s) quits."""
        import time
        now = time.time()
        if self.text:
            self.text = ""
            self._last_ctrl_c_time = now
        else:
            if now - self._last_ctrl_c_time < 1.0:
                self.app.exit()
            else:
                self._last_ctrl_c_time = now


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


def _concretize(text: str) -> str:
    """Replace abstract style tags with concrete colour tags."""
    for k in _STYLE_KEYS:
        text = text.replace(k, _STYLE_MAP[k])
    return text

class TuiStartupOutput(StartupOutput):
    """Startup output adapter that writes directly to the TUI RichLog.

    Abstract style names (``[title]``, ``[accent]``) in markup strings are
    replaced with concrete colour tags via ``_concretize()``.

    Rich renderables (Panels, Rules, Align) are rendered through a private
    ``rich.Console`` with the ``monkee_theme`` theme, capturing the ANSI
    output.  The result is converted to a ``rich.text.Text`` with fully
    resolved concrete styles and correct width.
    """

    def __init__(self, app: WisemonkeyTui):
        self.app = app
        self._rc: Console | None = None  # lazy init

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

    def print(self, text: str) -> None:
        self.app._write(_concretize(text))

    def print_rich(self, renderable) -> None:
        self.app._write_rich(self._render_to_text(renderable))

    def newline(self) -> None:
        self.app._write("")

    def rule(self, style: str = "dim", title: str = "") -> None:
        self.app._write_rich(self._render_to_text(Rule(style=style, title=title)))

    def info(self, text: str) -> None:
        self.app._write(
            _concretize(f"[bold deep_sky_blue3]\u21e8[/bold deep_sky_blue3] {text}")
        )


class WisemonkeyTui(App):
    """Full-screen Textual TUI for Wisemonkey."""

    TITLE = "Wisemonkey"
    CSS_PATH = "../styles/wm.tcss"

    def __init__(self, config_path: str | None = None, session: str = "default"):
        super().__init__()
        self.config_path = config_path
        self.session = session
        self.core: Core | None = None
        self.output: TuiStartupOutput = TuiStartupOutput(self)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="output", highlight=True, markup=True, wrap=True)
        with Container(id="bottom-area"):
            yield Static(id="status-bar")
            with Container(id="input-widget"):
                yield _SubmitTextArea.code_editor(id="input",
                                                  placeholder="Type a message...",
                                                  soft_wrap=True,
                                                  language="markdown")
            yield Footer(id="bottom-bar")

    def on_mount(self) -> None:
        """Set up the agent core and render startup info."""
        self.sub_title = f"Session: {self.session}"

        try:
            self.core = Core(self.config_path, self.session)
        except Exception as e:
            self._write(f"[red]Agent initialisation failed: {e}[/]")
            return

        # Render startup info via the shared module
        startup_info(self.core, self.output)
        self._update_status()

        # Stream buffers: accumulate content until a newline is hit
        self._stream_buffer = ""
        self._reasoning_buffer = ""

        # Spinner interval handles
        self._prompt_spinner_interval: Timer | None = None
        self._thinking_spinner_interval: Timer | None = None

        # Prompt UI for interactive commands
        self._prompt_ui = _TuiPromptUi(self)

        # Input history
        sess_dir = str(self.core.memory.session_dir)
        self._history = History(sess_dir)

        # Focus the input field by default
        inp = self.query_one("#input", _SubmitTextArea)
        inp.focus()

    # ---- reasoning callback -------------------------------------------------

    def _reasoning_callback(self, stage: Stage, content: str = "", reasoning_visible: bool = True) -> None:
        """Called from worker thread on START / PROCESS / STOP of reasoning.

        START: show "Thinking..." in the status bar with a spinner.
        PROCESS: accumulate reasoning in a buffer, flush on newline.
        STOP: stop spinner, flush remaining buffer, restore status.

        All UI writes go through ``call_from_thread``, same pattern as
        ``_append_content``.
        """
        if stage == Stage.START:
            self.call_from_thread(self.query_one("#status-bar", Static).update, " \U0001f4a1 Thinking...")
            self._thinking_spinner_chars = "\u25d0\u25d3\u25d1\u25d2"
            self._thinking_spinner_idx = 0

            def _tick() -> None:
                sb = self.query_one("#status-bar", Static)
                sb.update(
                    f" {self._thinking_spinner_chars[self._thinking_spinner_idx % 4]} Thinking..."
                )
                self._thinking_spinner_idx += 1

            self._thinking_spinner_interval = self.set_interval(0.2, _tick)

        elif stage == Stage.PROCESS:
            if content and reasoning_visible:
                self._reasoning_buffer += content
                if "\n" in content:
                    buf = self._reasoning_buffer
                    self._reasoning_buffer = ""
                    self.call_from_thread(self._write, f"[dim]{buf}[/dim]")

        elif stage == Stage.STOP:
            if self._thinking_spinner_interval:
                self._thinking_spinner_interval.stop()
                self._thinking_spinner_interval = None
            # Flush any remaining reasoning content
            if self._reasoning_buffer:
                self.call_from_thread(self._write, f"[dim]{self._reasoning_buffer}[/dim]")
                self._reasoning_buffer = ""
            self.call_from_thread(self._write, "[green]\u2714[/green] \U0001f4a1 Done thinking")
            self.call_from_thread(self._update_status)

    # ---- prompt callback (spinner) -----------------------------------------

    def _prompt_callback(self, stage: Stage) -> None:
        """Called when the LLM starts/stops processing a prompt.

        On START: replace the status bar with "⏳ Processing prompt..."
        and start a spinner animation via set_interval.
        On STOP: stop the spinner and restore the normal status.
        """
        if stage == Stage.START:
            self.query_one("#status-bar", Static).update(" ⏳ Processing prompt...")
            self._prompt_spinner_chars = "◐◓◑◒"
            self._prompt_spinner_idx = 0

            def _tick() -> None:
                sb = self.query_one("#status-bar", Static)
                sb.update(
                    f" {self._prompt_spinner_chars[self._prompt_spinner_idx % 4]} Processing prompt..."
                )
                self._prompt_spinner_idx += 1

            self._prompt_spinner_interval = self.set_interval(0.2, _tick)

        elif stage == Stage.STOP:
            if self._prompt_spinner_interval:
                self._prompt_spinner_interval.stop()
                self._prompt_spinner_interval = None
            self._update_status()

    # ---- output helpers ----------------------------------------------------

    def _write(self, text: str) -> None:
        """Append plain or markup text to the output log."""
        text = _concretize(text)
        self.query_one("#output", RichLog).write(text)

    def _write_rich(self, renderable) -> None:
        """Append a Rich renderable to the output log."""
        self.query_one("#output", RichLog).write(renderable)

    def _update_status(self) -> None:
        """Refresh the status and bottom bars text."""
        if not self.core:
            self.query_one("#status-bar", Static).update(" Not connected")
            return
        model = self.core.config.get("model.name", "?")
        sess = self.core.memory.session
        self.query_one("#status-bar", Static).update(
            f" Model: [bold]{model}[/bold]  |  Session: [bold]{sess}[/bold]"
        )

    # ---- input handling ----------------------------------------------------

    def _handle_command(self, user_input: str) -> None:
        """Execute a slash command and print its result.

        Runs in a background thread so that TuiPromptUi can block waiting
        for user input without freezing the Textual event loop.
        """
        self.output.rule(style="dim")
        tokens = user_input.split()
        command, params = registry.lookup(tokens)

        if not command:
            self._write(f"[red]Command not found: {user_input}[/red]")
            return

        self._run_command_in_thread(command, params)

    @work(thread=True, exit_on_error=False)
    def _run_command_in_thread(self, command, params) -> None:
        """Run a slash command on a worker thread."""
        ok_flag, msg, content, md, should_exit = registry.execute(
            self.core, command, params, self._prompt_ui
        )

        if should_exit:
            self.call_from_thread(self._write, "[bold]Goodbye![/bold]")
            self.call_from_thread(self.exit)
            return

        if ok_flag:
            if content:
                self.call_from_thread(self._write, content)
            elif md:
                self.call_from_thread(self._write_rich, Markdown(md))
            if msg:
                self.call_from_thread(self._write, f"[green]✔[/green] {msg}")
        else:
            if msg:
                self.call_from_thread(self._write, f"[red]✗[/red] {msg}")

        self.call_from_thread(self._update_status)

    def _handle_prompt(self, user_input: str) -> None:
        """Send the user message to the LLM in a background thread."""
        self.output.rule(style="dim")
        self._write("[medium_orchid bold]Wisemonkey:[/medium_orchid bold]")
        self._stream_buffer = ""
        self._run_turn(user_input)

    # ---- turn execution (threaded) -----------------------------------------

    BINDINGS = [
        # Enter / shift+enter are handled by _SubmitTextArea
        # Up / down for history are handled by _SubmitTextArea (cursor-boundary logic)
    ]

    def action_newline(self) -> None:
        """Insert a newline in the text area."""
        inp = self.query_one("#input", _SubmitTextArea)
        row, col = inp.cursor_location
        text = inp.text
        lines = text.split("\n")
        line = lines[row]
        # Split the current line at cursor and join with newline
        lines[row] = line[:col]
        lines.insert(row + 1, line[col:])
        inp.text = "\n".join(lines)
        inp.cursor_location = (row + 1, 0)

    def action_submit_text(self) -> None:
        """Submit the current text in the input area."""
        inp = self.query_one("#input", _SubmitTextArea)
        text = inp.text.strip()
        # If we are waiting for an event, we can input nothing to use default
        if self._prompt_ui._pending_event is None and not text:
            return
        inp.text = ""
        self._history.add(text)
        # If a prompt_ui request is pending, fulfill it instead of sending
        # the text as a chat message.
        if self._prompt_ui._pending_event is not None:
            self._prompt_ui._submit(text)
        else:
            self._handle_user_input(text)

    def action_history_up(self) -> None:
        """Move up in history - only if called from _SubmitTextArea (cursor at start)."""
        entry = self._history.up()
        if entry is not None:
            inp = self.query_one("#input", _SubmitTextArea)
            inp.text = entry
            inp.cursor_location = (0, 0)

    def action_history_down(self) -> None:
        """Move down in history - only if called from _SubmitTextArea (cursor at end)."""
        entry = self._history.down()
        inp = self.query_one("#input", _SubmitTextArea)
        if entry is not None:
            inp.text = entry
            lines = entry.split("\n")
            inp.cursor_location = (len(lines) - 1, len(lines[-1]))
        else:
            inp.text = ""


    def _handle_user_input(self, user_input: str) -> None:
        """Process a user message or slash command."""
        self._write(f"\n[gold1 bold]You:[/gold1 bold] {user_input}")

        if user_input.startswith("/"):
            self._handle_command(user_input)
        else:
            self._handle_prompt(user_input)

    @work(thread=True, exit_on_error=False)
    def _run_turn(self, user_input: str) -> None:
        """Run a full turn on a worker thread so the UI stays responsive."""
        if not self.core:
            self.call_from_thread(self._write, "[red]Core not initialised[/]")
            return

        self.call_from_thread(
            self.query_one("#status-bar", Static).update, " Processing\u2026"
        )

        try:
            (response, total_tokens, ntools, total_gen_time) = self.core.run_turn(
                user_input,
                prompt_callback=lambda s: self.call_from_thread(self._prompt_callback, s),
                reasoning_callback=self._reasoning_callback,
                content_callback=self._append_content,
                tool_callback=self._append_tool,
                cancel_callback=None,
                error_callback=None,
            )
            self.call_from_thread(
                self._finish_turn, response, total_tokens, ntools, total_gen_time
            )
        except TurnCancelled:
            self.call_from_thread(self._write, "[orange1]Turn cancelled[/]")
        except Exception as e:
            self.call_from_thread(self._write, f"[red]Error: {e}[/]")
        finally:
            self.call_from_thread(self._update_status)

    def _append_content(self, content: str) -> None:
        """Streaming callback – called from worker thread.

        Accumulates tokens in a buffer and only flushes to the RichLog
        when a newline is encountered.  This prevents every token from
        becoming its own RichLog entry (which causes double-spacing).
        """
        self._stream_buffer += content
        # Flush only on newlines to avoid fragmenting the output
        if "\n" in content:
            buf = self._stream_buffer
            self._stream_buffer = ""
            self.call_from_thread(self._write, buf)

    def _append_tool(self, tool_name: str, tool_args) -> None:
        """Tool activation callback – called from worker thread."""
        self.call_from_thread(
            self._write,
            f"[dim]🛠️ Activating tool: [steel_blue3]{tool_name}[/steel_blue3][/dim]",
        )

    def _finish_turn(
        self, response: str, tokens: int, ntools: int, gen_time: float
    ) -> None:
        """Called on the main thread after a turn completes."""
        # Flush any remaining buffered content
        if self._stream_buffer:
            self._write(self._stream_buffer)
            self._stream_buffer = ""

        if response == "[Cancelled]":
            return

        if self.core:
            length, max_sz, rate = self.core.memory.get_chat_stats()
            label = f"  {gen_time:.1f}s  |  {tokens} tokens  |  {ntools} tools  |  Mem: {length}/{max_sz} ({rate:.2f}%)  "
            self._write("")
            self.output.rule(style="dim", title=label)
            self._write("")

    # ---- lifecycle ---------------------------------------------------------

    def on_exit(self) -> None:
        """Persist memory and shut down before quitting."""
        if self.core:
            self.core.save_memory()
            self.core.shutdown()
