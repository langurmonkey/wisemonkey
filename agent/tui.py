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

import os
import re
import threading

from rich.markdown import Markdown
from rich.panel import Panel

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import TextArea, Header, RichLog, Static, Footer
from textual import work
from textual.binding import Binding
from textual.timer import Timer
from textual.events import Paste as PasteEvent

from agent.core import Core, Stage, TurnCancelled
from agent.commands import registry, Command
from agent.history import History
from agent.output import TuiOutputAdapter, set_output
from agent.startup import startup_info

# Number of characters above which the paste action creates a file
PASTE_THRESHOLD = 1000
# Regex to find a path-like token immediately left of the cursor.
# Matches an optional ~ or leading / followed by any non-whitespace chars
# that look like path components.
_PATH_RE = re.compile(r'(?:^|(?<=\s))(~?/?(?:[^\s]*/)+[^\s]*|~?/[^\s]*)$')
# Spinner characters
SPINNER_CHARS ="⣾⣽⣻⢿⡿⣟⣯⣷"

class _PromptInput(TextArea):
    """TextArea that handles history on up/down, Ctrl+C to clear, and paste threshold."""

    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("shift+enter", "newline", "New line"),
        Binding("ctrl+c", "clear", "Clear"),
        Binding("ctrl+o", "open_in_pager", "Open chat in pager"),
    ]

    # Special autocomplete list
    # When this is set, COMMANDS is ignored
    SPECIAL: list[str] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_ctrl_c_time: float = 0.0

        # Autocomplete
        self.COMMANDS = []
        for cmd in registry.list_commands():
            self.COMMANDS.append(cmd.name)
            if '-' in cmd.name:
                self.COMMANDS.append(cmd.name.replace('-', ' '))

        # Special autocomplete list
        # When this is set, COMMANDS is ignored
        self.SPECIAL: list[str] | None = None

    def set_core(self, core: Core | None):
        self.core = core

    def set_special_suggestions(self, sp: list[str] | None):
        self.SPECIAL = sp

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
        row, _ = self.cursor_location
        if row == 0:
            self._wm_app.action_history_up(self.text)
        else:
            super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        """Override: navigate history when at very end of last line."""
        row, _ = self.cursor_location
        lines = self.text.split("\n")
        if row == len(lines) - 1:
            self._wm_app.action_history_down()
        else:
            super().action_cursor_down(select)

    def action_clear(self) -> None:
        """Clear text"""
        self.clear()

    def action_open_in_pager(self) -> None:
        """Write the current session log to a temp file and open it in $PAGER."""
        import subprocess
        import tempfile

        if not self.core:
            return

        history = self.core.memory.get_chat_unformatted()

        lines = []
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if isinstance(content, list):
                # tool-use turns have content as a list of blocks
                content = "\n".join(
                    block.get("text", "") for block in content if block.get("type") == "text"
                )
            label = "You" if role == "user" else "Wisemonkey"
            lines.append(f"{'─' * 60}\n{label}\n{'─' * 60}\n{content}\n")

        text = "\n".join(lines)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="wisemonkey_", delete=False
        ) as f:
            f.write(text)
            tmp_path = f.name

        pager = os.environ.get("PAGER", "less")
        with self.app.suspend():
            subprocess.call([pager, tmp_path])

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

    def update_suggestion(self) -> None:
        row, col = self.cursor_location
        line = self.document.get_line(row)
        current = line[:col]  # text on this line up to the cursor

        if self.SPECIAL:
            # Use special list
            candidates = [
                c for c in self.SPECIAL
                if c.startswith(current) and c != current
            ]
            if candidates:
                best = min(candidates, key=len)
                self.suggestion = best[len(current):]
                return
            self.suggestion = ""
            return
            
        else:
            # Slash commands
            if line.startswith("/") and col == len(line):
                candidates = [
                    c for c in self.COMMANDS
                    if c.startswith(current) and c != current
                ]
                if candidates:
                    best = min(candidates, key=len)
                    self.suggestion = best[len(current):]
                    return
                self.suggestion = ""
                return

            # File system paths
            m = _PATH_RE.search(current)
            if m:
                token = m.group(0)
                suffix = self._path_suggestions(token)
                self.suggestion = suffix
                return

            self.suggestion = ""

    @staticmethod
    def _path_suggestions(token: str) -> str:
        """Return the completion suffix for *token*, or '' if none."""
        expanded = os.path.expanduser(token)

        # Decide what directory to scan and what prefix to match against.
        if expanded.endswith("/"):
            # User typed a full dir path ending in /  -> list contents
            directory = expanded
            prefix = ""
            # The suggestion should start with nothing (entries are below the slash)
            offset = 0
        else:
            directory = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)
            offset = len(prefix)

        try:
            entries = os.scandir(directory)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return ""

        matches = []
        with entries:
            for entry in entries:
                if entry.name.startswith(prefix) and entry.name != prefix:
                    name = entry.name
                    if entry.is_dir(follow_symlinks=False):
                        name += "/"
                    matches.append(name)

        if not matches:
            return ""

        # Prefer the shortest match so Tab always advances one component.
        best = min(matches, key=len)
        return best[offset:]

    def watch_selection(self) -> None:
        # selection changes whenever the cursor moves (typing, arrows, etc.)
        self.update_suggestion()

    async def _on_key(self, event: events.Key) -> None:
        if self.suggestion and event.key in ("right", "ctrl+f", "tab"):
            # accept: insert the suggestion text at the cursor
            self.insert(self.suggestion)
            self.suggestion = ""
            event.prevent_default()
            event.stop()
        elif event.key == "escape" and self.suggestion:
            self.suggestion = ""
            event.prevent_default()
            event.stop()



class WisemonkeyTui(App):
    """Full-screen Textual TUI for Wisemonkey."""

    TITLE = "Wisemonkey"
    CSS_PATH = "../styles/wm.tcss"

    def __init__(self, config_path: str | None = None, session: str = "default"):
        super().__init__()
        self.config_path = config_path
        self.session = session
        self.core: Core | None = None
        self.output: TuiOutputAdapter = TuiOutputAdapter(self)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="output", highlight=True, markup=True, wrap=True)
        with Container(id="bottom-area"):
            yield Static(id="status-bar")
            with Container(id="input-widget"):
                prompt = _PromptInput.code_editor(id="input",
                                                  placeholder="Type a message...",
                                                  soft_wrap=True,
                                                  language="markdown")
                yield prompt
            yield Footer(id="bottom-bar")

    def on_mount(self) -> None:
        """Set up the agent core and render startup info."""
        self.sub_title = f"Session: {self.session}"

        try:
            self.core = Core(self.config_path, self.session)
        except Exception as e:
            self.output._write(f"[err]Agent initialisation failed: {e}[/]")
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
        self._prompt_ui = TuiOutputAdapter(self)
        set_output(self._prompt_ui)

        # Input history
        sess_dir = str(self.core.memory.session_dir)
        self._history = History(sess_dir)

        # Focus the input field by default
        inp = self.query_one("#input", _PromptInput)
        inp.set_core(self.core)
        inp.focus()

        self._cancel_event = threading.Event()
        self._turn_active = False

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
            self.call_from_thread(self.query_one("#status-bar", Static).update, "💡 Thinking...")
            self._thinking_spinner_chars = SPINNER_CHARS
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
                    self.output.print(f"[dim]{buf}[/dim]")

        elif stage == Stage.STOP:
            if self._cancel_event.is_set():
                # Swallow the "Done thinking" message on cancellation
                if self._thinking_spinner_interval:
                    self._thinking_spinner_interval.stop()
                    self._thinking_spinner_interval = None
                self._reasoning_buffer = ""
                return

            if self._thinking_spinner_interval:
                self._thinking_spinner_interval.stop()
                self._thinking_spinner_interval = None
            # Flush any remaining reasoning content
            if self._reasoning_buffer:
                self.output.print(f"[dim]{self._reasoning_buffer}[/dim]")
                self._reasoning_buffer = ""

            self.output.ok("💡 Done thinking")
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
            self._prompt_spinner_chars = SPINNER_CHARS
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
        tokens = user_input.split()
        command, params = registry.lookup(tokens)

        if not command:
            self.output.err(f"Command not found: {user_input}")
            return

        self._run_command_in_thread(command, params)

    @work(thread=True, exit_on_error=False)
    def _run_command_in_thread(self, command: Command, params: list[str]) -> None:
        """Run a slash command on a worker thread."""
        ok_flag, msg, content, md, should_exit = registry.execute(
            self.core, command, params, self._prompt_ui
        )

        if should_exit:
            self.output.print("[bold]Goodbye![/bold]")
            self.call_from_thread(self.exit)
            return

        if ok_flag:
            if params:
                param_list = ' '.join(params)
            else:
                param_list = ''
            if content:
                panel = Panel(content,
                            border_style="output-frame",
                            title=f"{command.name} {param_list}",
                            subtitle=f"{command.name} {param_list}",
                            highlight=True)
                self.output.print_rich(panel)
            elif md:
                panel = Panel(Markdown(md),
                            border_style="output-frame",
                            title=f"{command.name} {param_list}",
                            subtitle=f"{command.name} {param_list}",
                            highlight=True)
                self.output.print_rich(panel)

            if msg:
                self.output.ok(msg)
        else:
            if msg:
                self.output.err(msg)

        self.call_from_thread(self._update_status)

    def _handle_prompt(self, user_input: str) -> None:
        """Send the user message to the LLM in a background thread."""
        self.output.rule(style="user")
        self.output.rule(style="agent", title="[agent]▶▶▶ Wisemonkey[/agent]", align="left")
        self._stream_buffer = ""
        self._run_turn(user_input)

    BINDINGS = [
        # Enter / shift+enter are handled by _PromptInput
        # Up / down for history are handled by _PromptInput (cursor-boundary logic)
        Binding("ctrl+s", "cancel_turn", "Cancel turn"),
    ]

    def action_cancel_turn(self) -> None:
        """Signal the running worker thread to cancel the current turn."""
        if self._cancel_event:
            self._cancel_event.set()


    def action_newline(self) -> None:
        """Insert a newline in the text area."""
        inp = self.query_one("#input", _PromptInput)
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
        inp = self.query_one("#input", _PromptInput)
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

    def action_history_up(self, current_text: str) -> None:
        """Move up in history - only if called from _SubmitTextArea (cursor at start)."""
        entry = self._history.up(current_text)
        if entry is not None:
            inp = self.query_one("#input", _PromptInput)
            inp.text = entry
            inp.cursor_location = (0, 0)

    def action_history_down(self) -> None:
        """Move down in history - only if called from _SubmitTextArea (cursor at end)."""
        entry = self._history.down()
        inp = self.query_one("#input", _PromptInput)
        if entry is not None:
            inp.text = entry
            lines = entry.split("\n")
            inp.cursor_location = (len(lines) - 1, len(lines[-1]))


    def _handle_user_input(self, user_input: str) -> None:
        """Process a user message or slash command."""
        self.output.newline()
        self.output.rule(style="user", title="[user]▶▶▶ You[/user]", align="left")
        self.output.print(f"{user_input}")
        self.output.newline()

        if user_input.startswith("/"):
            self._handle_command(user_input)
        else:
            self._handle_prompt(user_input)

    def set_special_suggestions(self, sp: list[str] | None):
        inp = self.query_one("#input", _PromptInput)
        inp.set_special_suggestions(sp)

    @work(thread=True, exit_on_error=False)
    def _run_turn(self, user_input: str) -> None:
        """Run a full turn on a worker thread so the UI stays responsive."""
        if not self.core:
            self.output.err("Core not initialised")
            return

        self.call_from_thread(
            self.query_one("#status-bar", Static).update, " Processing\u2026"
        )

        # Reset before each turn
        self._cancel_event.clear()
        self._turn_active = True

        def poll():
            return self._cancel_event.is_set()

        try:
            (response, total_tokens, ntools, total_gen_time) = self.core.run_turn(
                user_input,
                prompt_callback=lambda s: self.call_from_thread(self._prompt_callback, s),
                reasoning_callback=self._reasoning_callback,
                content_callback=self._append_content,
                tool_callback=self._append_tool,
                cancel_callback=self._cancel_cb,
                error_callback=None,
                poll=poll
            )
            self.call_from_thread(
                self._finish_turn, response, total_tokens, ntools, total_gen_time
            )
        except TurnCancelled:
            self.output.err("Turn cancelled")
        except Exception as e:
            self.output.err(f"Error: {e}")
        finally:
            self._turn_active = False
            self.call_from_thread(self._update_status)

    def _append_content(self, content: str) -> None:
        """Streaming callback – called from worker thread.

        Accumulates tokens in a buffer and only flushes complete lines
        to the RichLog when a newline is encountered.  This preserves
        line breaks (so RichLog wraps at natural line endings) while
        still preventing every token from becoming its own entry.
        """
        if self._cancel_event.is_set():
            return

        parts = content.split("\n")
        # First part goes into the buffer (may be partial).
        self._stream_buffer += parts[0]

        if len(parts) > 1:
            # Flush the completed line.
            self.output.print(self._stream_buffer)
            self._stream_buffer = ""
            # Remaining parts (except the last) are complete lines.
            for p in parts[1:-1]:
                self.output.print(p)
            # Last part starts a new buffer (still accumulating).
            self._stream_buffer = parts[-1]

    def _append_tool(self, tool_name: str, tool_args, captured_output: str = "") -> None:
        """Tool activation callback – called from worker thread."""
        text = f"[dim]🛠️ Activating tool: [steel_blue3]{tool_name}[/steel_blue3][/dim]"
        if captured_output:
            text += "\n" + captured_output
        self.output.print(text)

    def _finish_turn(
        self, response: str, tokens: int, ntools: int, gen_time: float
    ) -> None:
        """Called on the main thread after a turn completes."""
        if self._cancel_event.is_set():
            self._stream_buffer = ""  # discard anything left
            return

        # Flush any remaining buffered content
        if self._stream_buffer:
            self.output.print(self._stream_buffer)
            self._stream_buffer = ""

        if response == "[Cancelled]":
            return

        if self.core:
            length, max_sz, rate = self.core.memory.get_chat_stats()
            label = f"{gen_time:.1f}s  |  {tokens} tokens  |  {ntools} tools  |  Mem: {length}/{max_sz} ({rate:.2f}%)"
            self.output.newline()
            self.output.rule(style="agent", title=label)

    def _cancel_cb(self, e) -> None:
        self.output.err("Turn cancelled")
        raise TurnCancelled() from e

    # ---- lifecycle ---------------------------------------------------------

    def on_exit(self) -> None:
        """Persist memory and shut down before quitting."""
        if self.core:
            self.core.save_memory()
            self.core.shutdown()
