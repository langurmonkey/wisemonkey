"""
Textual TUI for Wisemonkey.

Provides a full-screen terminal UI with:
  - Title bar (Header) at the top
  - Scrollable output area (RichLog) in the middle
  - Text input at the bottom
  - Status bar below the input

Coexists with the terminal-based agent (agent/agent.py).
Launch with: wisemonkey --tui <session>
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from rich.align import Align
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, RichLog, Input, Static
from textual import work

from agent.core import Core, Stage, TurnCancelled
from agent.commands import registry
from agent.utils import contractuser, pretty_timedelta
from agent.startup import check_updates


class WisemonkeyTui(App):
    """Full-screen Textual TUI for Wisemonkey."""

    TITLE = "Wisemonkey"
    CSS = """
    Screen {
        layout: vertical;
    }

    #output {
        height: 1fr;
        border: none;
        padding: 0 1;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    #input-container {
        height: 3;
        border-top: solid $border;
        padding: 0 1;
        dock: bottom;
    }

    #input {
        width: 100%;
        margin-top: 0;
    }

    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
        dock: bottom;
    }

    Header {
        dock: top;
    }
    """

    def __init__(self, config_path: str | None = None, session: str = "default"):
        super().__init__()
        self.config_path = config_path
        self.session = session
        self.core: Core | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="output", highlight=True, markup=True, wrap=True)
        yield Container(
            Input(id="input", placeholder="Type a message...",),
            id="input-container",
        )
        yield Static(id="status-bar")

    def on_mount(self) -> None:
        """Set up the agent core and render startup info."""
        self.sub_title = f"Session: {self.session}"

        try:
            self.core = Core(self.config_path, self.session)
        except Exception as e:
            self._write(f"[red]Agent initialisation failed: {e}[/]")
            return

        self._write_startup_info()
        self._update_status()

    # ── output helpers ──────────────────────────────────────────

    def _write(self, text: str) -> None:
        """Append plain or markup text to the output log."""
        self.query_one("#output", RichLog).write(text)

    def _write_rich(self, renderable) -> None:
        """Append a Rich renderable to the output log."""
        self.query_one("#output", RichLog).write(renderable)

    # ── startup screen ──────────────────────────────────────────

    def _write_startup_info(self) -> None:
        """Render the welcome banner, version, session info."""
        monkee = r"""
                               .-"-.
                             _/.-.-.\_
                            ( ( o o ) )
                             |/  "  \|
                              \      /
                              /""""`\
                             /       \
        """
        wisemonkey = r"""

██     ██ ██ ▄██████ ██████ ██▄  ▄██ ▄███▄ ███  ██ ██ ▄█▀ ██████ ██  ██
██ ▄█▄ ██ ██ ▀▀▀▄▄▄ ██▄▄   ██ ▀▀ ██ ██ ██  ██ ██ ██ ▀▄██ ██▄▄   ██▄▄
 ▀██▀██▀  ██ █████▀ ██▄▄▄▄ ██    ██ ▀████▀ ██   ██ ██ ▀▄█ ██▄▄▄▄   ██
        """
        title = Align.center(f"{monkee}{wisemonkey}", vertical="middle")
        self._write_rich(Panel(
            title,
            box=box.HEAVY,
            border_style="green",
            subtitle="Monkee at your service!",
        ))

        # Version & update info
        try:
            from importlib.metadata import version as _ver
            pkg_version = _ver("wisemonkey")
        except Exception:
            pkg_version = "0.0.0-dev"

        now = datetime.now()
        agent_dir = Path(__file__).resolve().parent
        repo_dir = agent_dir.parent

        updates_available, commit_hash, last_check = check_updates(repo_dir)
        d_check = pretty_timedelta(now - last_check)

        line = f"[bold green]⇒ Wisemonkey[/bold green] [dim]v{pkg_version}[/dim]"
        if commit_hash:
            line += f"  [dim]commit: {commit_hash}[/dim]"
        self._write(line)
        if updates_available:
            self._write(f"  [orange1]⟳ Updates available[/orange1] [dim]({d_check} ago)[/dim]")
        elif commit_hash:
            self._write(f"  [green]✓ Up to date[/green] [dim]({d_check} ago)[/dim]")

        # Session info
        mem = self.core.memory
        d_created = pretty_timedelta(now - mem.session_created) if mem.session_created else "?"
        d_accessed = pretty_timedelta(now - mem.session_accessed) if mem.session_accessed else "?"
        working_dir = contractuser(Path(os.getcwd()))

        self._write("")
        if mem.session_is_new:
            self._write(f"[bold]⇒ Session created:[/bold] '[bold]{mem.session}[/bold]'")
        else:
            self._write(f"[bold]⇒ Session restored:[/bold] '[bold]{mem.session}[/bold]'")
        self._write(f"  [dim]location:      {contractuser(mem.session_dir)}[/dim]")
        self._write(f"  [dim]working dir:   {working_dir}[/dim]")
        self._write(f"  [dim]created:       {d_created} ago[/dim]")
        if not mem.session_is_new:
            self._write(f"  [dim]last accessed: {d_accessed} ago[/dim]")

        self._write_rich(Rule(style="dim"))
        self._write("[dim]Type [/dim][italic]/help[dim] for commands, [/dim][italic]/configure[dim] to configure[/dim]")
        self._write_rich(Rule(style="dim"))

    def _update_status(self) -> None:
        """Refresh the status bar text."""
        if not self.core:
            self.query_one("#status-bar", Static).update(" Not connected")
            return
        model = self.core.config.get("model.name", "?")
        sess = self.core.memory.session
        self.query_one("#status-bar", Static).update(
            f" Model: [bold]{model}[/bold]  |  Session: [bold]{sess}[/bold]"
        )

    # ── input handling ──────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Called when the user presses Enter in the input field."""
        user_input = event.value.strip()
        if not user_input:
            return

        self.query_one("#input", Input).clear()
        self._write(f"\n[gold1 bold]You:[/gold1 bold] {user_input}")

        if user_input.startswith("/"):
            self._handle_command(user_input)
        else:
            self._handle_prompt(user_input)

    def _handle_command(self, user_input: str) -> None:
        """Execute a slash command and print its result."""
        tokens = user_input.split()
        command, params = registry.lookup(tokens)

        if not command:
            self._write(f"[red]Command not found: {user_input}[/]")
            return

        ok_flag, msg, content, md, should_exit = registry.execute(
            self.core, command, params
        )

        if should_exit:
            self._write("[bold]Goodbye![/bold]")
            self.exit()
            return

        if ok_flag:
            if content:
                self._write(content)
            elif md:
                self._write_rich(Markdown(md))
            if msg:
                self._write(f"[green]✓[/green] {msg}")
        else:
            if msg:
                self._write(f"[red]✗[/red] {msg}")

        self._update_status()

    def _handle_prompt(self, user_input: str) -> None:
        """Send the user message to the LLM in a background thread."""
        self._write_rich(Rule(style="dim"))
        self._write(f"[medium_orchid bold]Wisemonkey:[/medium_orchid bold]")
        self._run_turn(user_input)

    # ── turn execution (threaded) ──────────────────────────────

    @work(thread=True, exit_on_error=False)
    def _run_turn(self, user_input: str) -> None:
        """Run a full turn on a worker thread so the UI stays responsive."""
        if not self.core:
            self.call_from_thread(self._write, "[red]Core not initialised[/]")
            return

        # Switch to "waiting" status
        self.call_from_thread(
            self.query_one("#status-bar", Static).update, " Processing…"
        )

        try:
            (response, total_tokens, ntools, total_gen_time) = self.core.run_turn(
                user_input,
                prompt_callback=None,
                reasoning_callback=None,
                content_callback=lambda c: self._append_content(c),
                tool_callback=lambda n, a: self._append_tool(n),
                cancel_callback=None,
                error_callback=None,
            )
            self.call_from_thread(self._finish_turn, response, total_tokens, ntools, total_gen_time)
        except TurnCancelled:
            self.call_from_thread(self._write, "[orange1]Turn cancelled[/]")
        except Exception as e:
            self.call_from_thread(self._write, f"[red]Error: {e}[/]")
        finally:
            self.call_from_thread(self._update_status)

    def _append_content(self, content: str) -> None:
        """Streaming callback – called from worker thread."""
        self.call_from_thread(self._write, content)

    def _append_tool(self, tool_name: str) -> None:
        """Tool activation callback – called from worker thread."""
        self.call_from_thread(
            self._write,
            f"[dim]🛠️ Activating tool: [steel_blue3]{tool_name}[/steel_blue3][/dim]",
        )

    def _finish_turn(
        self, response: str, tokens: int, ntools: int, gen_time: float
    ) -> None:
        """Called on the main thread after a turn completes."""
        if response == "[Cancelled]":
            return

        length, max_sz, rate = self.core.memory.get_chat_stats()
        label = f"  {gen_time:.1f}s  |  {tokens} tokens  |  {ntools} tools  |  Mem: {length}/{max_sz} ({rate:.2f}%)  "
        self._write("")
        self._write_rich(Rule(style="dim", title=label))
        self._write("")

    # ── lifecycle ──────────────────────────────────────────────

    def on_exit(self) -> None:
        """Persist memory and shut down before quitting."""
        if self.core:
            self.core.save_memory()
            self.core.shutdown()