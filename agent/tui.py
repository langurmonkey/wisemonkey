"""Full-screen TUI for Wisemonkey using prompt_toolkit.

Replaces the simple REPL loop in agent.py with a full-screen terminal
application featuring a scrollable output area, a persistent input prompt,
and support for cancellation via Ctrl+C.

agent/agent.py is kept alongside this file for backwards compatibility.
"""

import datetime
import threading
import time
from functools import partial
from pathlib import Path

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.application import get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, NestedCompleter, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML, ANSI, merge_formatted_text, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text, split_lines
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.scroll import scroll_half_page_down, scroll_half_page_up
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    ConditionalContainer,
    Dimension,
    HSplit,
    ScrollablePane,
    VSplit,
    Window,
    WindowAlign,
)
from prompt_toolkit.layout.containers import to_container
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.processors import (
    AppendAutoSuggestion,
    ConditionalProcessor,
    HighlightMatchingBracketProcessor,
    HighlightSelectionProcessor,
    PasswordProcessor,
    TabsProcessor,
    TransformationInput,
    merge_processors,
)
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.shortcuts import PromptSession as PTSession
from prompt_toolkit.styles import Style
from prompt_toolkit.clipboard import InMemoryClipboard
from prompt_toolkit.output import ColorDepth
from pygments.lexers.markup import MarkdownLexer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text
from pubsub import pub

from agent.core import Core, Stage, TurnCancelled
from agent.commands import registry
from agent.utils import add_command, collapse_none_dicts, contractuser, pretty_timedelta
from agent.console import print, err, ok, info, newline, console as rich_console
from agent.startup import startup_info

# Constants
PASTE_THRESHOLD = 1500
txt_goodbye = "\nGoodbye!"


class _AnsiBufferControl(BufferControl):
    """BufferControl subclass that stores ANSI text lines for the output area.

    Instead of using prompt_toolkit's formatted text (which is slow for large
    buffers), we keep a list of ANSI strings and render them on the fly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ansi_lines: list[str] = []


class TuiAgent:
    """Full-screen TUI agent using prompt_toolkit."""

    def __init__(self, config_path=None, session="default"):
        self.core = Core(config_path, session)
        self._last_ctrl_c_time = 0.0
        self._cancel_event = threading.Event()
        self._turn_thread: threading.Thread | None = None
        self._output_text: list[str] = []  # ANSI strings appended to the output
        self._is_generating = False
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._spinner_timer = None

        # prompt_toolkit components (created in run())
        self._app: Application | None = None
        self._output_buffer: Buffer | None = None
        self._output_control = None
        self._output_window = None
        self._status_control = None
        self._status_window = None
        self._session: PromptSession | None = None

        pub.subscribe(self._on_prompt_update, "prompt-update")

    # ------------------------------------------------------------------ #
    #  Output helpers                                                     #
    # ------------------------------------------------------------------ #

    def _append_output(self, text: str = "", style: str = ""):
        """Append Rich-rendered text to the output area.

        If *style* is given we wrap *text* in Rich markup, render to ANSI,
        and append.  Otherwise *text* is treated as a plain ANSI string.
        """
        if style:
            ansi = self._rich_to_ansi(f"[{style}]{text}[/{style}]")
        elif text:
            ansi = text
        else:
            return

        self._output_text.append(ansi)
        self._refresh_output()

    def _append_output_raw(self, ansi: str):
        """Append a pre-rendered ANSI string to the output area."""
        if not ansi:
            return
        self._output_text.append(ansi)
        self._refresh_output()

    def _refresh_output(self):
        """Rebuild the output buffer from self._output_text and scroll to end."""
        if self._output_buffer is None:
            return
        full_text = "".join(self._output_text)
        self._output_buffer.text = full_text
        # Scroll to the end
        self._output_buffer.cursor_position = len(full_text)

    def _clear_output(self):
        self._output_text.clear()
        if self._output_buffer is not None:
            self._output_buffer.text = ""

    @staticmethod
    def _rich_to_ansi(rich_text: str) -> str:
        """Render a Rich markup string to an ANSI string."""
        buf = __import__("io").StringIO()
        c = Console(file=buf, force_terminal=True, color_system="truecolor", width=120)
        c.print(rich_text, end="")
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    #  Status bar                                                         #
    # ------------------------------------------------------------------ #

    def _update_status(self, text: str = ""):
        if self._status_control is None:
            return
        self._status_control.text = HTML(f" {text}")

    def _status_idle(self):
        model = self.core.config.get("model.name", "?")
        self._update_status(f"Ready — {model} — Enter: submit  |  Alt+Enter: newline  |  Ctrl+C: clear/quit")

    def _status_thinking(self):
        frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._spinner_idx += 1
        self._update_status(f"{frame} Thinking...")

    def _status_generating(self):
        frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._spinner_idx += 1
        self._update_status(f"{frame} Generating...")

    def _status_done(self, total_tokens, ntools, total_gen_time):
        length, max_, rate = self.core.memory.get_chat_stats()
        self._update_status(
            f"{total_gen_time:.1f}s  |  {total_tokens} tokens  |  {ntools} tools  |  Mem: {length}/{max_} ({rate:.2f}%)"
        )

    # ------------------------------------------------------------------ #
    #  Spinner / thinking animation                                       #
    # ------------------------------------------------------------------ #

    def _start_spinner(self, mode="thinking"):
        self._is_generating = True
        self._spinner_idx = 0
        self._spinner_mode = mode
        self._tick_spinner()

    def _tick_spinner(self):
        if not self._is_generating:
            return
        if self._spinner_mode == "thinking":
            self._status_thinking()
        else:
            self._status_generating()
        self._spinner_timer = threading.Timer(0.1, self._tick_spinner)
        self._spinner_timer.daemon = True
        self._spinner_timer.start()

    def _stop_spinner(self):
        self._is_generating = False
        if self._spinner_timer:
            self._spinner_timer.cancel()
            self._spinner_timer = None

    # ------------------------------------------------------------------ #
    #  Callbacks (called from core.run_turn and from the turn thread)     #
    # ------------------------------------------------------------------ #

    def _on_prompt_update(self):
        """Called when config changes require prompt session recreation."""
        # In TUI mode we recreate the session lazily on next input.
        pass

    def prompt_callback(self, stage: Stage):
        if stage == Stage.START:
            self._start_spinner("thinking")
        elif stage == Stage.STOP:
            self._stop_spinner()
            self._status_idle()

    def reasoning_callback(self, stage: Stage, content: str = "", reasoning_visible: bool = True):
        if stage == Stage.START:
            if reasoning_visible:
                self._append_output_raw(self._rich_to_ansi("💡 Thinking...\n"))
        elif stage == Stage.PROCESS:
            if reasoning_visible:
                self._append_output_raw(self._rich_to_ansi(f"[weak]{escape(content)}[/]"))
        elif stage == Stage.STOP:
            if reasoning_visible:
                self._append_output_raw(self._rich_to_ansi("💡 Done thinking\n"))

    def content_callback(self, content: str = ""):
        self._append_output_raw(escape(content))

    def tool_callback(self, tool_name: str, tool_args):
        self._append_output_raw(self._rich_to_ansi(f"\n🔧 Activating tool:  [tool]{tool_name}[/tool]\n"))

    def cancel_callback(self, e: KeyboardInterrupt):
        self._append_output_raw(self._rich_to_ansi("\n⏹  Turn cancelled by user  ⏹\n"))
        raise TurnCancelled() from e

    def error_callback(self, e, msg):
        raise RuntimeError(msg) from e

    # ------------------------------------------------------------------ #
    #  Slash commands                                                     #
    # ------------------------------------------------------------------ #

    def _handle_slash_command(self, user_input: str):
        tokens = user_input.split()
        command, params = registry.lookup(tokens)

        if command:
            no_errors, msg, content, md, should_exit = registry.execute(self.core, command, params)

            if should_exit:
                self._append_output_raw(self._rich_to_ansi(txt_goodbye))
                self._app.exit()
                return

            if no_errors:
                if content or md:
                    param_list = " ".join(params) if params else ""
                    if content:
                        cont = content
                    else:
                        cont = md
                    ansi = self._rich_to_ansi(
                        f"[accent-bold]{command.name} {param_list}[/accent-bold]\n{cont}"
                    )
                    self._append_output_raw(ansi)
                if msg:
                    self._append_output(msg, style="ok")
            else:
                if msg:
                    self._append_output(msg, style="err")
        else:
            self._append_output(f"Command not found: {user_input}", style="err")

    # ------------------------------------------------------------------ #
    #  Turn execution (runs in a background thread)                       #
    # ------------------------------------------------------------------ #

    def _run_turn_thread(self, user_input: str):
        """Run a turn in a background thread and post results back to the TUI."""
        prompt_cb = partial(self.prompt_callback)
        reasoning_cb = partial(self.reasoning_callback)
        content_cb = partial(self.content_callback)
        tool_cb = partial(self.tool_callback)
        cancel_cb = partial(self.cancel_callback)
        error_cb = partial(self.error_callback)

        try:
            (response, total_tokens, ntools, total_gen_time) = self.core.run_turn(
                user_input,
                prompt_cb,
                reasoning_cb,
                content_cb,
                tool_cb,
                cancel_cb,
                error_cb,
            )
            # Schedule status update on the main thread
            if self._app:
                self._app.call_from_executor(
                    lambda: self._status_done(total_tokens, ntools, total_gen_time)
                )
        except TurnCancelled:
            pass
        except Exception as e:
            self._append_output_raw(self._rich_to_ansi(f"\nError sending prompt: {e}\n"))
            self._append_output_raw(self._rich_to_ansi("  Partial response was saved to chat history.\n"))
        finally:
            self._stop_spinner()
            if self._app:
                self._app.call_from_executor(self._status_idle)

    def _submit_input(self, user_input: str):
        """Handle submitted user input (called from the prompt session)."""
        user_input = user_input.strip()
        if not user_input:
            return

        # Echo user input
        model = self.core.config.get("model.name", "?")
        self._append_output_raw(
            self._rich_to_ansi(f"\n[agent]▕ [bold]You[/bold] ▏[/agent]  [accent]⇒ {model}[/accent]\n")
        )
        self._append_output_raw(self._rich_to_ansi(f"{user_input}\n"))

        # Slash commands run synchronously (they're fast)
        if user_input.startswith("/"):
            self._handle_slash_command(user_input)
            return

        # Run the turn in a background thread
        self._cancel_event.clear()
        self._turn_thread = threading.Thread(
            target=self._run_turn_thread,
            args=(user_input,),
            daemon=True,
        )
        self._turn_thread.start()

    # ------------------------------------------------------------------ #
    #  Key bindings                                                       #
    # ------------------------------------------------------------------ #

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            buf = event.current_buffer
            text = buf.text
            # If the user is mid-paste or has selected text, let default handler work
            if text:
                buf.validate_and_handle()
            else:
                buf.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _(event):
            buf = event.current_buffer
            now = time.time()
            if buf.text:
                buf.reset()
                self._last_ctrl_c_time = int(now)
            elif now - self._last_ctrl_c_time < 1.0:
                # Double Ctrl+C → quit
                self._app.exit()
            else:
                buf.reset()
                self._last_ctrl_c_time = int(now)

        @kb.add(Keys.BracketedPaste)
        def _(event):
            text = event.data
            if len(text) > PASTE_THRESHOLD:
                file_path = self.core.memory.create_pasted_file(text)
                text = f"*Pasted file: {file_path}*\n"
            event.current_buffer.insert_text(text)

        return kb

    # ------------------------------------------------------------------ #
    #  Layout                                                             #
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> Layout:
        # Output area — a Buffer inside a ScrollablePane
        self._output_buffer = Buffer(read_only=True, multiline=True)
        self._output_control = BufferControl(
            buffer=self._output_buffer,
            focusable=False,
            preview_search=False,
        )
        self._output_window = Window(
            content=self._output_control,
            wrap_lines=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
            dont_extend_height=True,
            style="class:output",
        )

        # Status bar
        self._status_control = FormattedTextControl(
            text=HTML(" Ready"),
            focusable=False,
            show_cursor=False,
        )
        self._status_window = Window(
            content=self._status_control,
            height=1,
            style="class:statusbar",
            dont_extend_width=False,
            align=WindowAlign.LEFT,
        )

        # Input area — PromptSession handles this via the `bottom_toolbar`
        # We use a separate PromptSession for input
        root = HSplit(
            [
                self._output_window,
                Window(height=1, char="─", style="class:separator"),
                self._status_window,
            ]
        )
        return Layout(root, focused_element=self._output_window)

    def _build_style(self) -> Style:
        return Style.from_dict(
            {
                "output": "",
                "statusbar": "bg:#262626 #ffffff",
                "separator": "grey",
                "prompt": "ansiyellow",
                "frame.border": "ansiyellow",
                "bottom-toolbar": "#ffffff bg:#262626 noreverse",
                "kbd": "#ffd787 bold",
                "model": "#0087d7",
            }
        )

    # ------------------------------------------------------------------ #
    #  Prompt session (input)                                             #
    # ------------------------------------------------------------------ #

    def _create_prompt_session(self) -> PromptSession:
        kb = self._create_key_bindings()

        vi_mode = self.core.config.get("agent.vi_mode", False)

        # Build slash command completer
        commands = [cmd.name for cmd in registry.list_commands()]
        commands_dict = {}
        for command in commands:
            add_command(commands_dict, command)
        commands_dict = collapse_none_dicts(commands_dict)
        slash_completer = NestedCompleter.from_nested_dict(commands_dict)
        path_completer = PathCompleter()

        class HybridCompleter(Completer):
            def __init__(self, slash_comp, path_comp):
                self.slash_completer = slash_comp
                self.path_completer = path_comp

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if text.startswith("/"):
                    cmds = list(self.slash_completer.get_completions(document, complete_event))
                    if cmds:
                        return cmds
                words = text.rsplit(None, 1)
                last_word = words[-1] if words else ""
                last_word_stripped = last_word.lstrip("/")
                if last_word_stripped and (
                    "/" in last_word_stripped
                    or last_word_stripped.startswith("~")
                    or last_word_stripped.startswith(".")
                ):
                    fake_doc = Document(
                        text=last_word_stripped,
                        cursor_position=len(last_word_stripped),
                    )
                    path_completions = list(
                        self.path_completer.get_completions(fake_doc, complete_event)
                    )
                    if path_completions:
                        offset = len(text) - len(last_word_stripped)
                        for c in path_completions:
                            c.start_position += offset
                        return path_completions
                if text.startswith("~") or text.startswith(".") or text.startswith(".."):
                    return self.path_completer.get_completions(document, complete_event)
                return self.path_completer.get_completions(document, complete_event)

        completer = HybridCompleter(slash_completer, path_completer)

        history_path = self.core.memory.session_dir / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        model = self.core.config.get("model.name", "?")

        def prompt_toolbar():
            return HTML(
                "  <kbd>Alt</kbd>+<kbd>↵</kbd>: new line | <kbd>↵</kbd>: submit | <kbd>Ctrl</kbd>+<kbd>C</kbd>: clear / double-tap to quit"
            )

        session = PromptSession(
            message=HTML(f"▕ You ▏ <model>⇒ {model}</model>\n❯ "),
            history=FileHistory(str(history_path)),
            style=self._build_style(),
            multiline=True,
            key_bindings=kb,
            vi_mode=vi_mode,
            clipboard=InMemoryClipboard(),
            enable_open_in_editor=vi_mode,
            complete_while_typing=True,
            complete_in_thread=True,
            completer=completer,
            auto_suggest=AutoSuggestFromHistory(),
            lexer=PygmentsLexer(MarkdownLexer),
            bottom_toolbar=prompt_toolbar,
        )
        return session

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run_interactive(self):
        """Run the TUI agent (matches Agent interface)."""
        self._session = self._create_prompt_session()

        # Print startup info into the output area
        self._print_startup_info()

        self._status_idle()

        # Main input loop
        while True:
            try:
                user_input = self._session.prompt()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue

            self._submit_input(user_input)

        # Shutdown
        if self.core:
            self.core.save_memory()
            self.core.shutdown()

    def _print_startup_info(self):
        """Render startup info into the output buffer."""
        import io
        import shutil
        from rich import box
        from rich.align import Align
        from rich.panel import Panel
        from importlib.metadata import version, PackageNotFoundError

        term_size = shutil.get_terminal_size((80, 20))

        monkee = '''
                               .-"-.
                             _/.-.-.\\_
                            ( ( o o ) )
                             |/  "  \\|
                              \\ ܁܁ ⿽  /
                              /"""\\\\
                             /       \\
    '''
        if term_size.columns < 80:
            wisemonkey = "WISEMONKEY"
        else:
            wisemonkey = '''
                                                                    
██     ██ ██ ▄████▀▀▀ ████▀▀▀▀ ██▀  ▄██ ▄████▀▀▄ ██▀▀  ██ ██ ▄█▀ ████▀▀▀ ██  ██ 
██ ▄█▄ ██ ██ ██ ▀▀▀▄▄▄ ██▄▄   ██ ▀▀ ██ ██  ██ ██  ██ ▀▄██ ██ ██▀▀▀  ██▄▄    ▀██▀  
 ▀██▀██▀  ██ ██ ██████ ▀████▄▄▄ ██    ██ ██  ██ ██  ██ ▀██▀ ██ ██ ▄█▀ ██▄▄▄▄   ██   
        '''

        buf = io.StringIO()
        c = Console(file=buf, force_terminal=True, color_system="truecolor", width=120)

        title = Align.center(f"[title]{monkee}{wisemonkey}[/title]", vertical="middle")
        c.print(Panel(title, box=box.HEAVY, border_style="title", subtitle="Monkee at your service!"))
        c.print()

        new_session = self.core.memory.session_is_new
        session_dir = self.core.memory.session_dir
        working_dir = contractuser(Path.cwd())
        created = self.core.memory.session_created
        accessed = self.core.memory.session_accessed

        try:
            pkg_version = version("wisemonkey")
        except PackageNotFoundError:
            pkg_version = "0.0.0-dev"

        now = datetime.datetime.now()
        agent_dir = Path(__file__).resolve().parent
        repo_dir = agent_dir.parent

        from agent.startup import check_updates
        updates_available, commit_hash, last_check = check_updates(repo_dir)
        d_check = pretty_timedelta(now - last_check)

        c.print(f"[accent]Wisemonkey[/accent] [dim]v{pkg_version}[/dim]")
        if commit_hash:
            c.print(f"  [dim]commit: {commit_hash}[/dim]")
        if updates_available:
            c.print(f"   [warn]⟳ Updates available![/warn] [time](last check: {d_check} ago)[/time]")
            c.print("     [weak]run [accent]wisemonkey -u[/accent] to update[/weak]")
        elif commit_hash:
            c.print(f"   [dim]✓ Up to date[/dim] [time](last check: {d_check} ago)[/time]")

        c.print()

        d_created = pretty_timedelta(now - created) if created else None
        d_accessed = pretty_timedelta(now - accessed) if accessed else None
        if new_session:
            c.print(f"Session created: [accent-bold]'{self.core.memory.session}'[/accent-bold]")
        else:
            c.print(f"Session restored: [accent-bold]'{self.core.memory.session}'[/accent-bold]")
        c.print(f"[dim]   location:      {contractuser(session_dir)}[/dim]")
        c.print(f"[dim]   working dir:   {working_dir}[/dim]")
        c.print(f"[dim]   created:[/dim]       [time]{d_created} ago[/time]")
        if not new_session:
            c.print(f"[dim]   last accessed:[/dim] [time]{d_accessed} ago[/time]")

        c.print()

        chat_history = self.core.memory.get_chat_formatted(num_exchanges=3, timestamps=True, width=250)
        if chat_history:
            curr, max_, rate = self.core.memory.get_chat_stats()
            c.print(
                Panel(
                    Markdown(chat_history),
                    border_style="output-frame",
                    title="Previous conversation (last 3 exchanges, truncated)",
                    subtitle=f"Previous conversation stats: {curr}/{max_} - {rate:.2f}%",
                )
            )

        c.print()
        c.print("[weak]Type [accent]/configure[/accent] to configure the agent interactively[/weak]")
        c.print("[weak]Type [accent]/help[/accent] for command information[/weak]")

        self._append_output_raw(buf.getvalue())
