"""The agent loop of Wisemonkey.

The agent orchestrates the 'user-assistant' turns and delegates the actual turn
handling to the core.
"""

import time
import threading

from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from pubsub import pub

from agent.core import Core
from agent.emitter import TurnEmitter
from agent.ipc import (
    ContentPayload,
    Event,
    ReasoningPayload,
    StageKind,
    StagePayload,
    ToolCallPayload,
    ToolResultPayload,
    loopback_pair,
    payload_as,
)
from agent.commands import registry
from agent.utils import add_command, collapse_none_dicts, format_tool_args
from agent.output import RichOutputAdapter, set_output
from agent.console import print, err, ok, info, newline
from agent.startup import startup_info

# Try to import prompt_toolkit for rich input; fall back to plain input.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import NestedCompleter, PathCompleter, Completer
    from prompt_toolkit.clipboard import InMemoryClipboard
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.markup import MarkdownLexer
    _HAS_PROMPT_TOOLKIT = True

except ImportError:
    err("Could not initialize prompt toolkit")
    _HAS_PROMPT_TOOLKIT = False


# Constants
txt_goodbye = "\n[accent-bold]Goodbye![/accent-bold]"
PASTE_THRESHOLD = 1500

class Agent:
    def __init__(self, config_path=None, session='default'):
        self.core = Core(config_path, session)
        self.spinner_prompt = None
        self.spinner_thinking = None
        self._last_ctrl_c_time = 0  # Timestamp of last Control+C for double-tap detection
        self.output = RichOutputAdapter()
        set_output(self.output)
        # Event emitter (phase 1 of the client/server split): core callbacks
        # are translated into IPC events on a loopback transport and dispatched
        # by an event pump running on a background thread.
        transport, peer = loopback_pair()
        self.emitter = TurnEmitter(transport=transport)
        self._event_peer = peer
        self._event_thread = None
        self._events_running = False
        self._turn_in_progress = False
        pub.subscribe(self._create_prompt_session, "prompt-update")

    # ── event handling (IPC phase 1) ────────────────────────────────────────

    def start_events(self) -> None:
        """Start the event pump that renders IPC events from the emitter."""
        if self._event_thread is not None:
            return
        self._events_running = True
        self._event_thread = threading.Thread(
            target=self._event_loop, name="wisemonkey-events", daemon=True
        )
        self._event_thread.start()

    def stop_events(self, timeout: float = 2.0) -> None:
        """Stop the event pump and close the loopback transport."""
        self._events_running = False
        transport = self.emitter.transport
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        if self._event_thread is not None:
            self._event_thread.join(timeout=timeout)
            self._event_thread = None

    def _event_loop(self) -> None:
        """Drain events from the loopback peer until stopped.

        This runs on a dedicated thread; every UI mutation is funnelled back
        onto the main thread via ``call_soon_threadsafe``-style helpers (here:
        direct calls, since Rich's live display is thread-tolerant and the
        heavy lifting happens on the turn thread as before).
        """
        peer = self._event_peer
        while self._events_running:
            try:
                message = peer.recv(timeout=0.1)
            except Exception:
                break
            if message is None:
                continue
            try:
                self._handle_event(message)
            except Exception:
                # A rendering error must never take down the event pump.
                pass

    def _handle_event(self, message) -> None:
        """Dispatch a single protocol event to the UI."""
        if message.name == Event.STAGE:
            payload = payload_as(StagePayload, message)
            if payload.name == "prompt":
                if payload.stage == StageKind.START:
                    self._prompt_start()
                elif payload.stage == StageKind.STOP:
                    self._prompt_stop()

        elif message.name == Event.REASONING:
            payload = payload_as(ReasoningPayload, message)
            if payload.stage == StageKind.START:
                self._reasoning_start(payload.visible)
            elif payload.stage == StageKind.PROCESS:
                if payload.text and payload.visible:
                    print(f"[weak]{escape(payload.text)}[/]", end="")
            elif payload.stage == StageKind.STOP:
                self._reasoning_stop()

        elif message.name == Event.CONTENT:
            self.content_callback(payload_as(ContentPayload, message).text)

        elif message.name == Event.TOOL_CALL:
            payload = payload_as(ToolCallPayload, message)
            self.tool_callback(payload.name, payload.arguments)

        elif message.name == Event.TOOL_RESULT:
            payload = payload_as(ToolResultPayload, message)
            self.tool_result_callback(
                payload.id, payload.name, payload.content,
                payload.is_error, payload.duration,
            )

        elif message.name == Event.CANCELLED:
            print("[warn]⏹  Turn cancelled by user  ⏹[/warn]")

        # TURN_START / TURN_END / STATUS are consumed by the turn runner
        # (run_interactive), which reads the TurnResult directly in phase 1.

    # ── stage handlers (extracted from the old callbacks) ───────────────────

    def _spinner(self, text: str):
        """Start a status spinner on the active output adapter, if supported."""
        console = getattr(self.output, "_console", None)
        return console.status(text) if console else None

    def _prompt_start(self) -> None:
        self.spinner_prompt = self._spinner("⏳ Processing prompt...")
        if self.spinner_prompt:
            self.spinner_prompt.start()

    def _prompt_stop(self) -> None:
        if self.spinner_prompt:
            self.spinner_prompt.stop()
            self.spinner_prompt = None
        ok("⏳ Prompt processed")

    def _reasoning_start(self, visible: bool) -> None:
        if visible:
            info("💡 Thinking...\n")
        else:
            self.spinner_thinking = self._spinner("💡 Thinking...")
            if self.spinner_thinking:
                self.spinner_thinking.start()

    def _reasoning_stop(self) -> None:
        if self.spinner_thinking:
            self.spinner_thinking.stop()
            self.spinner_thinking = None
        ok("💡 Done thinking\n")

    def content_callback(self, content:str=""):
        """Called when new chunks arrive in streaming mode."""
        print(escape(content), end="")

    def tool_callback(self, tool_name: str, tool_args):
        newline()
        args_str = format_tool_args(tool_args)
        if args_str:
            info(f"🛠️ [weak]Activating tool:[/weak]  [tool]{tool_name}[/tool]  [weak]({escape(args_str)})[/weak]")
        else:
            info(f"🛠️ [weak]Activating tool:[/weak]  [tool]{tool_name}[/tool]")

    def tool_result_callback(self, tool_id, tool_name, content, is_error, duration):
        """Called after a tool finishes (event: TOOL_RESULT)."""
        if is_error:
            self.output.err(f"Tool {tool_name} failed: {content}")
        else:
            self.output.ok(f"Tool {tool_name} finished in {duration:.1f}s")

    def _statusline(self, total_tokens, ntools, total_gen_time):
        length, max, rate = self.core.memory.get_chat_stats()
        title = f"  {total_gen_time:.1f}s   |   {total_tokens} tokens   |   {ntools} tools   |   Mem: {length}/{max} ({rate:.2f}%)  "
        self.output.rule(title=title, style="status")

    def _cancel_all_spinners(self):
        if self.spinner_prompt:
            self.spinner_prompt.stop()
            self.spinner_prompt = None
        if self.spinner_thinking:
            self.spinner_thinking.stop()
            self.spinner_thinking = None
        
    def _create_prompt_session(self):
        # Key bindings:
        kb = KeyBindings()
        @kb.add('enter')
        def _(event):
            """Enter submits the input."""
            event.current_buffer.validate_and_handle()
        @kb.add('escape', 'enter')
        def _(event):
            """Alt+Enter inserts a newline."""
            event.current_buffer.insert_text('\n')
        @kb.add('c-c')
        def _(event):
            """Control+C: first press clears input, second press (within 1s) quits."""
            buffer = event.current_buffer
            now = time.time()

            if buffer.text:
                # First press with text: clear the buffer
                buffer.reset()
                self._last_ctrl_c_time = int(now)
            else:
                # Buffer is empty: check for double-tap
                if now - self._last_ctrl_c_time < 1.0:
                    # Double Control+C: quit
                    raise KeyboardInterrupt
                else:
                    # Single press on empty: just reset and record time
                    buffer.reset()
                    self._last_ctrl_c_time = int(now)
        def _handle_paste(text):
            """Intercept large pastes and save them to a file."""
            if len(text) > PASTE_THRESHOLD:
                file_path = self.core.memory.create_pasted_file(text)
                return f"*Pasted file: {file_path}*\n"
            return text
        # Bracketed paste: catches middle-click, Shift+Insert, and
        # Control+Shift+V
        @kb.add(Keys.BracketedPaste)
        def _(event):
            """Bracketed paste: intercept large pastes from any paste method."""
            text = event.data
            event.current_buffer.insert_text(_handle_paste(text))

        # Create prompt session now
        style = Style.from_dict({
            "prompt": "#0087d7",
            "frame.border": "#0087d7",
            "bottom-toolbar": "#ffffff bg:#262626 noreverse",
            "kbd": "#ffd787 bold",
            "model": "#005faf",
            "weak": "#393939",
            "unsafe-warn": "bold bg:#8b0000 #ffffff"
        })

        # Vi mode
        vi_mode = self.core.config.get("agent.vi_mode", False)

        # Build slash command dict
        commands = [cmd.name for cmd in registry.list_commands()]
        commands_dict = {}
        for command in commands:
            add_command(commands_dict, command)
        commands_dict = collapse_none_dicts(commands_dict)
        slash_completer = NestedCompleter.from_nested_dict(commands_dict)

        # Path auto completer (for file system paths)
        path_completer = PathCompleter()

        # Hybrid completer: detects if the current word looks like a path and uses
        # PathCompleter, otherwise falls back to slash commands completer.
        from prompt_toolkit.document import Document

        class HybridCompleter(Completer):
            def __init__(self, slash_comp, path_comp):
                self.slash_completer = slash_comp
                self.path_completer = path_comp

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor

                # Slash commands: always check first when line starts with '/'
                if text.startswith('/'):
                    cmds = list(self.slash_completer.get_completions(document, complete_event))
                    if cmds:
                        return cmds
                    # If no slash completions match, fall through to path check below

                # Extract the last word/token being typed (strip leading '/' for path detection)
                words = text.rsplit(None, 1)
                last_word = words[-1] if words else ""
                # Remove leading slash so "/embed ~/Doc" doesn't trigger path on "/embed"
                last_word_stripped = last_word.lstrip('/')

                # If the last word looks like a path, use PathCompleter.
                # We create a synthetic Document containing only the path portion,
                # because PathCompleter checks the full text and fails when there's
                # non-path text (like "/embed ") before the cursor.
                if last_word_stripped and ('/' in last_word_stripped or last_word_stripped.startswith('~') or last_word_stripped.startswith('.')):
                    fake_doc = Document(
                        text=last_word_stripped,
                        cursor_position=len(last_word_stripped),
                    )
                    path_completions = list(self.path_completer.get_completions(fake_doc, complete_event))
                    if path_completions:
                        offset = len(text) - len(last_word_stripped)
                        for c in path_completions:
                            c.start_position += offset
                        return path_completions

                # If the whole line starts with a path prefix (no command), use PathCompleter
                if text.startswith('~') or text.startswith('.') or text.startswith('..'):
                    return self.path_completer.get_completions(document, complete_event)

                # Fall back to path completer
                return self.path_completer.get_completions(document, complete_event)

        completer = HybridCompleter(slash_completer, path_completer)

        # History path
        history_path = self.core.memory.session_dir / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Toolbar
        def prompt_toolbar():
            return HTML("  <kbd>Alt</kbd>+<kbd>↵</kbd>: new line | <kbd>↵</kbd>: submit | <kbd>!</kbd>: shell command | <kbd>Ctrl</kbd>+<kbd>C</kbd>: clear / double-tap to quit")

        model = self.core.config.get("model.name")
        unsafe = self.core.config.get("agent.unsafe", False)
        unsafe_warn = (
            HTML('  <unsafe-warn>⚠ UNSAFE MODE</unsafe-warn>')
            if unsafe else ""
        )
        self._session = PromptSession(
                    style=style,
                    message=HTML(f"⩥ You ⩤   <weak>model:</weak> <model>{model}</model>  <weak>session:</weak> <model>{self.core.memory.session}</model>{unsafe_warn}\n❯ "),
                    history=FileHistory(str(history_path)),
                    show_frame=True,
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

        
    def run_interactive(self):
        """Run the agent in interactive mode."""

        startup_info(self.core, self.output)

        if _HAS_PROMPT_TOOLKIT:
            self._create_prompt_session()
            def get_input(): return str(self._session.prompt()).strip()
        else:
            # Rich
            def get_input(): return Prompt.ask(prompt="[user]⩥ [bold]You[/bold] ⩤[/user]\n❯", console=self.output._console)

        # Phase 1 (client/server): the turn is driven through the event
        # emitter. The core's callbacks are wired to the emitter, whose events
        # are rendered by the event pump (self._event_loop). Cancellation is
        # state: Ctrl+C calls emitter.cancel(), which poll() surfaces to the
        # core between streamed chunks. The TurnResult reports the outcome.
        self.emitter.reset()
        self.start_events()

        # Main loop
        while True:
            try:
                user_input = get_input()
            except (EOFError, KeyboardInterrupt):
                self.output.print(txt_goodbye)
                break

            if not user_input:
                continue

            # Ctrl+C while a turn is running: cancel the turn (state, not an
            # exception — the core observes it via poll() between chunks).
            if self._turn_in_progress:
                self.emitter.cancel("user")
                continue

            # Process user-invoked shell commands (`!` prefix)
            if user_input.startswith("!"):
                from agent.shellcmd import append_to_memory, run_shell_command

                command = user_input[1:].strip()
                if command:
                    result = run_shell_command(command, self.output)
                    append_to_memory(self.core, command, result)
                    self.output.newline()
                else:
                    self.output.err("Empty shell command")
                continue

            # Process slash commands
            if user_input.startswith("/"):
                tokens = user_input.split()
                command, params = registry.lookup(tokens)

                if command:
                    no_errors, msg, content, md, should_exit = registry.execute(self.core, command, params, self.output)

                    if should_exit:
                        self.output.print(txt_goodbye)
                        break

                    if no_errors:
                        # Content in rich or Markdown format
                        if content or md:
                            if params:
                                param_list = ' '.join(params)
                            else:
                                param_list = ''

                            if content:
                                cont = content
                            elif md:
                                cont = Markdown(md)
                            self.output.print_rich(Panel(cont,
                                                        border_style="output-frame",
                                                        title=f"{command.name} {param_list}",
                                                        subtitle=f"{command.name} {param_list}",
                                                        highlight=True))

                        # Short status message
                        if msg:
                            self.output.ok(msg)
                        self.output.newline()

                    else:
                        # Error
                        if msg:
                            self.output.err(f"{msg}")

                else:
                    self.output.err(f"Command not found: {user_input}")
                    
                continue

            else:
                self.output.newline()
                self.output.rule(style="agent")
                self.output.print(f"[agent]⩥ [bold]Wisemonkey[/bold] ⩤ [/agent]  [accent]⇒ {self.core.config.get('model.name')}[/accent]")
                self.output.print("  [kbd]Ctrl[/kbd]+[kbd]C[/kbd]: Cancel turn\n")
                try:
                    self._turn_in_progress = True
                    result = self.core.run_turn(
                        user_input,
                        self.emitter.prompt,
                        self.emitter.reasoning,
                        self.emitter.content,
                        self.emitter.tool_call,
                        self.emitter.cancelled,
                        self.emitter.error,
                        tool_result_callback=self.emitter.tool_result,
                        poll=self.emitter.poll,
                    )
                    response = result.response
                    total_tokens = result.total_tokens
                    ntools = result.n_tools
                    total_gen_time = result.gen_time
                    self.output.newline()

                    self.output.newline()
                    if result.cancelled:
                        continue  # skip status line, go straight back to prompt

                    self._statusline(total_tokens, ntools, total_gen_time)
                    self.output.newline()
                    self.output.newline()

                    if self.core.config.get("agent.markdown", False):
                        # Print markdown
                        md = self.core.memory.get_chat_unformatted()[-1]['content']
                        md = Panel(Markdown(md),
                                    border_style="output-frame",
                                    title=f"Markdown",
                                    subtitle=f"Markdown",
                                    highlight=True)
                        self.output.print_rich(md)
                        self._statusline(total_tokens, ntools, total_gen_time)
                except Exception as e:
                    self._cancel_all_spinners()
                    self.emitter.cancel("error")
                    self.output.err(f"Error sending prompt: {e}")
                    # The turn's partial conversation has already been persisted
                    # by core.run_turn(), so we just continue to the next prompt.
                    self.output.print("  [dim]Partial response was saved to chat history.[/dim]")
                finally:
                    self._turn_in_progress = False
                    self._cancel_all_spinners()

        # Persist memory, stop the event pump, and shut down core on exit
        if self.core:
            self.stop_events()
            self.core.save_memory()
            self.core.shutdown()
