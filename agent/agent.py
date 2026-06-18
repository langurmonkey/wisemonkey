"""The agent loop of Wisemonkey.

The agent orchestrates the 'user-assistant' turns and delegates the actual turn
handling to the core.
"""

import time
from pathlib import Path
from functools import partial

from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from pubsub import pub

from agent.core import Core, Stage, TurnCancelled
from agent.commands import registry
from agent.utils import add_command, collapse_none_dicts
from agent.console import print, err, ok, info, newline, console
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
        pub.subscribe(self._create_prompt_session, "prompt-update")

    def prompt_callback(self, stage:Stage):
        """Called when starting and ending prompt processing for a given turn"""
        match stage.value:
            case Stage.START.value:
                self.spinner_prompt = console.status("⏳ Processing prompt...")
                self.spinner_prompt.start()

            case Stage.STOP.value:
                if self.spinner_prompt:
                    self.spinner_prompt.stop()
                    self.spinner_prompt = None
                ok("⏳ Prompt processed")

            case _:
                raise RuntimeError(f"Prompt callback only has Start and Stop stages: {stage}")

    def reasoning_callback(self, stage:Stage, content:str="", reasoning_visible:bool=True):
        """Called when starting, processing, and ending the reasoning stage."""
        match stage.value:
            case Stage.START.value:
                if reasoning_visible:
                    info("💡 Thinking...\n")
                else:
                    self.spinner_thinking = console.status("💡 Thinking...")
                    self.spinner_thinking.start()
                
            case Stage.PROCESS.value:
                if reasoning_visible:
                    print(f"[weak]{escape(content)}[/]", end="")

            case Stage.STOP.value:
                if self.spinner_thinking:
                    self.spinner_thinking.stop()
                    self.spinner_thinking = None
                ok("💡 Done thinking\n")

    def content_callback(self, content:str=""):
        """Called when new chunks arrive in streaming mode."""
        print(escape(content), end="")

    def tool_callback(self, tool_name: str, tool_args):
        newline()
        info(f"🛠️ [weak]Activating tool:[/weak]  [tool]{tool_name}[/tool]")

    def cancel_callback(self, e: KeyboardInterrupt):
        """Handles the Control+c during inference, as a keyboard interrupt"""
        print("[warn]⏹  Turn cancelled by user  ⏹[/warn]")
        raise TurnCancelled() from e

    def error_callback(self, e, msg):
        raise RuntimeError(msg) from e


    def _statusline(self, total_tokens, ntools, total_gen_time):
        length, max, rate = self.core.memory.get_chat_stats()
        title = f"  {total_gen_time:.1f}s   |   {total_tokens} tokens   |   {ntools} tools   |   Mem: {length}/{max} ({rate:.2f}%)  "
        console.rule(title=title, style="status")

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
            "prompt": "ansiyellow",
            "frame.border": "ansiyellow",
            "bottom-toolbar": "#ffffff bg:#262626 noreverse",
            "kbd": "#ffd787 bold",
            "model": "#0087d7"
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
            return HTML("  <kbd>Alt</kbd>+<kbd>↵</kbd>: new line | <kbd>↵</kbd>: submit | <kbd>Ctrl</kbd>+<kbd>C</kbd>: clear / double-tap to quit")

        model = self.core.config.get("model.name")
        self._session = PromptSession(
                    style=style,
                    message=HTML(f"⩥ You ⩤ <model>⇒ {model}</model>\n❯ "),
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

        startup_info(self.core)

        if _HAS_PROMPT_TOOLKIT:
            self._create_prompt_session()
            def get_input(): return str(self._session.prompt()).strip()
        else:
            # Rich
            def get_input(): return Prompt.ask(prompt="[user]⩥ [bold]You[/bold] ⩤[/user]\n❯", console=console)

        # Wrap each callback to pass self
        prompt_cb = partial(self.prompt_callback)
        reasoning_cb = partial(self.reasoning_callback)
        content_cb = partial(self.content_callback)
        tool_cb = partial(self.tool_callback)
        cancel_cb = partial(self.cancel_callback)
        error_cb = partial(self.error_callback)

        # Main loop
        while True:
            try:
                user_input = get_input()
            except (EOFError, KeyboardInterrupt):
                print(txt_goodbye)
                break

            if not user_input:
                continue

            # Process slash commands
            if user_input.startswith("/"):
                tokens = user_input.split()
                command, params = registry.lookup(tokens)

                if command:
                    no_errors, msg, content, md, should_exit = registry.execute(self.core, command, params)

                    if should_exit:
                        print(txt_goodbye)
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
                            print(Panel(cont,
                                        border_style="output-frame",
                                        title=f"{command.name} {param_list}",
                                        subtitle=f"{command.name} {param_list}",
                                        highlight=True))

                        # Short status message
                        if msg:
                            ok(msg)
                        newline()

                    else:
                        # Error
                        if msg:
                            err(f"{msg}")

                else:
                    err(f"Command not found: {user_input}")
                    
                continue

            else:
                newline()
                console.rule(style="agent")
                print(f"[agent]⩥ [bold]Wisemonkey[/bold] ⩤ [/agent]  [accent]⇒ {self.core.config.get('model.name')}[/accent]")
                print("  [kbd]Ctrl[/kbd]+[kbd]C[/kbd]: Cancel turn\n")
                try:
                    (response,
                        total_tokens,
                        ntools,
                        total_gen_time) = self.core.run_turn(
                                                          user_input,
                                                          prompt_cb,
                                                          reasoning_cb,
                                                          content_cb,
                                                          tool_cb,
                                                          cancel_cb,
                                                          error_cb
                                                      )
                    newline()
                    newline()
                    if response == "[Cancelled]":
                        continue  # skip status line, go straight back to prompt
                    self._statusline(total_tokens, ntools, total_gen_time)
                    newline()
                except Exception as e:
                    self._cancel_all_spinners()
                    err(f"Error sending prompt: {e}")
                    # The turn's partial conversation has already been persisted
                    # by core.run_turn(), so we just continue to the next prompt.
                    print("  [dim]Partial response was saved to chat history.[/dim]")
                    

        # Persist memory and shut down core on session exit
        if self.core:
            self.core.save_memory()
            self.core.shutdown()
