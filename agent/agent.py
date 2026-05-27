"""The agent loop.

The agent orchestrates the 'user-assistant' turns and delegates the actual turn
handling to the core.
"""

from pathlib import Path
from xdg_base_dirs import xdg_data_home
from functools import partial
from textwrap import shorten

from rich import box
from rich.markup import escape
from rich.prompt import Prompt
from rich.align import Align
from rich.markdown import Markdown
from rich.panel import Panel

from agent.core import Core, Stage, TurnCancelled
from agent.commands import registry
from agent.console import console


# Try to import prompt_toolkit for rich input; fall back to plain input.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import FuzzyWordCompleter
    from prompt_toolkit.clipboard import InMemoryClipboard
    from prompt_toolkit.formatted_text import HTML
    _HAS_PROMPT_TOOLKIT = True

except ImportError:
    console.print("[red]ERROR:[/red] could not initialize prompt toolkit")
    _HAS_PROMPT_TOOLKIT = False


txt_goodbye = "\n[bold blue]Goodbye![/bold blue]"


class Agent:
    def __init__(self, config_path=None):
        self.core = Core(config_path)
        self.spinner_prompt = None
        self.spinner_thinking = None

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
                console.print("[green]✓[/] ⏳ Prompt processed")

            case _:
                raise RuntimeError(f"Prompt callback only has Start and Stop stages: {stage}")

    def reasoning_callback(self, stage:Stage, content:str=None, show_thinking:bool=True):
        """Called when starting, processing, and ending the reasoning stage."""
        match stage.value:
            case Stage.START.value:
                if show_thinking:
                    console.print("[orange1]⇨[/] 💡 Thinking...")
                else:
                    self.spinner_thinking = console.status("💡 Thinking...")
                    self.spinner_thinking.start()
                
            case Stage.PROCESS.value:
                if show_thinking:
                    console.print(f"[grey39]{content}[/]", end="")

            case Stage.STOP.value:
                if self.spinner_thinking:
                    self.spinner_thinking.stop()
                    self.spinner_thinking = None
                console.print("[green]✓[/] 💡 Done thinking")

    def content_callback(self, content: str = None):
        """Called when new chunks arrive in streaming mode."""
        console.print(content, end="")

    def tool_callback(self, tool_name: str, tool_args):
        console.print(f"[black on #66aa99] ⚙ Activating tool: {tool_name} [/black on #66aa99]")

    def cancel_callback(self, e: KeyboardInterrupt):
        """Handles the Ctrl+c during inference, as a keyboard interrupt"""
        console.print("\n[bold yellow]⏹  Turn cancelled[/bold yellow]")
        raise TurnCancelled() from e

    def error_callback(self, e, msg):
        raise RuntimeError(msg) from e


    def _statusline(self, total_tokens, ntools, total_gen_time):
        len, max, rate =self.core.memory.get_chat_stats()
        console.print(f"[black on #777777]   {total_gen_time:.1f}s  ∣  {total_tokens} tokens  |  {ntools} tools  |  Mem: {len}/{max} ({rate:.2f}%)    [/black on #777777]", justify="full")
        console.print()

        
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

        # Create prompt session now
        style = Style.from_dict({
            "prompt": "ansiyellow",
            "frame.border": "ansiyellow",
        })

        # Vi mode
        vi_mode = self.core.config.get("agent.vi_mode", False)

        # Slash commands autocompleter
        commands = [cmd.name for cmd in registry.list_commands()]
        slash_completer = FuzzyWordCompleter(commands)

        # History path
        history_path = xdg_data_home() / "langur-agent" / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Toolbar
        prompt_toolbar = lambda : HTML(" <b>Alt</b>+<b>Enter</b>: new line | <b>Enter</b>: submit prompt | <b>Ctrl</b>+<b>C</b>: quit")

        return PromptSession(
                    style=style,
                    message="⩥ You ⩤\n❯ ",
                    history=FileHistory(str(history_path)),
                    show_frame=True,
                    multiline=True,
                    key_bindings=kb,
                    vi_mode=vi_mode,
                    clipboard=InMemoryClipboard(),
                    enable_open_in_editor=vi_mode,
                    complete_while_typing=True,        
                    complete_in_thread=True,
                    completer=slash_completer,
                    auto_suggest=AutoSuggestFromHistory(),
                    bottom_toolbar=prompt_toolbar,
        )
        
    def run_interactive(self):
        """Run the agent in interactive mode."""
        import shutil
        term_size = shutil.get_terminal_size((80, 20))
        if term_size.columns < 80:
            languragent="LANGUR AGENT"
        else:
            languragent = '''
██      ▄▄▄  ▄▄  ▄▄  ▄▄▄▄ ▄▄ ▄▄ ▄▄▄▄    ▄████▄  ▄▄▄▄ ▄▄▄▄▄ ▄▄  ▄▄ ▄▄▄▄▄▄
██     ██▀██ ███▄██ ██ ▄▄ ██ ██ ██▄█▄   ██▄▄██ ██ ▄▄ ██▄▄  ███▄██   ██  
██████ ██▀██ ██ ▀██ ▀███▀ ▀███▀ ██ ██   ██  ██ ▀███▀ ██▄▄▄ ██ ▀██   ██  
            '''
        title = Align.center(f"[bold blue]{languragent}[/bold blue]", vertical='middle')
        console.print(Panel(title, box=box.HEAVY, border_style="blue", subtitle="Monkee at your service!"))
        console.print()

        # Print history
        history = self.core.memory.get_chat_unformatted()
        if history:
            lines = []
            for turn in history:
                content = shorten(f"{escape(turn['content'])}", width=550)
                lines.append(f"## {turn['role'].capitalize()}:")
                lines.append(f"{content}")

            curr, max, rate = self.core.memory.get_chat_stats()
            console.print(Panel(Markdown('\n'.join(lines)),
                            border_style="magenta",
                            title="Previous conversation",
                            subtitle=f"Previous conversation stats: {curr}/{max} - {rate:.2f}%"))

        console.print()

        # Get help
        console.print("[gray39]Type [bold green]/help[/bold green] for command information[/gray39]")
        if _HAS_PROMPT_TOOLKIT:
            self._session = self._create_prompt_session()
            # Prompt Toolkit
            style = Style.from_dict({
                "prompt": "ansiyellow",
            })
            get_input = lambda: str(
                self._session.prompt()
            ).strip()
        else:
            # Rich
            get_input = lambda: Prompt.ask(prompt="[yellow]⩥ You ⩤[/yellow]\n❯",
                                           console=console)

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
                console.print(txt_goodbye)
                break

            if not user_input:
                continue

            # Process slash commands
            if user_input.startswith("/"):
                tokens = user_input.split()
                command, params = registry.lookup(tokens)

                if command:
                    ok, msg, content, md, should_exit = registry.execute(self, command, params)
                    if should_exit:
                        console.print(txt_goodbye)
                        break
                    if ok:
                        # Content in rich or Markdown format
                        if content or md:
                            if params:
                                param_list = ' '.join(params)
                            else:
                                params_list = ''

                            cont = content if content else Markdown(md)
                            console.print(Panel(cont,
                                            border_style="magenta",
                                            title=f"{command.name} {param_list}",
                                            subtitle=f"{command.name} {param_list}"))

                        # Short status message
                        if msg:
                            console.print(f"[green]OK[/green]: {msg}")
                        console.print()
                    else:
                        if msg:
                            console.print(f"[red]ERROR[/red]: {msg}")
                else:
                    console.print(f"[red]ERROR:[/red] command not found: {user_input}")
                    
                continue

            else:
                console.print(f"\n[magenta]⩥ Langur Agent ⩤ [/magenta]  ⦗[blue]{self.core.config.get('model.name')}[/blue]⦘")
                console.print("  [dim][bold]Ctrl[/bold]+[bold]C[/bold]: Cancel turn[/dim]\n")
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
                console.print()
                if response == "[Cancelled]":
                    continue  # skip status line, go straight back to prompt
                self._statusline(total_tokens, ntools, total_gen_time)

        # Persist memory on session exit
        self.core.save_memory()
