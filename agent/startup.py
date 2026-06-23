import os
from datetime import datetime
from pathlib import Path

from rich import box
from rich.align import Align
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from agent.utils import contractuser, pretty_timedelta


class StartupOutput:
    """Abstract base for startup-output adapters.

    Implementations wrap console printing (CLI agent) or RichLog writing (TUI).
    """

    def print(self, text: str) -> None:
        """Print plain or markup text."""

    def print_rich(self, renderable) -> None:
        """Print a Rich renderable."""

    def newline(self) -> None:
        """Print a blank line."""

    def rule(self, style: str = "dim", title: str = "") -> None:
        """Print a horizontal rule line."""

    def info(self, text: str) -> None:
        """Print an info line with the \u21d2 prefix."""


class ConsoleStartupOutput(StartupOutput):
    """Startup output adapter backed by agent.console (CLI agent)."""

    def print(self, text: str) -> None:
        from agent.console import print as cprint
        cprint(text)

    def print_rich(self, renderable) -> None:
        from agent.console import console
        console.print(renderable)

    def newline(self) -> None:
        from agent.console import newline as nl
        nl()

    def rule(self, style: str = "dim", title: str = "") -> None:
        from agent.console import console
        console.rule(style=style)

    def info(self, text: str) -> None:
        from agent.console import info as cinfo
        cinfo(text)


def check_updates(repo_dir):
    """Check for updates"""
    from agent.update import UpdatesManager
    um = UpdatesManager()
    return um.check_updates(repo_dir)


def startup_info(core, output: StartupOutput):
    """Print startup information.

    Delegates all output to the *output* adapter so the same code works
    in both the terminal agent (console) and the TUI (RichLog).
    """

    import shutil
    from importlib.metadata import version as _ver, PackageNotFoundError

    term_size = shutil.get_terminal_size((80, 20))
    # ASCII monkey: Modified from "Monkey Typing" by Joan G. Stark (Spunk)
    # https://www.asciiart.eu/animals/monkeys
    monkee = r'''
                               .-"-.
                             _/.-.-.\_
                            ( ( o o ) )
                             |/  "  \|
                              \  ⏝  /
                              /`"""`\
                             /       \
    '''
    if term_size.columns < 80:
        wisemonkey = "WISEMONKEY"
    else:
        # ASCII title generated with https://patorjk.com/software/taag/
        wisemonkey = '''
                                                                    
██     ██ ██ ▄█████ ██████ ██▄  ▄██ ▄████▄ ███  ██ ██ ▄█▀ ██████ ██  ██ 
██ ▄█▄ ██ ██ ▀▀▀▄▄▄ ██▄▄   ██ ▀▀ ██ ██  ██ ██ ▀▄██ ████   ██▄▄    ▀██▀  
 ▀██▀██▀  ██ █████▀ ██▄▄▄▄ ██    ██ ▀████▀ ██   ██ ██ ▀█▄ ██▄▄▄▄   ██   
        '''
    title = Align.center(f"[title]{monkee}{wisemonkey}[/title]", vertical='middle')
    output.print_rich(Panel(title,
                            box=box.HEAVY,
                            border_style="title",
                            subtitle="Monkee at your service!"))
    output.newline()

    session_dir = core.memory.session_dir
    working_dir = contractuser(Path(os.getcwd()))
    created = core.memory.session_created
    accessed = core.memory.session_accessed
    output.rule()

    # Build version string
    try:
        pkg_version = _ver("wisemonkey")
    except PackageNotFoundError:
        pkg_version = "0.0.0-dev"

    now = datetime.now()
    agent_dir = Path(__file__).resolve().parent
    repo_dir = agent_dir.parent

    updates_available, commit_hash, last_check = check_updates(repo_dir)
    d_check = pretty_timedelta(now - last_check) if last_check else "never"

    version_str = f"[accent]Wisemonkey[/accent] [dim]v{pkg_version}[/dim]"
    if commit_hash:
        version_str += f"  [dim]commit: {commit_hash}[/dim]"
    output.print(version_str)
    if updates_available:
        output.print(f"   [warn]↳ Updates available![/warn] [time](last check: {d_check})[/time]")
        output.print("     [weak]run [accent]wisemonkey -u[/accent] to update[/weak]")
    elif commit_hash:
        output.print(f"   [dim]✓ Up to date[/dim] [time](last check: {d_check})[/time]")

    output.newline()

    # Session info
    new_session = core.memory.session_is_new
    d_created = pretty_timedelta(now - created) if created else "?"
    d_accessed = pretty_timedelta(now - accessed) if accessed else "?"
    if new_session:
        output.info(f"Session created: [accent-bold]'{core.memory.session}'[/accent-bold]")
    else:
        output.info(f"Session restored: [accent-bold]'{core.memory.session}'[/accent-bold]")
    output.print(f"[dim]   location:      {contractuser(session_dir)}[/dim]")
    output.print(f"[dim]   working dir:   {working_dir}[/dim]")
    output.print(f"[dim]   created:[/dim]       [time]{d_created}[/time]")
    if not new_session:
        output.print(f"[dim]   last accessed:[/dim] [time]{d_accessed}[/time]")
    output.rule()

    # Chat history
    chat_history = core.memory.get_chat_formatted(num_exchanges=3,
                                                  timestamps=True,
                                                  width=250)
    if chat_history:
        curr, max_sz, rate = core.memory.get_chat_stats()
        output.print_rich(Panel(Markdown(chat_history),
                                border_style="output-frame",
                                title="Previous conversation (last 3 exchanges, truncated)",
                                subtitle=f"Previous conversation stats: {curr}/{max_sz} - {rate:.2f}%"))

    output.newline()
    output.rule()
    output.info("[weak]Type [accent]/configure[/accent] to configure the agent interactively[/weak]")
    output.info("[weak]Type [accent]/help[/accent] for command information[/weak]")
    output.rule()
