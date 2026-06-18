import os
from datetime import datetime
from pathlib import Path

from rich import box
from rich.align import Align
from rich.markdown import Markdown
from rich.panel import Panel

from agent.core import Core
from agent.utils import contractuser, pretty_timedelta
from agent.console import print, info, newline, console

def check_updates(repo_dir):
    """Check for updates"""
    from agent.update import UpdatesManager
    um = UpdatesManager()
    return um.check_updates(repo_dir)

def startup_info(core: Core):
    """ Prints startup information 

    This method prints startup information when Wisemonkey boots. It provides the user with
    information about the current session, the update status, the chat history, and more.
    """
    import shutil
    term_size = shutil.get_terminal_size((80, 20))
    monkee = '''
                               .-"-.
                             _/.-.-.\\_
                            ( ( o o ) )
                             |/  "  \\|
                              \\ ݁݁ ⏝  /
                              /`"""`\\
                             /       \\
    '''
    if term_size.columns < 80:
        wisemonkey = "WISEMONKEY"
    else:
        wisemonkey = '''
                                                                    
██     ██ ██ ▄█████ ██████ ██▄  ▄██ ▄████▄ ███  ██ ██ ▄█▀ ██████ ██  ██ 
██ ▄█▄ ██ ██ ▀▀▀▄▄▄ ██▄▄   ██ ▀▀ ██ ██  ██ ██ ▀▄██ ████   ██▄▄    ▀██▀  
 ▀██▀██▀  ██ █████▀ ██▄▄▄▄ ██    ██ ▀████▀ ██   ██ ██ ▀█▄ ██▄▄▄▄   ██   
        '''
    title = Align.center(f"[title]{monkee}{wisemonkey}[/title]", vertical='middle')
    print(Panel(title,
                box=box.HEAVY,
                border_style="title",
                subtitle="Monkee at your service!"))
    newline()

    new_session = core.memory.session_is_new
    session_dir = core.memory.session_dir
    working_dir = contractuser(Path(os.getcwd()))
    created = core.memory.session_created
    accessed = core.memory.session_accessed
    console.rule(style="weak")

    # Print build/version and update
    from importlib.metadata import version, PackageNotFoundError
    try:
        pkg_version = version("wisemonkey")
    except PackageNotFoundError:
        pkg_version = "0.0.0-dev"

    # Determine the repository directory (where this source file lives, or the installed package)
    # Prefer the source tree over the installed package location
    now = datetime.now()
    agent_dir = Path(__file__).resolve().parent
    repo_dir = agent_dir.parent

    updates_available, commit_hash, last_check = check_updates(repo_dir)
    d_check = pretty_timedelta(now - last_check)

    version_str = f"[accent]Wisemonkey[/accent] [dim]v{pkg_version}[/dim]"
    if commit_hash:
        version_str += f"  [dim]commit: {commit_hash}[/dim]"
    info(f"{version_str}")
    if updates_available:
        print(f"   [warn]⟳ Updates available![/warn] [time](last check: {d_check} ago)[/time]")
        print("     [weak]run [accent]wisemonkey -u[/accent] to update[/weak]")
    elif commit_hash:
        print(f"   [dim]✓ Up to date[/dim] [time](last check: {d_check} ago)[/time]")

    newline()

    # Print session
    d_created = pretty_timedelta(now - created) if created else None
    d_accessed = pretty_timedelta(now - accessed) if accessed else None
    if new_session:
        info(f"Session created: [accent-bold]'{core.memory.session}'[/accent-bold]")
    else:
        info(f"Session restored: [accent-bold]'{core.memory.session}'[/accent-bold]")
    print(f"[dim]   location:      {contractuser(session_dir)}[/dim]")
    print(f"[dim]   working dir:   {working_dir}[/dim]")
    print(f"[dim]   created:[/dim]       [time]{d_created} ago[/time]")
    if not new_session:
        print(f"[dim]   last accessed:[/dim] [time]{d_accessed} ago[/time]")
    console.rule(style="weak")

    # Print history
    chat_history = core.memory.get_chat_formatted(num_exchanges=3,
                                                       timestamps=True,
                                                       width=250)

    if chat_history:
        curr, max, rate = core.memory.get_chat_stats()
        print(Panel(Markdown(chat_history),
                        border_style="output-frame",
                        title="Previous conversation (last 3 exchanges, truncated)",
                        subtitle=f"Previous conversation stats: {curr}/{max} - {rate:.2f}%"))

    newline()

    # Info
    console.rule(style="weak")
    info("[weak]Type [accent]/configure[/accent] to configure the agent interactively[/weak]")
    info("[weak]Type [accent]/help[/accent] for command information[/weak]")
    console.rule(style="weak")
