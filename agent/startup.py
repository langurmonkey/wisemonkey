import os
from datetime import datetime
from pathlib import Path

from rich import box
from rich.align import Align
from rich.markdown import Markdown
from rich.panel import Panel

from agent.memory import CONVERSATIONAL_ROLES
from agent.output import OutputAdapter
from agent.utils import contractuser, pretty_timedelta

# How much of the previous conversation to recap at startup. One turn is the
# last question and its final answer, which is enough to pick up where we left
# off; more turns fills the panel faster than it helps.
RECAP_EXCHANGES = 1
# Truncation widths for the recap. Assistant answers get the wider one: a
# question is usually one line, so capping it tightly only throws away the
# context that made the answer make sense.
RECAP_WIDTH = 320
RECAP_ASSISTANT_WIDTH = 400


def check_updates(repo_dir):
    """Check for updates"""
    from agent.update import UpdatesManager

    um = UpdatesManager()
    return um.check_updates(repo_dir)


def startup_info(core, output: OutputAdapter):
    """Print startup information.

    Delegates all output to the *output* adapter so the same code works
    in both the terminal agent (console) and the TUI (RichLog).
    """

    import shutil
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _ver

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
        wisemonkey = """
                                                                    
██     ██ ██ ▄█████ ██████ ██▄  ▄██ ▄████▄ ███  ██ ██ ▄█▀ ██████ ██  ██ 
██ ▄█▄ ██ ██ ▀▀▀▄▄▄ ██▄▄   ██ ▀▀ ██ ██  ██ ██ ▀▄██ ████   ██▄▄    ▀██▀  
 ▀██▀██▀  ██ █████▀ ██▄▄▄▄ ██    ██ ▀████▀ ██   ██ ██ ▀█▄ ██▄▄▄▄   ██   
        """
    if core.config.get("agent.banner", False):
        title = Align.center(f"[title]{monkee}{wisemonkey}[/title]", vertical="middle")
        output.print_rich(
            Panel(
                title,
                box=box.HEAVY,
                border_style="title",
                subtitle="Monkee at your service!",
            )
        )
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

    now = datetime.now().astimezone()
    agent_dir = Path(__file__).resolve().parent
    repo_dir = agent_dir.parent

    updates_available, commit_hash, last_check = check_updates(repo_dir)
    d_check = pretty_timedelta(now - last_check) if last_check else "never"

    version_str = f"[accent]Wisemonkey[/accent] [dim]v{pkg_version}[/dim]"
    if commit_hash:
        version_str += f"  [dim]commit: {commit_hash}[/dim]"
    output.print(version_str)
    if updates_available:
        output.print(
            f"   [warn]↳ Updates available![/warn] [time](last check: {d_check})[/time]"
        )
        output.print("     [weak]run [accent]wmk -u[/accent] to update[/weak]")
    elif commit_hash:
        output.print(f"   [dim]✓ Up to date[/dim] [time](last check: {d_check})[/time]")

    output.newline()

    # Session info
    new_session = core.memory.session_is_new
    # Both timestamps are written naive by `Memory` and parsed straight back
    # from `.session-metadata`, so they are normalized here: `now` is
    # timezone-aware, and subtracting a naive datetime from it raises. The
    # two lines used to disagree, which crashed on every restored session.
    d_created = pretty_timedelta(now - created.astimezone()) if created else "?"
    d_accessed = pretty_timedelta(now - accessed.astimezone()) if accessed else "?"
    # Context usage: total tokens of the full prompt (same as /context)
    # as a percentage of the configured budget.
    sections = core.get_context_breakdown()
    ctx_total = sum(tokens for _, tokens in sections)
    ctx_max = core.config.get("agent.max_chat_history", 80000)
    ctx_rate = (ctx_total / ctx_max * 100) if ctx_max else 0.0
    if new_session:
        output.info(
            f"Session created: [accent-bold]'{core.memory.session}'[/accent-bold]"
        )
    else:
        output.info(
            f"Session restored: [accent-bold]'{core.memory.session}'[/accent-bold]"
        )
    output.print(f"[dim]   location:       {contractuser(session_dir)}[/dim]")
    output.print(f"[dim]   working dir:    {working_dir}[/dim]")
    output.print(f"[dim]   created:[/dim]        [time]{d_created}[/time]")
    if not new_session:
        output.print(f"[dim]   last accessed:[/dim]  [time]{d_accessed}[/time]")
    output.print(
        f"[dim]   context:        {ctx_total} tokens ({ctx_rate:.2f}% of {ctx_max} budget)[/dim]"
    )
    output.rule()

    # Chat history
    #
    # This is a recap for the *human*, not a prompt for the model, so it drops
    # tool calls and results: tool output is the least readable thing in the
    # panel and it crowds out the conversation, which is the point of a recap.
    #
    # It also keeps one answer per turn. A turn that ran several tools emitted
    # an assistant message per narration step, so counting entries rather than
    # turns filled the panel with Assistant blocks and pushed the user's
    # question off the end -- the same noise as the tool results, one level
    # up. `conversational_turns` groups by turn and keeps the question plus the
    # final answer, so RECAP_EXCHANGES is a number of Q/A pairs.
    #
    # Assistant answers get a wider budget than user questions: a question is
    # usually one line, so a shared width only throws away the context that
    # made the answer make sense. `mark_incomplete` appends "(continued)" when
    # the final answer was cut off.
    recap_roles = CONVERSATIONAL_ROLES
    shown = core.memory.conversational_count(RECAP_EXCHANGES)
    chat_history = core.memory.get_chat_history_formatted(
        num_exchanges=RECAP_EXCHANGES,
        timestamps=False,
        width=RECAP_WIDTH,
        roles=recap_roles,
        assistant_width=RECAP_ASSISTANT_WIDTH,
        mark_incomplete=True,
    )
    if chat_history:
        noun = "exchange" if shown == 1 else "exchanges"
        output.print_rich(
            Panel(
                Markdown(chat_history),
                border_style="output-frame",
                title=f"Previous conversation (last {shown} {noun}, truncated)",
                subtitle=f"Previous conversation stats: {ctx_total} tks - {ctx_rate:.2f}% of {ctx_max} budget",
            )
        )

    output.newline()
    output.rule()
    output.info(" [accent]/[/accent] [weak]slash commands[/weak]")
    output.info(
        " [accent]![/accent] [weak]shell command, attach output to context[/weak]"
    )
    output.info(
        " [accent]@[/accent] [weak]reference file/directory, attach to context[/weak]"
    )
    output.info(" [accent]?[/accent] [weak]show help[/weak]")
    output.info(
        " [accent]/configure[/accent]  [weak]configure the agent interactively[/weak]"
    )
    output.rule()
