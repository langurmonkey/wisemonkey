"""
Centralized slash command registry.

Contains all slash commands and the scaffolding necessary to register
and execute them.
"""

from __future__ import annotations

import ast
import datetime
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pubsub import pub

from agent.output import OutputAdapter, get_output_or_ipc
from agent.tokens import count_tokens
from agent.utils import resize_image

# Global error messages for commands
no_params_error = "This command does not take any parameters"


def smart_cast(value, target_type):
    """Cast a string value to a given type, handling booleans correctly."""
    if target_type is bool:
        return ast.literal_eval(value)  # 'False' → False
    return target_type(value)  # '42' → 42, '3.14' → float, etc.


def empty():
    pass


@dataclass(frozen=True)
class Command:
    """A single slash command definition."""

    name: str
    description: str = ""
    handler: Callable = (
        empty  # (agent, params: list[str], output: OutputAdapter | None) -> str | None
    )
    aliases: list[str] = field(default_factory=list)
    examples: list[tuple[str, str]] = field(default_factory=list)


class CommandRegistry:
    """Module-level singleton registry for slash commands."""

    def __init__(self):
        self._commands: dict[str, Command] = {}  # primary name -> Command

    def register(self, cmd: Command) -> None:
        """Register a command and all its aliases."""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd  # alias points to same Command

    def lookup(self, tokens: list[str]) -> tuple[Command | None, list[str] | None]:
        """
        Look up a command by name or alias (case-insensitive).

        Returns:
        - command: Command or None     - command instance
        - tokens: list[str] or None    - command arguments as a token list
        -
        """
        if not tokens:
            return None, None

        n = len(tokens)
        for words in reversed(range(1, n + 1)):
            name = "-".join(tokens[i] for i in range(words))
            command = self._commands.get(name.lower())
            if command:
                return command, tokens[words:]

        return None, None

    def execute(
        self,
        core,
        cmd: Command,
        params: list[str] | None,
        output: OutputAdapter | None = None,
    ) -> tuple[bool, str | None, str | None, str | None, bool]:
        """
        Execute a command.

        Parameters
        ----------
        core : Core
            The agent core instance.
        cmd : Command
            The command to execute.
        params : list[str] | None
            Command arguments.
        output : OutputAdapter | None
            UI abstraction for interactive prompts.  When *None*, commands
            that need user input will fall back to the classic rich.prompt
            behaviour.

        Returns:
        - ok: bool          - status of the operation
        - msg: str          - short message with an informative message
        - content: str      - long text in python rich format
        - markdown: str     - long text in Markdown format
        - should_exit: bool - boolean indicating whether the agent must exit
        """
        if params and (params[0].lower() == "-h" or params[0].lower() == "help"):
            return True, None, self._command_str(cmd), None, False

        ok, msg, content, markdown = cmd.handler(core, params, output)
        should_exit = msg in ("EXIT", "exit")
        return ok, msg, content, markdown, should_exit

    def run_command(
        self, core, command: str, output: OutputAdapter | None = None
    ) -> tuple[bool, str | None, str | None, str | None, bool]:
        """
        Shortcut to run a command from a string.

        Returns:
        - ok: bool          - status of the operation
        - msg: str          - short message with an informative message
        - content: str      - long text in python rich format
        - markdown: str     - long text in Markdown format
        - should_exit: bool - boolean indicating whether the agent must exit
        """
        cmd, params = self.lookup(command.split())
        if cmd:
            return self.execute(core, cmd, params, output)
        else:
            raise RuntimeError(f"command not found: {command}")

    def list_commands(self) -> list[Command]:
        """
        Return all unique commands (deduplicated by primary name) and
        in alphabetical order.
        """
        import collections

        seen = set()
        result = []
        commands = collections.OrderedDict(sorted(self._commands.items()))
        for cmd in commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd)
        return result

    def names(self) -> list[str]:
        """Return all command names (primary + aliases) for completion."""
        return list(self._commands.keys())

    def get_commands_str(self):
        """Prints all defined commands to the output"""
        all = ""
        for cmd in self.list_commands():
            all += f"{self._command_str(cmd)}"

        return all

    def _command_str(self, cmd):
        aliases = ", ".join(f"[green]{a}[/]" for a in cmd.aliases)
        primary = f"[green]{cmd.name.replace('-', ' ')}[/]"
        if aliases:
            names = f"{primary}, {aliases}"
        else:
            names = primary
        result = ""
        result += f"• {names} → {cmd.description}\n"

        if cmd.examples:
            result += "[output-frame]  Examples:[/output-frame]\n"
            # Descriptions line up in a column: the command list is short and
            # reading it with the descriptions ragged would be worse than a
            # little arithmetic here.
            width = max(len(example) for example, _ in cmd.examples)
            for example, description in cmd.examples:
                if description:
                    result += (
                        f"[time]    {example:<{width}}[/time]"
                        f"  [weak]{description}[/weak]\n"
                    )
                else:
                    result += f"[time]    {example}[/time]\n"

        return result


# Module-level singleton
registry = CommandRegistry()


# Decorator for commands
def cmd(
    name: str,
    description: str = "",
    aliases: list[str] = [],
    examples: list[tuple[str, str]] = [],
):
    """
    Decorator to register commands.

    Commands return:
    - ok:bool      - False if there was an error
    - msg:str      - A one-line status message
    - content:str  - Multi-line content in rich formatting
    - markdown:str - Multi-line content in markdown
    """

    def command(handler):
        registry.register(Command(name, description, handler, aliases, examples))

    return command


def _fallback_output() -> OutputAdapter:
    """Return a RichOutputAdapter for when no output is explicitly provided."""
    from agent.output import RichOutputAdapter

    return RichOutputAdapter()


# Built-in command handlers
@cmd("/quit", "Exit the agent", aliases=["/exit", "/q"])
def _cmd_quit(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    return True, "EXIT", None, None


@cmd(
    "/reasoning",
    "Configure model reasoning",
)
def _cmd_reasoning(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:

    if params:
        return False, no_params_error, None, None

    ui = output or _fallback_output()
    from agent.config import get_config

    config = get_config()

    try:
        # Reasoning effort
        opts = [
            ("max", "Max"),
            ("xhigh", "Extra high"),
            ("high", "High"),
            ("medium", "Medium"),
            ("low", "Low"),
            ("minimal", "Minimal"),
            ("none", "Disable model reasoning"),
        ]
        defa = core.router.thinking_effort

        effort = ui.ask_choice(
            message="Choose the reasoning effort:",
            options=opts,
            default=defa,
        )
        core.router.thinking_effort = effort

        # Reasoning display
        opts = [("true", "Yes"), ("false", "No")]
        defa = config.get("model.thinking.display")

        visible = ui.ask_choice(
            message="Display reasoning:",
            options=opts,
            default=str(defa).lower(),
        )
        visible_bool = visible == "true"
        config.set("model.thinking.display", visible_bool)
    except Exception as e:
        ui.err(str(e))

    return True, f"reasoning effort: {effort}, show reasoning: {visible}", None, None


@cmd(
    "/notes",
    "List all notes",
)
def _cmd_notes(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    notes = core.memory.get_notes()
    buff = ""
    if notes:
        for note in notes:
            buff += f"📋️  [blue]{note['id']}[/blue] ({note['category']}):\n"
            buff += f"[weak]{note['content']}[/weak]\n\n"
        return True, None, buff, None
    else:
        return False, "no notes found", None, None


@cmd(
    "/notes-add",
    "Add a note to memory",
    examples=[
        ("/notes add This is my note", "Add a new note"),
    ],
)
def _cmd_notes_add(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        core.memory.add_note(" ".join(params))
        return True, "note added successfully", None, None

    return False, "please, provide a note", None, None


@cmd(
    "/sessions",
    "List all sessions with their size and last access",
    examples=[
        ("/sessions", "List every session, most recently used first"),
    ],
)
def _cmd_sessions(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.memory import list_sessions
    from agent.utils import contractuser, pretty_timedelta

    sessions = list_sessions()
    if not sessions:
        return False, "no sessions found", None, None

    current = getattr(core.memory, "session", None)

    # Fixed-width columns keep the list readable without a rich Table: the
    # result is rendered inside a Panel as markup text, not as a renderable,
    # and it travels over IPC as a string in both local and remote mode.
    name_width = max(6, max(len(s["name"]) for s in sessions))
    header = (
        f"{'':2} {'Name':<{name_width}}  {'Messages':>8}  {'Last access':<14}  Location"
    )
    lines = [f"[dim]{header}[/dim]"]

    for info in sessions:
        marker = "[accent]▸[/accent]" if info["name"] == current else " "
        if info["name"] == current:
            name_cell = f"[accent-bold]{info['name']:<{name_width}}[/accent-bold]"
        else:
            name_cell = f"{info['name']:<{name_width}}"
        accessed = info["accessed"]
        when = "unknown"
        if accessed is not None:
            # `pretty_timedelta` subtracts a naive `datetime.now()`, so it
            # cannot take an aware datetime. Compute the delta here instead:
            # metadata timestamps are written naive, hence the normalize.
            if accessed.tzinfo is None:
                accessed = accessed.astimezone()
            ago = pretty_timedelta(
                datetime.datetime.now().astimezone() - accessed
            )
            # `pretty_timedelta` returns None for a zero delta (its final
            # branch is a string literal, not a return), so "just now" has to
            # be spelled out here rather than crash on the f-string.
            when = f"{ago} ago" if ago else "just now"
        lines.append(
            f"{marker} {name_cell}  {info['messages']:>8}  {when:<14}  "
            f"[dim]{contractuser(info['dir'])}[/dim]"
        )

    lines.append("")
    lines.append(f"[dim]{len(sessions)} session(s). Switch with [/dim][accent]/resume <name>[/accent]")
    return True, None, "\n".join(lines), None


@cmd(
    "/resume",
    "Switch to another session, keeping this process running",
    examples=[
        ("/resume", "Resume the most recently used other session"),
        ("/resume gaiasky", "Switch to the session named 'gaiasky'"),
    ],
)
def _cmd_resume(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    from agent.memory import is_valid_session_name, list_sessions

    mem = core.memory

    # `registry.execute` intercepts `-h`/`help` before the handler, so this
    # only triggers if the command is invoked another way. Guard anyway:
    # the alternative is silently creating a session directory named "-h".
    if params and params[0].lower() in ("-h", "help"):
        return False, "unknown session name: -h (this is the help flag)", None, None

    if len(params) > 1:
        return False, "too many parameters: /resume takes at most one session name", None, None

    sessions = list_sessions()
    current = getattr(mem, "session", None)

    if params:
        name = params[0]
    else:
        # No argument: the most recently used session that is not this one.
        # Falling back to the current session is right -- "resume" with
        # nothing to switch to should be a no-op, not an error.
        others = [s for s in sessions if s["name"] != current]
        if not others:
            return False, "no other session to resume", None, None
        name = others[0]["name"]

    if not is_valid_session_name(name):
        return False, f"invalid session name: {name!r}", None, None

    if name == current:
        return True, f"already in session '{name}'", None, None

    known = {s["name"] for s in sessions}
    if name not in known:
        # Not an error: `wmk <new-name>` creates sessions, so /resume should
        # too. Say so explicitly, since it is a silent new directory.
        previous = mem.switch_session(name)
        result = (
            f"Switched from session [accent-bold]{previous}[/accent-bold] to "
            f"[accent-bold]{name}[/accent-bold] (new session).\n"
            f"[dim]Switching rebinds chat history, notes and the user profile; "
            f"the current session was saved first.[/dim]"
        )
        pub.sendMessage("prompt-update")
        return True, None, result, None

    previous = mem.switch_session(name)
    if hasattr(core, "reset_session_state"):
        core.reset_session_state()
    result = (
        f"Switched from session [accent-bold]{previous}[/accent-bold] to "
        f"[accent-bold]{name}[/accent-bold].\n"
        f"[dim]Loaded {len(mem.get_chat_history_unformatted())} message(s) from "
        f"{len(mem.get_notes())} note(s).[/dim]"
    )
    # The prompt message, the footer status line and the history file all
    # name the session, so the prompt has to be rebuilt.
    pub.sendMessage("prompt-update")
    return True, None, result, None


@cmd(
    "/session",
    "Print session information",
)
def _cmd_session(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.utils import contractuser

    mem = core.memory
    name = mem.session
    session_dir = mem.session_dir
    working_dir = contractuser(Path(os.getcwd()))
    created = mem.session_created
    accessed = mem.session_accessed
    # Context usage: total tokens of the full prompt (same as /context)
    # as a percentage of the configured budget.
    sections = core.get_context_breakdown()
    total = sum(tokens for _, tokens in sections)
    max_tokens = core.config.get("agent.max_chat_history", 80000)
    rate = (total / max_tokens * 100) if max_tokens else 0.0
    result = ""
    result += f"Name:           [accent-bold]{name}[/accent-bold]\n"
    result += f"Location:       {contractuser(session_dir)}\n"
    result += f"Working dir:    {working_dir}\n"
    result += f"Created:        {created}\n"
    result += f"Last accessed:  {accessed}\n"
    result += f"Context:        [accent-bold]{total}[/accent-bold] tokens ({rate:.2f}% of {max_tokens} budget)"
    return True, None, result, None


@cmd(
    "/session-agent",
    "Show the session agent memory contents (user profile and notes)",
)
def _cmd_session_agent(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    mem = core.memory.get_formatted()
    if mem:
        # Format in Markdown
        return True, None, None, mem
    else:
        return False, "agent memory is empty", None, None


@cmd(
    "/history",
    "Show the session chat history contents, one turn per interaction",
    examples=[
        ("/history", "Print the entire session chat history"),
        ("/history 2", "Print the last 2 turns (prompt + answer)"),
    ],
)
def _cmd_history(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Print the last *n* turns of the chat history (0 = all).

    The count is a number of *turns* -- a prompt plus everything up to the
    next prompt -- like ``/history-compact`` and ``/history-info``, not a
    number of stored entries: an entry count lands in the middle of a turn,
    so asking for three of them can show two tool calls and an answer with no
    question in sight. The recap roles are what keeps the machinery out of the
    panel: tool calls and results carry nothing a human wants to re-read, and
    the turn is summarized by its final answer.
    """
    from agent.memory import CONVERSATIONAL_ROLES

    n = 0
    if params:
        try:
            n = int(params[0])
        except ValueError:
            return False, f"Parameter must be integer: {params[0]}", None, None

    mem = core.memory.get_chat_history_formatted(
        num_exchanges=n, timestamps=True, roles=CONVERSATIONAL_ROLES
    )
    tokens, max, rate = core.memory.get_chat_stats()
    stats = f"Memory status: {tokens}/{max} tokens ({rate:.2f}%)"
    # Format in Markdown
    return True, stats, None, mem


def _short_utc(when: str) -> str:
    """Shorten an entry's ISO-ish ``utc`` stamp to ``YYYY-MM-DD HH:MM``.

    History entries store ``str(datetime)`` (e.g.
    ``2026-06-26 12:00:00.123456+00:00``), whose microseconds and offset are
    noise when the question is only "when did this interaction happen". An
    un-parseable or missing stamp is passed through as-is rather than dropped,
    so an entry never looks older than it is just because its timestamp was
    in an unexpected shape.
    """
    if not when:
        return "—"
    try:
        return datetime.datetime.fromisoformat(when).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return when


@cmd(
    "/history-info",
    "Get info on the chat history interactions and each one's share of the history",
    examples=[
        (
            "/history-info",
            "Number the interactions and their percentage of the history size",
        ),
    ],
)
def _cmd_history_info(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Summarise how the chat history is split across interactions.

    Each interaction is a user message plus everything up to the next one
    (answer, tool calls and results), and its size is the tokens it
    contributes to the context. Percentages are shares of the live history,
    so they add up to 100% -- which interaction is eating the window.

    Entries are numbered oldest first and stamped with the moment each
    interaction began, so the numbering cannot be mistaken for "most recent
    first" (the two directions are indistinguishable from the sizes alone).
    """
    if params:
        return False, no_params_error, None, None

    info = core.memory.get_chat_history_info()
    if not info:
        return False, "agent memory is empty", None, None

    count = len(info)
    total = sum(size for _, size in info)
    tokens, max_tokens, rate = core.memory.get_chat_stats()

    stamps = [_short_utc(when) for when, _ in info]
    stamp_width = max(len(s) for s in stamps)
    number_width = len(str(count))
    lines = []
    for i, ((_, size), stamp) in enumerate(zip(info, stamps), start=1):
        pct = (size * 100.0 / total) if total else 0.0
        lines.append(
            f"{i:>{number_width}}. {stamp:<{stamp_width}}  "
            f"{pct:6.2f}%  ({size:,} tokens)"
        )

    plural = "" if count == 1 else "s"
    msg = (
        f"{count} interaction{plural} in chat history, oldest first "
        f"({tokens}/{max_tokens} tokens, {rate:.2f}% of the window)"
    )
    return True, msg, "\n".join(lines), None


@cmd(
    "/history-clear",
    "Clear the current session chat history; gets an optional integer with the number of exchanges to show",
    examples=[
        ("/history", "Clear chat memory for this session"),
        ("/history clear 10", "Clear the 10 oldest chat exchanges of this session"),
    ],
)
def _cmd_history_clear(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    n = 0
    if params:
        try:
            n = int(params[0])
        except ValueError:
            return False, f"the parameter must be an integer: '{params[0]}'", None, None

    cleared = core.memory.clear_chat(n)
    return True, f"{cleared} exchanges cleared", None, None


@cmd(
    "/undo",
    "Remove the last exchange(s) from the context; optional integer count (default 1)",
    examples=[
        ("/undo", "Drop the last exchange (prompt + answer + tool calls)"),
        ("/undo 3", "Drop the last three exchanges"),
    ],
)
def _cmd_undo(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Drop whole exchanges from the tail of the chat history.

    The counterpart of ``/history-clear``, which removes them from the *front*
    and knows nothing about turns. One bad answer otherwise poisons the rest
    of the session: the model keeps re-reading it and keeps building on it.
    """
    n = 1
    if params:
        try:
            n = int(params[0])
        except ValueError:
            return False, f"the parameter must be an integer: '{params[0]}'", None, None

    mem = core.memory
    entries = len(mem.get_chat_history_unformatted())
    if not entries:
        return False, "there is nothing to undo", None, None

    removed = mem.drop_last_exchanges(n)
    if not removed:
        return False, "there is nothing to undo", None, None
    return True, f"{removed} entries removed ({n} exchange(s))", None, None


@cmd(
    "/retry",
    "Re-run the last prompt from scratch, optionally dropping its answer from the context",
    aliases=["/r"],
    examples=[
        ("/retry", "Re-run the last prompt, keeping the old answer in context"),
        ("/retry --drop", "Re-run it and remove the old exchange first"),
    ],
)
def _cmd_retry(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Re-run the previous prompt.

    By default the old exchange stays in the context, which is occasionally
    what you want ("now do it properly"). ``--drop`` removes it first, so the
    re-run sees the same context the original did -- that is the mode that
    actually undoes a bad turn.
    """
    drop = any(p in ("--drop", "-d") for p in params)
    if any(not (p in ("--drop", "-d")) for p in params):
        return False, f"unknown option: {' '.join(params)}", None, None

    prompt = core.memory.last_user_prompt()
    if not prompt:
        return False, "no previous prompt in this session", None, None

    if drop:
        removed = core.memory.drop_last_exchanges(1)
        core.pending_prompt = prompt
        return True, f"re-running ({removed} entries dropped)", None, None

    core.pending_prompt = prompt
    return True, "re-running the last prompt", None, None


@cmd(
    "/edit",
    "Edit the last prompt in $EDITOR and re-run it, dropping its answer",
    examples=[
        ("/edit", "Open the last prompt in $EDITOR, then re-run it"),
    ],
)
def _cmd_edit(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Amend the last prompt and re-run it as the same turn.

    The edited prompt *replaces* the old exchange rather than being appended
    after it: keeping both would leave the model reading the wrong version of
    the question it is answering.
    """
    if params:
        return False, no_params_error, None, None

    prompt = core.memory.last_user_prompt()
    if not prompt:
        return False, "no previous prompt in this session", None, None

    edited = _edit_in_editor(prompt, output)
    if edited is None:
        return False, "edit cancelled", None, None
    edited = edited.strip()
    if not edited:
        return False, "the edited prompt is empty; nothing to re-run", None, None
    if edited == prompt:
        return False, "the prompt is unchanged; nothing to re-run", None, None

    removed = core.memory.drop_last_exchanges(1)
    core.pending_prompt = edited
    return True, f"re-running the edited prompt ({removed} entries dropped)", None, None


def _edit_in_editor(text: str, output: OutputAdapter | None) -> str | None:
    """Open *text* in ``$EDITOR`` and return the edited content.

    Returns None when the editor could not be run or exited non-zero, which
    is what an abort (``Esc`` in vim, empty buffer) looks like. The scratch
    file is deleted afterwards either way.
    """
    import os
    import subprocess
    import tempfile

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"

    # .md so syntax highlighting and spell-checking see a markdown document.
    fd, path = tempfile.mkstemp(suffix=".md", prefix="wisemonkey-prompt-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")

        cmd = [editor, path]
        if output is not None:
            result = output.run_subprocess(cmd)
        else:
            result = subprocess.run(cmd)

        if getattr(result, "returncode", 1) != 0:
            return None
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@cmd(
    "/last-prompt",
    "Print the most recent user prompt (used by /retry and /edit)",
)
def _cmd_last_prompt(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None
    text = core.memory.last_user_prompt()
    if not text:
        return False, "no previous prompt in this session", None, None
    return True, None, text, None


@cmd(
    "/history-compact",
    "Compact the session chat history by summarizing it into a shorter form",
    examples=[
        ("/history-compact", "Summarize every interaction but the most recent one"),
        ("/history-compact 5", "Summarize only the 5 oldest interactions"),
        ("/history-compact 0", "Summarize the entire chat history"),
    ],
)
def _cmd_history_compact(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Summarize part of the chat history in place.

    With no argument every interaction except the most recent one is summarized
    and replaced by a single briefing, so the task being worked on right now
    stays in the live context instead of being buried in the summary. With a
    count, only the oldest *n* interactions are summarized and replaced while
    everything more recent is left untouched -- the way to shed the expensive
    early part of a long session without losing what is being worked on. The
    count is capped at the number of interactions present, so an over-large
    value is *not* an error, it just means "all"; ``0`` summarizes everything
    (the latest interaction included).
    """
    if params:
        try:
            num_exchanges = int(params[0])
        except ValueError:
            return False, f"Parameter must be integer: {params[0]}", None, None
        if num_exchanges < 0:
            return False, "the number of interactions must be positive", None, None
    else:
        # No argument: keep the latest interaction out of the summarized slice.
        # ``num_exchanges <= 0`` means "all" in the memory layer, so the count
        # is derived from the history instead of passed as 0 -- otherwise a
        # single-interaction history would summarize the very exchange meant to
        # survive.
        total = len(core.memory.get_chat_history_info())
        if total == 0:
            return False, "agent memory is empty", None, None
        if total == 1:
            return (
                False,
                "nothing to compact: only the latest interaction is present",
                None,
                None,
            )
        num_exchanges = total - 1

    history_text = core.memory.get_chat_history_formatted_oldest(num_exchanges)
    if not history_text:
        return False, "agent memory is empty", None, None

    chars_before = len(history_text)
    tks_before = count_tokens(history_text)

    content = (
        "You are a technical editor. Summarize and compact this conversation as a dense agent briefing. Follow these guidelines:\n"
        "- Include what was being worked on\n"
        "- Remove greetings, small talk, pleasantries, repeated interactions, and filler words\n"
        "- Remember files created or modified and how\n"
        "- Preserve file names, technical constraints, and logical reasoning\n"
        "- Add facts worth remembering long term\n"
        "- Output format is markdown, use bullet points if needed\n"
        "CONVERSATION:\n"
    )
    content += history_text

    messages = [{"role": "user", "content": content}]

    output = get_output_or_ipc()
    output.print("⏳ Compacting chat history...")
    try:
        response = core.llm_chat_raw(messages)

        summary = response.choices[0].message.content
        chars_after = len(summary)
        tks_after = count_tokens(summary)

        total_exchanges = len(core.memory.get_chat_history_info())
        covered = (
            total_exchanges
            if num_exchanges <= 0
            else min(num_exchanges, total_exchanges)
        )
        core.memory.compact_chat_history(summary, num_exchanges)

        kept = total_exchanges - covered
        kept_note = (
            f", {kept} newer interaction{'s' if kept != 1 else ''} kept"
            if kept > 0
            else ""
        )
        plural = "" if covered == 1 else "s"
        return (
            True,
            f"Memory compacted successfully from {chars_before} to {chars_after} chars "
            f"({tks_before} -> {tks_after} tks): {covered} interaction{plural} "
            f"summarized{kept_note}",
            None,
            None,
        )
    except Exception as e:
        return False, f"Memory compact operation failed: {e}", None, None


@cmd(
    "/plan-read",
    "Read a saved session plan by name",
    examples=[
        ("/plan read NEW_FEATURE", "Read a specific plan"),
        ("/plan read REFACTOR", "Read the most recent matching plan"),
    ],
)
def _cmd_plan_read(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if not params:
        return (
            False,
            "provide a plan name (e.g., REFACTOR or REFACTOR_20260626_120000)",
            None,
            None,
        )

    name = " ".join(params).strip().upper().replace(" ", "_")
    from agent.memory import Memory

    mem = Memory()
    plans_dir = mem.session_dir / "plans"
    if not plans_dir.exists():
        return False, "no plans directory found", None, None

    # Try exact match first
    exact = plans_dir / f"{name}.md"
    target = None
    if exact.exists():
        target = exact
    else:
        matches = sorted(plans_dir.glob(f"{name}_*.md"), reverse=True)
        if matches:
            target = matches[0]

    if not target:
        return False, f"no plan found matching '{name}'", None, None

    content = target.read_text(encoding="utf-8")
    return True, f"Plan: {target.name}", content, content


@cmd(
    "/plan-list",
    "List all saved session plans in the current session",
    aliases=["/plan-ls", "/plans-list", "/plans-ls"],
)
def _cmd_plan_list(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.memory import Memory

    mem = Memory()
    plans_dir = mem.session_dir / "plans"
    if not plans_dir.exists():
        return False, "no plans directory found", None, None

    import datetime
    import re

    files = sorted(plans_dir.glob("*.md"), reverse=True)
    if not files:
        return False, "no session plans found", None, None

    results = ""
    for f in files:
        raw = f.read_text(encoding="utf-8")
        # Parse frontmatter for status
        meta = {}
        match = re.match(r"^---\s*\n(.*?)\n---", raw, re.DOTALL)
        if match:
            for line in match.group(1).strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
        name_display = f.stem
        status = meta.get("status", "unknown")
        mod = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        from agent.utils import pretty_timedelta

        age = pretty_timedelta(mod)
        results += f"  \u2022 [bold]{name_display}[/bold] \u2014 {status} ({age} ago)\n"

    return True, f"{len(files)} plan(s) found", results, None


@cmd(
    "/plan-edit",
    "Edit a saved session plan with $EDITOR or $VISUAL",
    examples=[
        ("/plan edit REFACTOR_20260626_120000", "Edit a specific plan"),
        ("/plan edit REFACTOR", "Edit the most recent matching plan"),
    ],
)
def _cmd_plan_edit(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if not params:
        return (
            False,
            "provide a plan name (e.g., REFACTOR or REFACTOR_20260626_120000)",
            None,
            None,
        )

    name = " ".join(params).strip().upper().replace(" ", "_")
    from agent.memory import Memory

    mem = Memory()
    plans_dir = mem.session_dir / "plans"
    if not plans_dir.exists():
        return False, "no plans directory found", None, None

    # Try exact match first
    exact = plans_dir / f"{name}.md"
    target = None
    if exact.exists():
        target = exact
    else:
        matches = sorted(plans_dir.glob(f"{name}_*.md"), reverse=True)
        if matches:
            target = matches[0]

    if not target:
        return False, f"no plan found matching '{name}'", None, None

    import subprocess
    import os

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    cmd = [editor, str(target)]
    if output is not None:
        result = output.run_subprocess(cmd)
    else:
        result = subprocess.run(cmd)

    ok = result.returncode == 0
    if ok:
        return True, f"Plan '{target.name}' edited successfully", None, None
    else:
        return False, f"Editor exited with code {result.returncode}", None, None


@cmd(
    "/embed",
    "Embed a document into the session vector store",
    examples=[
        ("/embed ~/documents/research_paper.pdf", ""),
        ("/embed ./notes.md", ""),
    ],
)
def _cmd_embed(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if not params:
        return False, "please provide a file path", None, None

    file_path = " ".join(params)
    output = get_output_or_ipc()
    output.print(f"⏳ Embedding: {file_path}...")
    file_path = os.path.expanduser(file_path)

    # Lazily initialize the vector store on first use
    if core.memory.vectorstore is None:
        from agent.memory import _load_vectorstore

        core.memory.vectorstore = _load_vectorstore(core.memory.session_dir)

    if core.memory.vectorstore is None:
        return (
            False,
            "Vector store is not available. Check that chromadb and tiktoken are installed, and embedding config is correct.",
            None,
            None,
        )

    try:
        count = core.memory.vectorstore.ingest(file_path)
        return (
            True,
            f"Successfully embedded {count} chunks from '{file_path}'",
            None,
            None,
        )
    except Exception as e:
        return False, f"Embedding failed: {e}", None, None


@cmd(
    "/tools",
    "List all available tools ⚙",
)
def _cmd_tools(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str

    return True, None, get_tools_str(), None


@cmd(
    "/tools-native",
    "List native tools ⚙",
)
def _cmd_tools_native(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str

    buff = get_tools_str(prefix="mcp_", contains=False)
    if not buff:
        return False, "no native tools found", None, None
    return True, None, buff, None


@cmd(
    "/tools-mcp",
    "List MCP tools ⚙",
)
def _cmd_tools_mcp(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str

    buff = get_tools_str(prefix="mcp_", contains=True)
    if not buff:
        return False, "no MCP tools found", None, None
    return True, None, buff, None


@cmd(
    "/skills",
    "List loaded skills ⚔",
)
def _cmd_skills(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    buff = core.skills.get_skills_str()
    if not buff:
        return False, "no skills found", None, None
    return True, None, buff, None


# Sentinel values for the endpoint and API-key pickers: "type something else"
# is a first-class option of the list it belongs to, not a separate question
# asked after it.
_NEW_ENDPOINT = "__new_endpoint__"
_NEW_KEY = "__new_key__"
_NEW_KEY_NAME = "__new_key_name__"


def _default_key_var(provider: str) -> str:
    """Fallback variable name for a key that is typed in.

    Namespaced by provider, so two endpoints of the same provider do not
    silently share one variable.
    """
    return f"WM_API_KEY_{(provider or 'generic').upper()}"


def _endpoints(config) -> list[dict]:
    """Normalise ``model.endpoints`` into a list of endpoint dicts.

    Each entry is either a bare URL string or a mapping with ``name``,
    ``base_url`` (``url`` is accepted too), ``provider`` and ``api_key_env``.
    Malformed entries are dropped rather than raising: one typo must not make
    ``/model`` unusable.
    """
    raw = config.get("model.endpoints") or []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []

    entries: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            item = {"base_url": item}
        if not isinstance(item, dict):
            continue
        url = str(item.get("base_url") or item.get("url") or "").strip()
        if not url:
            continue
        env = item.get("api_key_env")
        entries.append(
            {
                "name": str(item.get("name") or "").strip(),
                "base_url": url,
                "provider": str(item.get("provider") or "").strip().lower(),
                "api_key_env": str(env or "").strip(),
            }
        )
    return entries


def _endpoint_label(entry: dict) -> str:
    """One-line description of an endpoint entry for the picker."""
    bits = [entry["name"] or entry["base_url"]]
    if entry["name"]:
        bits.append(entry["base_url"])
    if entry["provider"]:
        bits.append(entry["provider"])
    if entry["api_key_env"]:
        bits.append(f"key {entry['api_key_env']}")
    return "  ·  ".join(bits)


def _key_variable_options(
    base_url: str,
    current_env: str,
    entry_env: str,
    suggested_env: str,
    candidates: list[str],
) -> list[tuple[str, str]]:
    """The API-key variable choices for one endpoint, most likely first.

    The variable name for an OpenAI-compatible host is arbitrary, so having to
    recall the exact one is the failure mode this list exists to remove. Order:
    auto-detect, the endpoint's own variable (from its entry), the host's
    conventional name, whatever is configured now, the resolution chain, and
    every ``*_API_KEY`` variable that is on disk or in the environment -- so a
    key one already has is visible instead of something to remember -- then the
    two ways of entering something new.
    """
    from agent.config import api_key_env_names

    options: list[tuple[str, str]] = [
        ("", "Auto-detect: " + " or ".join(candidates)),
    ]
    on_disk = set(api_key_env_names())

    def add(name: str) -> None:
        if not name or any(n == name for n, _ in options):
            return
        if os.environ.get(name, "").strip():
            mark = " (set)"
        elif name in on_disk:
            mark = " (empty)"
        else:
            mark = ""
        options.append((name, name + mark))

    for name in (entry_env, suggested_env, current_env, *candidates):
        add(name)
    for name in api_key_env_names():
        add(name)

    options.append((_NEW_KEY_NAME, "Enter a different variable name..."))
    options.append(
        (_NEW_KEY, "Enter a new API key (saved to the config .env file)")
    )
    return options


def _store_new_key(
    ui, name: str, base_url: str, provider: str
) -> tuple[str, str]:
    """Ask for a key, write it to the config ``.env``; return (var, path).

    The value goes to ``~/.config/wisemonkey/.env`` (mode 600), never to
    config.yaml: config.yaml is the file that gets shared or committed, and a
    secret does not belong in it. Returns ``("", "")`` when there is nothing to
    store, which leaves the normal resolution chain in charge.
    """
    from agent.config import set_env_var
    from agent.router import suggest_key_env

    fallback = name or suggest_key_env(base_url) or _default_key_var(provider)
    value = ui.ask_string(
        " API key (stored in the config .env file, never in config.yaml)"
    ).strip()
    if not value:
        return "", ""

    if not name:
        name = str(
            ui.ask_string(
                " Environment variable to store it in", default=fallback
            )
            or ""
        ).strip()
    if not name:
        name = fallback
    path = set_env_var(name, value)
    return name, str(path)


def _configure_endpoint(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Configure a whole endpoint in one flow.

    URL, provider, API key and model are one decision, not four: OpenRouter,
    OpenCode and a local vLLM are all ``generic`` but need three *different*
    keys, and the key belongs to the endpoint rather than to the model.

    Steps: URL -> provider -> key variable -> router init -> model. The URL step
    is a picker over ``model.endpoints`` (the URLs one actually uses, each
    remembering the key variable it wants) with "enter a new URL" as one of its
    options, and the key step likewise lets one name a new variable or paste a
    key that is then stored in the config .env. Neither list is a cage: free
    text is always one keystroke away.
    """
    ui = output or _fallback_output()
    from agent.config import get_config
    from agent.router import (
        Provider,
        _match_provider_url,
        api_key_candidates,
        suggest_key_env,
    )

    config = get_config()

    # --- 1. Endpoint URL -------------------------------------------------
    # The configured list first, free text behind one entry of it. With no list
    # configured this is exactly the old flow: ask for the URL.
    current_url = config.get("model.base_url") or ""
    entries = _endpoints(config)
    selected_entry: dict | None = None
    if entries:
        endpoint_options = [
            (str(i), _endpoint_label(entry)) for i, entry in enumerate(entries)
        ]
        endpoint_options.append((_NEW_ENDPOINT, "Enter a new URL..."))
        default_entry = next(
            (
                str(i)
                for i, entry in enumerate(entries)
                if entry["base_url"].rstrip("/") == current_url.rstrip("/")
            ),
            _NEW_ENDPOINT,
        )
        picked = ui.ask_choice(
            message=" Choose an endpoint:",
            options=endpoint_options,
            default=default_entry,
        )
        if picked not in (None, "", _NEW_ENDPOINT):
            try:
                selected_entry = entries[int(picked)]
            except (TypeError, ValueError, IndexError):
                selected_entry = None

    if selected_entry is None:
        new_url = ui.ask_string(
            " Enter the endpoint URL", default=current_url
        ).strip()
    else:
        new_url = selected_entry["base_url"]
    config.set("model.base_url", new_url)

    # --- 2. Provider -----------------------------------------------------
    # Precedence: what the entry declares, then a canonical URL (which decides
    # the provider outright), then a question. Asking about a known URL is how
    # people end up sending their Anthropic key to a generic host.
    detected = _match_provider_url(new_url)
    current_provider = config.get("model.provider", "")
    provider_opts = [
        (Provider.OPENAI.value, "OpenAI"),
        (Provider.ANTHROPIC.value, "Anthropic"),
        (Provider.OLLAMA.value, "Ollama"),
        (Provider.LMSTUDIO.value, "LM Studio"),
        (Provider.GENERIC.value, "Generic (OpenAI-compatible)"),
    ]
    known_providers = {p for p, _ in provider_opts}
    declared_provider = (selected_entry or {}).get("provider", "")

    if declared_provider in known_providers:
        selected_provider = declared_provider
    elif detected is not None and detected.value in known_providers:
        selected_provider = detected.value
    else:
        # Unknown host (OpenRouter, OpenCode, vLLM, ...): ask, defaulting to
        # the current setting or `generic`. Only the canonical URLs are in
        # _PROVIDER_URLS, so anything else is an OpenAI-compatible endpoint
        # unless the user says otherwise.
        default_provider = (
            current_provider
            if current_provider in known_providers
            else Provider.GENERIC.value
        )
        selected_provider = ui.ask_choice(
            message="Provider for this URL (OpenAI-compatible?):",
            options=provider_opts,
            default=default_provider,
        )
    config.set("model.provider", selected_provider)

    # --- 3. API key variable ---------------------------------------------
    # Asked before the router is reinitialized because the key is what the
    # router needs in order to list models at all, and because switching
    # routers is exactly the case where the right variable changes. The
    # variable *name* is stored, never the key, so config.yaml stays safe to
    # share; a key that is typed in here goes to the config .env instead.
    candidates = api_key_candidates(Provider(selected_provider))
    stored_key_path = ""
    if candidates:
        current_env = str(config.get("model.api_key_env", "") or "").strip()
        entry_env = (selected_entry or {}).get("api_key_env", "")
        suggested_env = suggest_key_env(new_url) or _default_key_var(
            selected_provider
        )
        key_options = _key_variable_options(
            base_url=new_url,
            current_env=current_env,
            entry_env=entry_env,
            suggested_env=suggested_env,
            candidates=candidates,
        )
        default_env = current_env or entry_env
        if not any(name == default_env for name, _ in key_options):
            default_env = ""

        new_env = str(
            ui.ask_choice(
                message=(
                    f" Environment variable holding the API key for "
                    f"{selected_provider}:"
                ),
                options=key_options,
                default=default_env,
            )
            or ""
        ).strip()

        if new_env == _NEW_KEY_NAME:
            new_env = str(
                ui.ask_string(
                    " Environment variable name for this key",
                    default=entry_env or suggested_env,
                )
                or ""
            ).strip()
            # A name with no value is the dead end the old flow ended in, so
            # offer to fill it in right here instead of failing after the fact.
            if new_env and not os.environ.get(new_env, "").strip():
                if ui.ask_confirm(
                    f" {new_env} is not set. Enter its value now?",
                    default=True,
                ):
                    new_env, stored_key_path = _store_new_key(
                        ui, new_env, new_url, selected_provider
                    )
        elif new_env == _NEW_KEY:
            new_env, stored_key_path = _store_new_key(
                ui, "", new_url, selected_provider
            )

        config.set("model.api_key_env", new_env)
        if new_env and not os.environ.get(new_env, "").strip():
            chain = " or ".join(candidates)
            return (
                False,
                f"{new_env} is not set in your environment or .env file. "
                f"Set it ({stored_key_path or '~/.config/wisemonkey/.env'}), "
                f"or choose auto-detect ({chain}).",
                None,
                None,
            )

    # Reinitialize router with new provider
    ok, msg = core.initialize_router()
    if not ok:
        return False, f"Failed to initialize {selected_provider}: {msg}", None, None

    # Model selection
    try:
        models = core.get_models()
    except Exception as e:
        return False, f"{e}", None, None

    if not models:
        return (
            True,
            f"Endpoint: [accent]{new_url}[/accent] · "
            f"Provider: [accent]{selected_provider}[/accent] "
            f"(no models listed)",
            None,
            None,
        )

    opts = [(m["id"], m["id"]) for m in models]
    defa = core.router.model_name

    result = ui.ask_choice(
        message="Choose a model:",
        options=opts,
        default=defa,
    )

    try:
        success = core.set_model(result)
        if success:
            pub.sendMessage("prompt-update")
            key_src = config.get("model.api_key_env", "") or "auto"
            if stored_key_path:
                key_src += f" (saved to {stored_key_path})"
            return (
                True,
                f"Endpoint: [accent]{new_url}[/accent] · "
                f"Provider: [accent]{selected_provider}[/accent] · "
                f"Key: [accent]{key_src}[/accent] · "
                f"Model: [accent]{result}[/accent]",
                None,
                None,
            )
        else:
            return False, "Model could not be set", None, None
    except NameError as e:
        return False, f"{e}", None, None


@cmd(
    "/model",
    "Configure the endpoint: URL, provider, API key variable and model",
    aliases=["/models", "/connect"],
)
def _cmd_models(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Entry point for the merged endpoint flow; see :func:`_configure_endpoint`."""
    return _configure_endpoint(core, params, output)


@cmd(
    "/url",
    "Configure the endpoint (alias of /model: URL, provider, key, model)",
)
def _cmd_url(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    """Alias of ``/model``.

    URL, provider, API key and model are one decision: an OpenAI-compatible
    host needs its key chosen alongside the URL, not as a separate step, so
    ``/url`` runs the same flow rather than setting the URL and leaving the
    endpoint half-configured.

    Delegates to :func:`_configure_endpoint` rather than to ``/model``, because
    the ``@cmd`` decorator registers a command and returns ``None`` -- there is
    no usable function under that name.
    """
    return _configure_endpoint(core, params, output)


@cmd(
    "/config-show",
    "Show current configuration",
)
def _cmd_config_show(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import log_config

    return True, None, log_config(), None


@cmd(
    "/config-edit",
    "Edit the configuration file with $EDITOR or $VISUAL",
)
def _cmd_config_edit(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import edit_base_config_visual

    result = edit_base_config_visual(output)
    if result.returncode != 0:
        return False, result.stderr or "Configuration editor failed", None, None

    ok, message = core.reload_config()
    if ok:
        # Refresh frontend settings (e.g. vi mode and prompt metadata).
        from pubsub import pub

        pub.sendMessage("prompt-update")
        return True, "Configuration edited and reloaded successfully", None, None
    return False, f"Configuration reload failed: {message}", None, None


@cmd(
    "/config",
    "Configure the agent interactively",
    aliases=["/configure"],
)
def _cmd_config(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    # Endpoint (URL, provider, key, model), then, reasoning, temperature, vi
    commands = ["/model", "/reasoning", "/temperature", "/vi", "/markdown"]

    for command in commands:
        ok, msg, _, _, _ = registry.run_command(core, command, output)
        if not ok:
            return False, msg, None, None

    return True, "Configuration updated", None, None


@cmd(
    "/mcp-edit",
    "Edit the mcp.json configuration file with $EDITOR or $VISUAL",
)
def _cmd_mcp_edit(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import edit_mcp_config_visual

    result = edit_mcp_config_visual(output)
    ok = result.returncode == 0
    if ok:
        core.initialize_mcp()
        return ok, "MCP configuration edited successfully", None, None
    else:
        return ok, result.stderr, None, None


@cmd(
    "/mcp",
    "Show the current MCP configuration file",
    aliases=["/mcp-show"],
)
def _cmd_mcp_show(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import get_mcp_config_path

    path = get_mcp_config_path()
    with path.open() as file:
        content = file.read()

    if content:
        return True, f"MCP configuration: {path}", content, None
    else:
        return False, "Could not load MCP configuration file", None, None


@cmd(
    "/mcp-tools",
    "List MCP tools ⚙",
)
def _cmd_mcp_tools(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str

    return True, None, get_tools_str(prefix="mcp_", contains=True), None


@cmd(
    "/temperature",
    "Set the inference temperature parameter in 0..2",
    aliases=["/temp", "/t"],
)
def _cmd_temperature(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    ui = output or _fallback_output()
    new_temp = ui.ask_float(
        " Enter the temperature [0..2]", default=core.router.temperature
    )
    if new_temp < 0 or new_temp > 2:
        return False, f"Temperature out of [0..2] range: {new_temp}", None, None

    core.router.temperature = new_temp

    return True, f"Temperature: {new_temp}", None, None


@cmd(
    "/vi",
    "Enable/disable vi input mode",
)
def _cmd_vi(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    ui = output or _fallback_output()
    from agent.config import get_config

    config = get_config()

    opts = [("true", "On"), ("false", "Off")]
    defa = config.get("agent.vi_mode")

    state = ui.ask_choice(
        message="Vi input mode:",
        options=opts,
        default=str(defa).lower(),
    )
    state_bool = state == "true"
    config.set("agent.vi_mode", state_bool)
    pub.sendMessage("prompt-update")
    return True, f"Vi mode: {state_bool}", None, None


@cmd(
    "/unsafe",
    "Enable/disable unsafe mode",
)
def _cmd_unsafe(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    ui = output or _fallback_output()
    from agent.config import get_config

    config = get_config()

    opts = [("true", "On"), ("false", "Off")]
    defa = config.get("agent.unsafe")

    state = ui.ask_choice(
        message="Unsafe mode:",
        options=opts,
        default=str(defa).lower(),
    )
    state_bool = state == "true"
    config.set("agent.unsafe", state_bool)
    pub.sendMessage("prompt-update")
    return True, f"Unsafe mode: {state_bool}", None, None


@cmd(
    "/markdown",
    "Configure markdown rendering after inference",
)
def _cmd_vi(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    ui = output or _fallback_output()
    from agent.config import get_config

    config = get_config()

    opts = [("true", "On"), ("false", "Off")]
    defa = config.get("agent.markdown")

    state = ui.ask_choice(
        message="Markdown rendering:",
        options=opts,
        default=str(defa).lower(),
    )
    state_bool = state == "true"
    config.set("agent.markdown", state_bool)
    pub.sendMessage("prompt-update")
    return True, f"Markdown rendering: {state_bool}", None, None


@cmd(
    "/attachimage",
    "Attach an image to the next prompt",
    examples=[
        ("/attachimage path/to/screenshot.png", ""),
        ("/attachimage ~/Pictures/photo.jpg", ""),
    ],
)
def _cmd_attach_image(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if not params:
        return False, "please provide an image file path", None, None

    file_path = os.path.expanduser(" ".join(params))
    path = Path(file_path)

    if not path.is_file():
        return False, f"file not found: {file_path}", None, None

    try:
        raw_bytes = path.read_bytes()
        result = resize_image(raw_bytes)
        core._pending_image = result
        return (
            True,
            f"🖼️  Image loaded and attached — will be sent with your next message. ([dim]{path.name}[/dim], {len(result['image_base64'])} bytes base64)",
            None,
            None,
        )
    except Exception as e:
        return False, f"failed to load image: {e}", None, None


@cmd(
    "/help",
    "Show command help",
    aliases=["/commands"],
)
def _cmd_help(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    return True, None, registry.get_commands_str(), None


@cmd(
    "/context",
    "Show context usage: token breakdown per section of the prompt",
)
def _cmd_context(
    core, params, output: OutputAdapter | None = None
) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    sections = core.get_context_breakdown()
    total = sum(tokens for _, tokens in sections)

    max_tokens = core.config.get("agent.max_chat_history", 80000)
    rate = (total / max_tokens * 100) if max_tokens else 0.0

    result = f"Context budget (agent.max_chat_history): [accent-bold]{max_tokens}[/accent-bold] tokens\n\n"
    for name, tokens in sections:
        pct = (tokens / total * 100) if total else 0.0
        bar_len = int(pct / 2.5)  # 40-char bar max
        bar = "█" * bar_len + "·" * (40 - bar_len)
        result += f"{name:<28} {tokens:>7}  {pct:5.1f}%  [weak]{bar}[/weak]\n"
    result += f"\n{'Total':<28} [accent-bold]{total:>7}[/accent-bold]  {100.0:5.1f}%  of budget: {rate:.2f}%"

    return True, None, result, None
