"""Terminal execution tool.

Allows the agent to run shell commands on the host machine.
Usage: run_command(command="ls -la", timeout=30)

Security: commands run in the current working directory. User confirmation
is required for dangerous commands and for any command that references a
file or directory outside the session working directory.

The session is two directories, not one: the working directory, and the
session's own scratch directory (``<session_dir>/tmp``), which every command
gets as ``TMPDIR``. Scratch work therefore stays inside the session instead of
the shared ``/tmp``, and writing there is not an outside path.
"""

import os
import re
import shlex
import subprocess
from pathlib import Path

from agent.config import get_config
from agent.output import get_output_or_ipc
from agent.tools import tool

# ──────────────────────────────────────────────
# Configurable danger detection
# ──────────────────────────────────────────────

DANGEROUS_PATTERNS = [
    # Destructive file operations
    r'\brm\b',
    r'\brmdir\b',
    r'\bdd\b',
    r'\bmkfs\b',
    r'\bmke2fs\b',
    r'\bmkswap\b',
    r'\bfdisk\b',
    r'\bparted\b',
    r'\bshred\b',
    r'\bwipefs\b',
    r'\bblkdiscard\b',

    # Permission / ownership changes
    r'\bchmod\s+-R\s*777\b',
    r'\bchown\s+-R\b',
    r'\bchattr\b',

    # System-level commands
    r'\breboot\b',
    r'\bshutdown\b',
    r'\bpoweroff\b',
    r'\binit\b',
    r'\bsystemctl\s+(stop|disable|mask|kill)\b',
    r'\bkmod\b',
    r'\bmodprobe\b',

    # Package management
    r'\bapt\b',
    r'\bapt\b',
    r'\bdpkg\b',
    r'\bpacman\b',
    r'\bparu\b',
    r'\byay\b',
    r'\byum\b',
    r'\bbrew\b',

    # Process / signal
    r'\bpkill\b',
    r'\bkillall\b',
    r'\bkill\s+-9\b',

    # Network / firewall
    r'\biptables\s+-F\b',
    r'\bnft\s+flush\b',
    r'\bufw\s+(disable|reset)\b',

    # Disk / mount
    r'\bmount\b',
    r'\bumount\b',
    r'\bswapoff\b',
    r'\bswapon\b',

    # Security / credentials
    r'\bpasswd\b',
    r'\buseradd\b',
    r'\buserdel\b',
    r'\bgroupadd\b',
    r'\bgroupdel\b',
    r'\bgpasswd\b',

    # Docker destructive
    r'\bdocker\s+(rm|rmi|system\s+prune|volume\s+rm|network\s+rm)\b',

    # Git destructive (force push, branch delete, reset hard)
    r'\bgit\s+push\s+.*-f\b',
    r'\bgit\s+push\s+.*--force\b',
    r'\bgit\s+branch\s+-[dD]\b',
    r'\bgit\s+reset\s+--hard\b',

    # Curl / wget to shell pipe (classic risk)
    r'\bcurl\b.*\|?\s*bash\b',
    r'\bwget\b.*\|?\s*bash\b',
    r'\bsh\s+[<(<]\s*.*curl\b',
    r'\bbash\s+[<(<]\s*.*curl\b',

    # Overwriting files with redirects
    r'>\s*/dev/',
    r'>\s*/proc/',
    r'>\s*/sys/',
]


def _is_dangerous(command: str) -> bool:
    """Check a command against dangerous patterns.

    Uses regex matching (case-insensitive). Returns True if any
    dangerous pattern is detected.
    """
    command_lower = command.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command_lower):
            return True
    return False


# ──────────────────────────────────────────────────────────────
# Paths outside the session working directory
# ──────────────────────────────────────────────────────────────

# A token is *path-like* when it starts with a path prefix (``~``, ``/``,
# ``./``, ``../``) or contains a separator. Bare file names (``notes.txt``) are
# relative to the working directory, so they need no check.
_PATHLIKE_RE = re.compile(r"^(?:~|/|\.{1,2}(?:/|$))")
# URLs also carry slashes; they are not filesystem paths.
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
# Anything whose expansion we cannot predict at parse time (``$HOME/x``).
_UNEXPANDABLE_RE = re.compile(r"[`$]")


def _looks_like_path(token: str) -> bool:
    """True if a shell token could name a file or directory."""
    return bool(_PATHLIKE_RE.match(token)) or "/" in token


def _candidate_paths(command: str) -> list[str]:
    """Extract the filesystem-looking tokens of a shell command.

    This is a static, best-effort analysis -- no shell is run to expand the
    command. Quoting, ``KEY=value`` assignments and redirections are handled;
    variables and URLs are skipped because their targets cannot be resolved
    here. A missed path only means a missed prompt, never a blocked command.
    """
    try:
        tokens = shlex.split(command, comments=False)
    except ValueError:
        # Unbalanced quotes: fall back to a naive split rather than give up.
        tokens = command.split()

    paths: list[str] = []
    for token in tokens:
        # ``--out=/tmp/x`` and ``FOO=/tmp/x``: keep the value half.
        if "=" in token:
            token = token.split("=", 1)[1]
        token = token.strip("'\"").rstrip(";,|&")
        if not token:
            continue
        if _URL_RE.match(token) or _UNEXPANDABLE_RE.search(token):
            continue
        if _looks_like_path(token):
            paths.append(token)
    return paths


def _session_temp_dir() -> Path | None:
    """The current session's scratch directory, or ``None``.

    Lazy import: ``agent.memory`` builds the session layout on first use, and
    this module must stay importable without touching the user's data
    directory.
    """
    from agent.memory import current_session_temp_dir

    return current_session_temp_dir()


def _session_pasted_dir() -> Path | None:
    """The current session's pasted-files directory, or ``None``.

    Lazy import for the same reason as :func:`_session_temp_dir`: this module
    must stay importable without touching the user's data directory.
    """
    from agent.memory import current_session_pasted_dir

    return current_session_pasted_dir()


def _tool_env() -> dict[str, str]:
    """The environment for a tool subprocess, with a session-private ``TMPDIR``.

    ``TMPDIR`` points inside the session, so anything a command leaves in the
    temp directory -- ``mktemp``, Python's ``tempfile``, a compiler's
    intermediates -- lands where the agent already owns the files instead of in
    the shared, world-readable ``/tmp``. It is the same directory
    :func:`_is_within_session` treats as inside, which is why writing to
    ``$TMPDIR`` needs no confirmation while a literal ``/tmp`` path still asks.
    """
    env = os.environ.copy()
    temp = _session_temp_dir()
    if temp is None:
        return env
    try:
        temp.mkdir(parents=True, exist_ok=True)
        temp.chmod(0o700)
    except OSError:
        # An unwritable scratch directory is not a reason to refuse a command:
        # the child simply keeps the inherited temp directory.
        return env
    env["TMPDIR"] = str(temp)
    return env


def _session_roots() -> list[Path]:
    """The directories that count as *inside* the session.

    The working directory, plus the session's two open directories: ``pasted/``
    (long prompts spilled to a file) and ``tmp/`` (the scratch space tool
    subprocesses get as ``TMPDIR``). Both are the user's own text rather than
    agent state.

    Everything else under the session directory is deliberately *not* a root.
    ``plans/``, ``notes.json``, ``chat_history.json`` and the vector store are
    agent state with a tool of their own -- ``/plan-read``, the memory tools,
    the store itself -- so reaching them by path is reaching around that tool,
    and the model has no business rewriting the file the agent is reading from.
    Sibling sessions are outside too, for the same reason as any other path.

    Roots are derived from the session, never hard-coded literals: the point of
    the check is that it is one rule rather than a list of trusted locations
    someone will keep appending to.
    """
    roots: list[Path] = []
    try:
        roots.append(Path.cwd().resolve())
    except OSError:
        pass
    for candidate in (_session_pasted_dir(), _session_temp_dir()):
        if candidate is None:
            continue
        try:
            roots.append(candidate.resolve())
        except (OSError, RuntimeError):
            pass
    return roots


def _is_within_session(path: Path | str) -> bool:
    """Whether *path* is inside the working directory or a session open dir.

    A path that cannot be resolved counts as *outside*: the check exists to ask
    about what it cannot vouch for, and a needless prompt costs one keystroke
    while a missed one costs whatever the command did.
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return any(
        resolved == root or root in resolved.parents for root in _session_roots()
    )


def _outside_session_paths(command: str) -> list[str]:
    """Return the command's path tokens that resolve outside the session.

    Each token is expanded (``~``) and resolved against the working directory,
    so ``../sibling`` and ``/etc/passwd`` are reported while ``src/app.py`` is
    not. Symlinks are resolved, so a link pointing out of the session counts as
    outside it. The session's ``pasted/`` and ``tmp/`` directories count as
    inside -- the user's own text and scratch space -- while the rest of the
    session directory (plans, notes, chat history, the vector store) is
    agent state with its own tools and therefore reaches the prompt.
    """
    outside: list[str] = []
    for raw in _candidate_paths(command):
        if _is_within_session(raw):
            continue
        if raw not in outside:
            outside.append(raw)
    return outside


def _confirmation_reasons(command: str, timeout: int) -> list[str]:
    """Collect the reasons this command needs explicit user approval.

    An empty list means none of the checks fired. Every reason is shown to the
    user, so the prompt says *why* it is being asked.
    """
    reasons: list[str] = []

    if _is_dangerous(command):
        reasons.append("Command matches dangerous patterns")

    if timeout > 60:
        reasons.append(f"Timeout ({timeout}s) is unusually long")

    if re.search(r'>\s*/(?:etc|boot|lib|usr|bin|sbin|opt)/', command):
        reasons.append("Command writes to a system directory")

    outside = _outside_session_paths(command)
    if outside:
        listed = ", ".join(outside[:5])
        if len(outside) > 5:
            listed += f", and {len(outside) - 5} more"
        reasons.append(f"Command references paths outside the session: {listed}")

    return reasons


def _prompt_user(command: str, reason: str) -> bool:
    """Ask the user to confirm (or reject) a command."""
    output = get_output_or_ipc()
    output.newline()
    output.print("⚠️ [warn]Command requires confirmation[/warn]", indent=2)
    output.print(f"[weak]Reason[/weak]: {reason}", indent=2)
    output.print(f"[prompt]$[/prompt] [cmd]{command}[/cmd]", indent=2)
    output.newline()

    confirmed = output.ask_confirm("[bold]Run this command?[/bold]", default=False)

    if confirmed:
        output.ok("Confirmed", indent=2)
    else:
        output.err("Cancelled by user", indent=2)

    return confirmed

# Actual run command tool
@tool(
    name="run_command",
    description=(
        "Execute a shell command and return the output.\n"
        "Use this when you need to run terminal commands, execute scripts, "
        "or interact with the filesystem via shell. "
        "Dangerous commands, and commands that touch files or directories "
        "outside the session working directory, will prompt the "
        "user for confirmation before running. The tool returns an error "
        "if the user declines.\n"
        "Use $TMPDIR for scratch files: it points at a private per-session "
        "temp directory, where writing needs no confirmation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds (default: 30)",
            },
            "_skip_confirmation": {
                "type": "boolean",
                "description": "Internal flag — do not use directly",
            },
        },
        "required": ["command"],
    },
)
def run_command_handler(args):
    """Execute a shell command and return its output.

    Before execution, collects the reasons the command needs approval
    (dangerous patterns, a long timeout, a system-directory redirect, a path
    outside the session). If there are any, the user is asked to confirm; a
    decline runs nothing.
    """
    command = args.get("command", "")
    timeout = args.get("timeout", 30)
    # Allow explicit bypass for non-interactive use
    skip_confirmation = args.get("_skip_confirmation", False)

    # Unsafe mode: skip all user prompts when agent.unsafe is true in config
    if get_config().get("agent.unsafe", False):
        skip_confirmation = True

    if not command:
        return {"error": "No command provided"}

    # ── Danger check & user prompt ──
    if not skip_confirmation:
        danger_reasons = _confirmation_reasons(command, timeout)

        if danger_reasons:
            reason = "; ".join(danger_reasons)
            confirmed = _prompt_user(command, reason)
            if not confirmed:
                return {
                    "error": (
                        "Command execution was cancelled by the user. "
                        "You asked to run a command that needs confirmation "
                        "and the user declined. Ask the user what they'd "
                        "like to do instead, or explain why this command "
                        "needs to be run."
                    ),
                    "user_cancelled": True,
                    "command": command,
                }
        else:
            # Safe-looking commands still get a lightweight prompt
            # so the user always sees what's about to run.
            output = get_output_or_ipc()
            output.print(f"[path][bold]$[/bold] {command}[/path]", indent=2)

    # Execute
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            # A session-private TMPDIR, so scratch work stays inside the
            # session (see _tool_env).
            env=_tool_env(),
        )
        output = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
        if result.returncode == 0:
            output["success"] = True
        else:
            output["success"] = False
        return output

    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}
