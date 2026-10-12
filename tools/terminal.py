"""Terminal execution tool.

Allows the agent to run shell commands on the host machine.
Usage: run_command(command="ls -la", timeout=30)

Security: commands run in the current working directory. User confirmation
is required for dangerous commands and for any command that references a
file or directory outside the session working directory.
"""

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


def _outside_session_paths(command: str) -> list[str]:
    """Return the command's path tokens that resolve outside the cwd.

    Each token is expanded (``~``) and resolved against the working directory,
    so ``../sibling`` and ``/etc/passwd`` are reported while ``src/app.py`` is
    not. Symlinks are resolved, so a link pointing out of the session counts as
    outside it.
    """
    cwd = Path.cwd().resolve()
    outside: list[str] = []
    for raw in _candidate_paths(command):
        try:
            resolved = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved == cwd or cwd in resolved.parents:
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
        "if the user declines."
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
    outside the session working directory). If there are any, the user is
    asked to confirm; a decline runs nothing.
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
