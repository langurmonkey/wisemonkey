"""Terminal execution tool.

Allows the agent to run shell commands on the host machine.
Usage: run_command(command="ls -la", timeout=30)

Security: commands run in the current working directory.
User confirmation is required for dangerous commands.
"""

import subprocess
import re

from rich.prompt import Confirm as RichConfirm

from agent.tools import tool
from agent.console import print

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

    # Directory changes
    r'\bcd\b',

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


def _requires_extra_confirmation(command: str, timeout: int) -> bool:
    """Decide if this command needs explicit user approval.

    Returns True if:
      - The command matches a dangerous pattern, OR
      - The timeout is long (> 60s), OR
      - The command involves writing to files outside common safe paths
    """
    if _is_dangerous(command):
        return True
    if timeout > 60:
        return True
    # Optional: catch commands that redirect output to system locations
    if re.search(r'>\s*/(?:etc|boot|lib|usr|bin|sbin|opt)/', command):
        return True
    return False


def _prompt_user(command: str, reason: str) -> bool:
    """Ask the user to confirm (or reject) a command."""
    print()
    print("[bold yellow]⚠️  Command requires confirmation[/bold yellow]")
    print(f"  [dim]Reason[/dim]: {reason}")
    print(f"  [white on #444444] [bold]$[/bold] {command} [/white on #444444]")
    print()

    confirmed = RichConfirm.ask("[bold]Run this command?[/bold]", default=False)

    if confirmed:
        print("  [green]✓ Confirmed[/green]")
    else:
        print("  [red]✗ Cancelled by user[/red]")

    return confirmed

def _prompt_user_pt(command: str, reason: str) -> bool:
    """Ask the user to confirm (or reject) a command."""
    from prompt_toolkit.shortcuts import yes_no_dialog

    confirmed = yes_no_dialog(
        title="<b>⚠️  Command requires confirmation</b>",
        text=f"Reason: {reason}\n <b>$</b> {command}\nRun this command?"
    ).run()

    if confirmed:
        print("  [green]✓ Confirmed[/green]")
    else:
        print("  [red]✗ Cancelled by user[/red]")

    return confirmed

# Actual tool
@tool(
    name="run_command",
    description=(
        "Execute a shell command and return the output.\n"
        "Use this when you need to run terminal commands, execute scripts, "
        "or interact with the filesystem via shell. "
        "Dangerous commands will prompt the "
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

    Before execution, checks for dangerous patterns. If flagged,
    prompts the user for confirmation.
    """
    command = args.get("command", "")
    timeout = args.get("timeout", 30)
    # Allow explicit bypass for non-interactive use
    skip_confirmation = args.get("_skip_confirmation", False)

    if not command:
        return {"error": "No command provided"}

    # ── Danger check & user prompt ──
    if not skip_confirmation:
        danger_reasons = []

        if _is_dangerous(command):
            danger_reasons.append("Command matches dangerous patterns")

        if timeout > 60:
            danger_reasons.append(f"Timeout ({timeout}s) is unusually long")

        if re.search(r'>\s*/(?:etc|boot|lib|usr|bin|sbin|opt)/', command):
            danger_reasons.append("Command writes to a system directory")

        if danger_reasons:
            reason = "; ".join(danger_reasons)
            confirmed = _prompt_user(command, reason)
            if not confirmed:
                return {
                    "error": (
                        "Command execution was cancelled by the user. "
                        "You asked to run a potentially dangerous command "
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
            print(f"  [white on #444444] [bold]$[/bold] {command} [/white on #444444]")

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
