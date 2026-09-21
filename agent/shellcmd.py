"""User-invoked shell commands (the `!` prefix).

When the user types `!git status` at the prompt, the command is executed in
the shell, its output is shown, and the result is appended to the chat memory
so the model can use it as context on the next turn. This is the shell
analogue of slash commands: `!` runs locally, `/` runs agent commands.
"""

from __future__ import annotations

import subprocess

from agent.output import OutputAdapter, get_output_or_ipc

# Default timeout for user-invoked shell commands (seconds).
DEFAULT_TIMEOUT = 60


def run_shell_command(
    command: str,
    output: OutputAdapter | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Run a shell command and return a result dict.

    The command is executed synchronously with ``shell=True``. Output is
    printed to the given output adapter (or the active one), and the command
    plus its result are appended to the chat memory as a user exchange so it
    becomes part of the model's context.

    Parameters:
        command: str          - The shell command to run (without the `!`).
        output: OutputAdapter - Where to print output (defaults to the active
                                adapter, falling back to an IPC one).
        timeout: int          - Seconds before the command is killed.

    Returns the result dict from the subprocess run:
    ``{stdout, stderr, exit_code, success}`` or ``{error}``.
    """
    if output is None:
        output = get_output_or_ipc()

    output.print(f"[path][bold]$[/bold] {command}[/path]")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        output.err(f"Command timed out after {timeout}s")
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        output.err(f"Command failed: {e}")
        return {"error": str(e)}

    # Render the output
    if result.stdout:
        output.print(result.stdout.rstrip())
    if result.stderr:
        output.print(f"[err]{result.stderr.rstrip()}[/err]")
    if result.returncode != 0:
        output.err(f"Exit code: {result.returncode}")

    # Append to chat memory so the model sees the command and its output.
    content = f"$ {command}\n"
    if result.stdout:
        content += f"stdout:\n{result.stdout.rstrip()}\n"
    if result.stderr:
        content += f"stderr:\n{result.stderr.rstrip()}\n"
    content += f"exit code: {result.returncode}"

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
        "success": result.returncode == 0,
        "_chat_content": content,
    }


def append_to_memory(core, command: str, result: dict) -> None:
    """Append a shell command exchange to the session chat memory."""
    content = result.get("_chat_content")
    if content is None:
        return
    core.memory.add_chat_exchange(core, "user", content)
