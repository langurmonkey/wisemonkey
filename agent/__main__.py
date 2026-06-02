#!/usr/bin/env python3
"""
Langur Agent - entry point.

Langur Agent is a simple but powerful AI agent for the Linux and macOS terminal.
It features sessions, memory management, tools, skills, slash commands, and more.
"""

import argparse
import sys
import os
import traceback
import subprocess

from importlib.metadata import version as get_version
from rich.prompt import Confirm
from pathlib import Path
from xdg_base_dirs import xdg_data_home, xdg_config_home

from agent import Agent
from agent.utils import contractuser
from agent.console import print, err, ok, console

# Ensure the project root (parent of agent/) is on the path
# This handles both pip-installed and direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Run the agent interactively or as a one-shot query."""
    try:
        pkg_version = get_version("langur-agent")
    except Exception:
        # Fallback for development/uninstalled cases
        from agent import __version__
        pkg_version = __version__

    parser = argparse.ArgumentParser(
        description="Langur Agent - A simple and hackable AI agent for the Linux and macOS terminal",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        metavar="PATH",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--edit-config",
         action="store_true",
          help="Open the default configuration file in $EDITOR")
    parser.add_argument(
        '-V', '-v', '--version',
        action='version',
        version="%(prog)s ("+pkg_version+")")
    parser.add_argument(
        '-u', '--update',
        action='store_true',
        help='Update langur-agent from upstream and reinstall',
    )
    parser.add_argument(
        '-s', '--session',
        type=str,
        default='default',
        help='Name of the session to create or continue',
    )
    parser.add_argument(
        '-ls', '--ls',
        action='store_true',
        help='List existing sessions',
    )
    parser.add_argument(
        '-rm', '--rm',
        type=str,
        help='Delete a session by name',
    )
    args = parser.parse_args()

    SESSIONS_DIR = xdg_data_home() / "langur-agent" / "sessions"

    # Handle --update
    if args.update:
        XDG_DATA = xdg_data_home() or str(Path.home() / ".local" / "share")
        install_dir = f"{XDG_DATA}/langur-agent/repository"
        if not Path(install_dir).exists():
            print(f"langur-agent not installed. Installing to {install_dir}...")
            subprocess.run(['bash', '-c', f'BRANCH=main INSTALL_DIR="{install_dir}" curl -fsSL https://codeberg.org/langurmonkey/langur-agent/raw/branch/main/install.sh | bash'], check=True)
        else:
            print(f"Updating langur-agent in {install_dir}...")
            subprocess.run(['git', 'pull'], cwd=install_dir, check=True)
            print("Update complete.")
        return

    # Edit configuration
    if args.edit_config:
        from agent.config import edit_config_visual
        edit_config_visual()
        return

    # List sessions
    if args.ls:
        sessions = [f for f in os.listdir(SESSIONS_DIR) if os.path.isdir(os.path.join(SESSIONS_DIR, f))]
        print("Sessions:")
        for sess in sessions:
            print(f"- [accent-bold]{sess}[/accent-bold] - [dim]{contractuser(os.path.join(SESSIONS_DIR, sess))}[/dim]")
        return

    # Delete session
    if args.rm:
        session_dir = os.path.join(SESSIONS_DIR, args.rm)
        if os.path.isdir(session_dir):
            remove = Confirm.ask(f"Are you sure you want to [red bold]permanently delete[/red bold] the session [accent-bold]{args.rm}[/accent-bold]?", console=console)

            if remove:
                import shutil
                shutil.rmtree(session_dir)
                ok(f"Session deleted: [accent-bold]{args.rm}[/]")
        else:
            err(f"Session does not exist: [accent-bold]{args.rm}[/]")

        return
            

    try:
        agent = Agent(config_path=args.config, session=args.session)
    except Exception as e:
        err(f"Agent creation failed: {e}")
        sys.exit(1)

    # Interactive mode
    try:
        agent.run_interactive()
    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
