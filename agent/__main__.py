#!/usr/bin/env python3
"""
Wisemonkey main.

This file contains the main, which parses the CLI arguments and routes the actions to
the proper modules. It also creates the actual agent and runs it.
"""

import argparse
import sys
import os
import traceback

from importlib.metadata import version as get_version
from importlib.metadata import metadata
from rich.prompt import Confirm
from pathlib import Path
from xdg_base_dirs import xdg_data_home

from agent import Agent
from agent.utils import contractuser
from agent.console import print, err, ok, console, newline

# Ensure the project root (parent of agent/) is on the path
# This handles both pip-installed and direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Run the agent interactively or as a one-shot query."""
    try:
        pkg_version = get_version("wisemonkey")
        pkg_name = metadata("wisemonkey")['Name'].capitalize()
        pkg_summary = metadata("wisemonkey")['Summary']
    except Exception:
        # Fallback for development/uninstalled cases
        from agent import __version__
        pkg_version = __version__
        pkg_name = "Wisemonkey"
        pkg_summary = "Simple and hackable AI agent"

    parser = argparse.ArgumentParser(
        description=f"{pkg_name} - {pkg_summary}",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        metavar="PATH",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-o", "--onboard",
        action='store_true',
        help="Interactively configure the agent",
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
        help='Update wisemonkey from upstream and reinstall',
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

    SESSIONS_DIR = xdg_data_home() / "wisemonkey" / "sessions"

    # Handle --update
    if args.update:
        from agent.update import UpdatesManager
        UpdatesManager().perform_update()
        return

    # Handle --onboard
    if args.onboard:
        newline()
        console.rule(f"[accent-bold]{pkg_name.upper()}[/accent-bold] ONBOARDING")
        newline()
        # Load configuration
        from agent.core import Core
        core = Core(args.config, args.session, full_startup=False)

        from agent.commands import registry
        cool, msg, _, _, _ = registry.run_command(core, "/config")
        if cool:
            ok(msg)
            newline()
            print("Start the agent with [accent-bold]wisemonkey \\[--session session-name][/accent-bold]")
        else:
            err(msg)

        return
        
        

    # Edit configuration
    if args.edit_config:
        from agent.config import edit_base_config_visual
        edit_base_config_visual()
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
        traceback.print_exc()
        sys.exit(1)

    # Interactive mode
    try:
        agent.run_interactive()
    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
