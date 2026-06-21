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

from agent.utils import contractuser
from agent.console import print, err, ok, console, newline

# Ensure the project root (parent of agent/) is on the path
# This handles both pip-installed and direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def running_under_uv() -> bool:
    # Common env vars used by launchers (may vary by uv version)
    uv_markers = ["UV", "UV_PROJECT_ENVIRONMENT", "UVX"]
    if any(k in os.environ for k in uv_markers):
        return True
    return False

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
        "session",
        type=str,
        default="default",
        nargs='?',
        help="Name of the session to create or continue",
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
        '-t', '--tui',
        action='store_true',
        help='WIP/EXPERIMENTAL! Launch the Textual full-screen TUI',
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

    # Check if program runs within the uvx environment
    is_uv: bool = running_under_uv()

    # Handle --update
    if args.update:
        if is_uv:
            print("[accent]--update[/accent] flag not needed when running via [accent-bold]uv[/accent-bold]/[accent-bold]uvx[/accent-bold].")
            return

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
        from agent.prompt_ui import RichPromptUi
        cool, msg, _, _, _ = registry.run_command(core, "/config", RichPromptUi())
        if cool:
            ok(msg)
            newline()
            print("Start the agent with [accent-bold]wisemonkey \\[session-name][/accent-bold]")
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
        import datetime
        from agent.utils import pretty_timedelta

        sessions = []
        for f in os.listdir(SESSIONS_DIR):
            sdir = os.path.join(SESSIONS_DIR, f)
            if not os.path.isdir(sdir):
                continue
            mdf = os.path.join(sdir, ".session-metadata")
            last_access = ""
            accessed_dt = None
            if os.path.isfile(mdf):
                try:
                    with open(mdf, "r") as fh:
                        for line in fh:
                            line = line.strip()
                            if line.startswith("accessed:"):
                                raw = line.partition(":")[2].strip()
                                accessed_dt = datetime.datetime.fromisoformat(raw)
                                break
                except (OSError, ValueError):
                    pass
            if accessed_dt:
                delta = datetime.datetime.now() - accessed_dt
                last_access = pretty_timedelta(delta) + " ago"
            sessions.append((accessed_dt or datetime.datetime.min, f, sdir, last_access))

        sessions.sort(key=lambda x: x[0], reverse=True)

        print("Sessions:")
        for _, name, sdir, last_access in sessions:
            line = f"- [accent-bold]{name}[/accent-bold] - [dim]{contractuser(sdir)}[/dim]"
            if last_access:
                line += f" [time]({last_access})[/time]"
            print(line)
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

    # Launch TUI
    if args.tui:
        try:
            from agent.tui import WisemonkeyTui
            app = WisemonkeyTui(config_path=args.config, session=args.session)
            app.run()
        except Exception as e:
            err(f"TUI launch failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        return

    # Create agent
    try:
        from agent import Agent
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
