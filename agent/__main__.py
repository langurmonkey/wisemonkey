#!/usr/bin/env python3
"""Langur Agent - entry point."""

import argparse
import sys
import traceback
from importlib.metadata import version as get_version
from rich import print
from pathlib import Path

# Ensure the project root (parent of agent/) is on the path
# This handles both pip-installed and direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent import Agent


def main():
    """Run the agent interactively or as a one-shot query."""
    try:
        pkg_version = get_version("langur-agent")
    except Exception:
        # Fallback for development/uninstalled cases
        from agent import __version__
        pkg_version = __version__

    parser = argparse.ArgumentParser(
        description="Langur Agent - Simple and hackable AI assistant",
    )
    parser.add_argument(
        "-c", "--config",
        metavar="PATH",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "query",
        nargs=argparse.REMAINDER,
        help="One-shot query (if provided, runs once and exits)",
    )
    parser.add_argument(
        '-V', '-v', '--version',
        action='version',
        version="%(prog)s ("+pkg_version+")")
    parser.add_argument(
        '-u', '--update',
        action='store_true',
        help='Update langur-agent from upstream and reinstall',
    )
    args = parser.parse_args()

    # Handle --update
    if args.update:
        import subprocess
        from xdg_base_dirs import xdg_data_home
        
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

    try:
        agent = Agent(config_path=args.config)
    except Exception as e:
        print(f"[red]ERROR:[/red] Agent creation failed: {e}")
        sys.exit(1)

    # One-shot mode: langur-agent "your query"
    if args.query:
        query = " ".join(args.query)
        result = agent.run(query)
        # run() returns (text, total_tokens, ntools, total_gen_time) tuple
        if isinstance(result, tuple):
            print(result[0])
            if len(result) > 3:
                print(f"\n[black on #777777]  ⏣  {result[3]:.1f}s  ⏣  {result[1]} tokens  ⏣  {result[2]} tools  [/black on #777777]")
        else:
            print(result)
        return

    # Interactive mode
    try:
        agent.run_interactive()
    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
