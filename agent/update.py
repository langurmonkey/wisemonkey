"""Update management for Wisemonkey.

Handles checking for updates, storing update metadata in a global
.updates.yml file next to the configuration, and performing updates.
"""

import subprocess
import yaml
from datetime import datetime, timedelta
from pathlib import Path

from agent.config import BASE_CONFIG_DIR

# Update metadata file
UPDATES_FILE = BASE_CONFIG_DIR / ".updates.yml"
# Update check interval in days
UPDATE_CHECK_INTERVAL = 2


class UpdatesManager:
    """Manages update checks and update metadata.

    Stores update metadata in a global .updates.yml file at
    ~/.config/wisemonkey/.updates.yml
    """

    def __init__(self):
        self._data = self._load()

    def _load(self):
        """Load update metadata from the global .updates.yml file."""
        if UPDATES_FILE.exists():
            try:
                with open(UPDATES_FILE, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except (yaml.YAMLError, OSError):
                pass
        return {}

    def _save(self):
        """Persist update metadata to the global .updates.yml file."""
        UPDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(UPDATES_FILE, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def get_last_check(self):
        """Return the datetime of the last update check, or None."""
        last_str = self._data.get("last_update_check")
        if last_str:
            try:
                return datetime.fromisoformat(last_str)
            except (ValueError, TypeError):
                return None
        return None

    def get_commit_hash(self):
        """Return the last known commit hash, or None."""
        return self._data.get("commit_hash")

    def get_updates_available(self):
        """Return whether updates are available."""
        return self._data.get("updates_available", False)

    def check_updates(self, repo_dir):
        """Check for updates from the git repository.

        Args:
            repo_dir: Path to the repository directory (parent of .git).

        Returns:
            (updates_available: bool, commit_hash: str | None)
        """
        now = datetime.now()

        # Skip the fetch check if we've checked within the last $UPDATE_CHECK_INTERVAL days
        last_check = self.get_last_check()
        if last_check and now - last_check < timedelta(days=UPDATE_CHECK_INTERVAL):
            return self.get_updates_available(), self.get_commit_hash(), last_check

        git_dir = Path(repo_dir) / ".git"
        commit_hash = None
        updates_available = False

        if git_dir.exists():
            try:
                # Current HEAD short hash
                result = subprocess.run(
                    ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    commit_hash = result.stdout.strip()

                # Check for remote updates
                result = subprocess.run(
                    ["git", "-C", str(repo_dir), "remote"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    subprocess.run(
                        ["git", "-C", str(repo_dir), "fetch", "--quiet"],
                        capture_output=True, timeout=15,
                    )
                    result = subprocess.run(
                        ["git", "-C", str(repo_dir),
                         "rev-list", "--count", "HEAD..@{upstream}"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        try:
                            behind_count = int(result.stdout.strip())
                            updates_available = behind_count > 0
                        except ValueError:
                            pass
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        # Persist results globally
        last_check = now.isoformat()
        self._data["last_update_check"] = now.isoformat()
        if commit_hash:
            self._data["commit_hash"] = commit_hash
        self._data["updates_available"] = updates_available
        self._save()

        return updates_available, commit_hash, last_check

    def perform_update(self):
        """Pull the latest code from upstream and reinstall.

        If the repository has not been cloned yet, the installer script
        is fetched from the internet. Updates metadata after completion.
        """
        from xdg_base_dirs import xdg_data_home
        from agent.console import print as aprint

        XDG_DATA = xdg_data_home()
        install_dir = Path(f"{XDG_DATA}/wisemonkey/repository")

        if not install_dir.exists():
            aprint(f"wisemonkey not installed. Installing to {install_dir}...")
            subprocess.run(
                [
                    "bash", "-c",
                    f'BRANCH=main INSTALL_DIR="{install_dir}" '
                    "curl -fsSL "
                    "https://codeberg.org/langurmonkey/wisemonkey/raw/branch/main/install.sh "
                    "| bash",
                ],
                check=True,
            )
        else:
            aprint(f"Updating wisemonkey in {install_dir}...")
            subprocess.run(["git", "pull"], cwd=install_dir, check=True)
            aprint("Update complete.")

        # Update metadata
        now = datetime.now()
        commit_hash = None
        git_dir = install_dir / ".git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(install_dir), "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    commit_hash = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        self._data["last_update_check"] = now.isoformat()
        if commit_hash:
            self._data["commit_hash"] = commit_hash
        self._data["updates_available"] = False
        self._save()
