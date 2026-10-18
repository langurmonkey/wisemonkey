"""In-memory configuration management with file persistence.

Uses a singleton pattern to keep configuration in memory.
Follows XDG Base Directory spec:
- Config: ~/.config/wisemonkey/config.yaml
- Fallback: if not found, copies ./config.yaml there
- If neither exists, returns defaults
"""
import json
import os
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from xdg_base_dirs import xdg_config_home

from agent.output import OutputAdapter

BASE_CONFIG_DIR = xdg_config_home() / "wisemonkey"

DEFAULT_BASE_CONFIG = Path(__file__).parent.parent / "config.yaml"
BASE_CONFIG_FILE = BASE_CONFIG_DIR / "config.yaml"

DEFAULT_MCP_CONFIG = Path(__file__).parent.parent / "mcp.json"
MCP_CONFIG_FILE = BASE_CONFIG_DIR / "mcp.json"

# Load .env:
# - current directory first,
# - then config directory,
# - then home directory (without overwriting)
_cwd_env = Path(os.getcwd()) / ".env"
_config_env = Path(BASE_CONFIG_DIR) / ".env"
if _cwd_env.exists():
    load_dotenv(_cwd_env)
elif _config_env.exists():
    load_dotenv(_config_env)
else:
    load_dotenv(Path.home() / ".env", override=False)


def _ensure_base_config():
    """Ensure base config directory exists, copy ./config.yaml if needed."""
    if BASE_CONFIG_FILE.exists():
        return BASE_CONFIG_FILE

    # Copy default file to configuration directory
    BASE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if DEFAULT_BASE_CONFIG.exists():
        shutil.copy2(DEFAULT_BASE_CONFIG, BASE_CONFIG_FILE)
        return BASE_CONFIG_FILE

    return BASE_CONFIG_FILE

def _ensure_mcp_config():
    """Ensure XDG config directory exists, create empty config if needed."""
    if MCP_CONFIG_FILE.exists():
        return MCP_CONFIG_FILE

    # Create empty mcp configuration.
    BASE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {"mcpServers": {}}
    MCP_CONFIG_FILE.write_text(json.dumps(config, indent=2))

    return MCP_CONFIG_FILE

def _get_defaults():
    """Loads the default config from wisemonkey/config.yaml"""
    repo_config = Path(__file__).parent.parent / "config.yaml"
    if repo_config.exists():
        with open(repo_config, "r") as f:
            file_config = yaml.safe_load(f)
            if file_config:
                return file_config
    return {}


class Config:
    """Singleton configuration manager.

    Keeps configuration in memory with optional file persistence.
    """

    _instance: "Config"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._config = _get_defaults()
        self._config_path = None

    @property
    def config_path(self):
        """Return the currently selected base config file path."""
        return self._config_path or _ensure_base_config()

    @property
    def mcp_config_path(self):
        """Get the MCP configuration file path"""
        return _ensure_mcp_config()

    def load(self, path=None):
        """Load configuration from a YAML file."""
        if path:
            config_path = Path(path)
        else:
            config_path = _ensure_base_config()

        self._config_path = config_path

        if config_path.exists():
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                self._config = file_config if isinstance(file_config, dict) else _get_defaults()
        else:
            agent_dir = Path(__file__).resolve().parent
            repo_dir = agent_dir.parent
            self.load(repo_dir / "config.yaml")

    def load_for_session(self, session_name: str):
        """Load configuration for a session, falling back to global config.
        
        Looks for $XDG_DATA_HOME/wisemonkey/sessions/<session_name>/config.yaml.
        If not found, uses the global config file.
        """
        from xdg_base_dirs import xdg_data_home
        session_dir = xdg_data_home() / "wisemonkey" / "sessions" / session_name
        session_config_path = session_dir / "config.yaml"
        
        if session_config_path.exists():
            self.load(str(session_config_path))
        else:
            # Fall back to global config
            self.load()

    def reload(self):
        """Reload the currently selected configuration file."""
        self.load(self._config_path)

    def save(self):
        """Persist current configuration to the file."""
        if self._config_path is None:
            self._config_path = _ensure_base_config()

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def set(self, key: str, value):
        """Set a configuration value using dot notation.

        Args:
            key: Dot-separated key (e.g., "model.temperature")
            value: Value to set
        """
        self.set_no_save(key, value)
        self.save()

    def set_no_save(self, key: str, value):
        """Set a configuration value using dot notation, but does not persist it.

        Args:
            key: Dot-separated key (e.g., "model.temperature")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get(self, key: str, default=None):
        """Get a configuration value using dot notation.

        Args:
            key: Dot-separated key (e.g., "model.temperature")
            default: Default value if key doesn't exist

        Returns:
            The configuration value or default
        """
        keys = key.split(".")
        config = self._config

        for k in keys:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return default

        return config

    def has(self, key: str):
        """Checks if a configuration value using dot notation exists.

        Args:
            key: Dot-separated key (e.g., "model.temperature")

        Returns:
            True if the configuration value exists
        """
        keys = key.split(".")
        config = self._config

        for k in keys:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return False
        return True
        

    def to_dict(self):
        """Return the full configuration as a dictionary (deep copy)."""
        import copy
        return copy.deepcopy(self._config)

    def reset(self):
        """Reset configuration to defaults from repo config.yaml."""
        self._config = _get_defaults()
        self._config_path = None

    def __repr__(self):
        return f"Config(config={self._config})"


# Module-level singleton access
_config_instance = None

def get_config():
    """Get the singleton Config instance."""
    global _config_instance
    if not _config_instance:
        _config_instance = Config()
    return _config_instance


def get_default_config_path() -> Path:
    """Get the default config file path (backward compatibility)."""
    return _ensure_base_config()

def get_mcp_config_path() -> Path:
    """Get the MCP configuration file path"""
    return _ensure_mcp_config()


# The .env file wisemonkey writes secrets to. It is loaded at import time from
# the current directory, the config directory or $HOME (in that order of
# preference); writes always land in the config directory, which is the one the
# user owns and the one that survives a change of working directory.
ENV_FILE = BASE_CONFIG_DIR / ".env"

# The convention this tool looks for when listing keys. It is a *listing*
# filter, not a resolution rule: api_key_candidates() alone decides which
# variables a provider is allowed to read a key from.
API_KEY_NAME_SUFFIX = "_API_KEY"


def env_file_paths() -> list[Path]:
    """The .env files that dotenv loads, most specific first, de-duplicated."""
    candidates = [
        Path(os.getcwd()) / ".env",
        ENV_FILE,
        Path.home() / ".env",
    ]
    paths: list[Path] = []
    for path in candidates:
        if path not in paths:
            paths.append(path)
    return paths


def api_key_env_names(suffix: str = API_KEY_NAME_SUFFIX) -> list[str]:
    """Names of the API-key variables defined in the .env files or the environment.

    Parsing is deliberately shallow -- a name is a name, and the value is never
    read out -- because this only feeds a choice list, so that a key one already
    has on disk is visible instead of something to remember. '.env' entry order
    first, then the environment, so the file the user edits leads.
    """
    names: list[str] = []

    def add(name: str) -> None:
        if name.endswith(suffix) and name not in names:
            names.append(name)

    for path in env_file_paths():
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            name, sep, _ = line.partition("=")
            if sep:
                add(name.strip())

    for name in os.environ:
        add(name)

    return names


def _quote_env_value(value: str) -> str:
    """Quote a .env value only when it needs it."""
    if value == "" or any(c in value for c in " \t\n#\"'\\$"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def set_env_var(name: str, value: str, path: Path | None = None) -> Path:
    """Store ``NAME=value`` in the config .env file, replacing any existing line.

    Secrets never go into config.yaml: that is the file which gets shared or
    committed, while .env is the one the user keeps. The file is created if
    missing and chmod 600 afterwards. The variable is also set in the current
    process, so the key works for the rest of this session without a restart.
    """
    target = Path(path) if path is not None else ENV_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    line = f"{name}={_quote_env_value(value)}"

    kept: list[str] = []
    replaced = False
    if target.exists():
        for existing in target.read_text().splitlines():
            stripped = existing.strip()
            body = (
                stripped[len("export "):].lstrip()
                if stripped.startswith("export ")
                else stripped
            )
            key = body.partition("=")[0].strip() if "=" in body else ""
            if key == name:
                if not replaced:
                    kept.append(line)
                    replaced = True
                continue
            kept.append(existing)
    if not replaced:
        kept.append(line)

    target.write_text("\n".join(kept).rstrip("\n") + "\n")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    os.environ[name] = value
    return target


def load_config(path=None):
    """Load config and return as dict (backward compatibility)."""
    config = get_config()
    config.load(path)
    return config.to_dict()


def log_config():
    """Log the current configuration (backward compatibility)."""
    config = get_config()
    path = config.config_path
    content = json.dumps(config.to_dict(), indent=4, sort_keys=True, default=str)
    return f"[bold]Config file[/bold]: [blue]{path}[/blue]\n{content}"

def edit_base_config_visual(output:OutputAdapter | None = None):
    """Edit the configuration file with $EDITOR or $VISUAL."""
    import subprocess
    config = get_config()
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    cmd = [editor, str(config.config_path)]
    if output is not None:
        return output.run_subprocess(cmd)
    else:
        return subprocess.run(cmd)

def edit_mcp_config_visual(output:OutputAdapter | None = None):
    """Edit the configuration file with $EDITOR or $VISUAL."""
    import subprocess
    config = get_config()
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    cmd = [editor, str(config.mcp_config_path)]
    if output is not None:
        return output.run_subprocess(cmd)
    else:
        return subprocess.run(cmd)
