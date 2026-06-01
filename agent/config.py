"""In-memory configuration management with file persistence.

Uses a singleton pattern to keep configuration in memory.
Follows XDG Base Directory spec:
- Config: ~/.config/langur-agent/config.yaml
- Fallback: if not found, copies ./config.yaml there
- If neither exists, returns defaults
"""

import shutil
import yaml
import json
import os
from pathlib import Path
from xdg_base_dirs import xdg_config_home
from dotenv import load_dotenv

# Load .env: current directory first, then home directory (without overwriting)
_cwd_env = Path(os.getcwd()) / ".env"
if _cwd_env.exists():
    load_dotenv(_cwd_env)
load_dotenv(Path.home() / ".env", override=False)

DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"
XDG_CONFIG_DIR = xdg_config_home() / "langur-agent"
XDG_CONFIG_FILE = XDG_CONFIG_DIR / "config.yaml"


def _ensure_xdg_config():
    """Ensure XDG config directory exists, copy ./config.yaml if needed."""
    if XDG_CONFIG_FILE.exists():
        return XDG_CONFIG_FILE

    XDG_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if DEFAULT_CONFIG.exists():
        shutil.copy2(DEFAULT_CONFIG, XDG_CONFIG_FILE)
        return XDG_CONFIG_FILE

    return XDG_CONFIG_FILE


def _get_defaults():
    """Return default configuration."""
    return {
        "model": {
            "name": "qwen/qwen3.6-35b-a3b",
            "base_url": "",
            "temperature": 0.8,
            "reasoning_effort": "medium",
            "reasoning_visible": False,
        },
        "embedding": {
            "name": "text-embedding-qwen3-embedding-0.6b",
            "base_url": "",
        },
        "agent": {
            "max_turns": 50,
            "system_prompt": "You are a helpful assistant.",
            "markdown": False,
            "max_chat_history": 128000,
            "vi_mode": False,
        },
    }


class Config:
    """Singleton configuration manager.

    Keeps configuration in memory with optional file persistence.
    """

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
        """Return the XDG config file path."""
        return _ensure_xdg_config()

    def load(self, path=None):
        """Load configuration from a YAML file."""
        if path:
            config_path = Path(path)
        else:
            config_path = _ensure_xdg_config()

        self._config_path = config_path

        if config_path.exists():
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._config = file_config
        else:
            self._config = _get_defaults()

    def save(self):
        """Persist current configuration to the file."""
        if self._config_path is None:
            self._config_path = _ensure_xdg_config()

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def set(self, key: str, value):
        """Set a configuration value using dot notation.

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
        self.save()

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
        """Return the full configuration as a dictionary."""
        return self._config.copy()

    def reset(self):
        """Reset configuration to defaults."""
        self._config = _get_defaults()
        self._config_path = None

    def __repr__(self):
        return f"Config(config={self._config})"


# Module-level singleton access
_config_instance = None


def get_config():
    """Get the singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def get_default_config_path():
    """Get the default config file path (backward compatibility)."""
    return _ensure_xdg_config()


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
