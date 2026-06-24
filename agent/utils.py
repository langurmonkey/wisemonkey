"""Utility functions for wisemonkey."""

from __future__ import annotations

import base64
import re
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image

# Terminal helpers

def term_width():
    import os
    try:
        w = os.get_terminal_size().columns
    except Exception:
        w = 80
    return w


# Image helpers

def resize_image(image_bytes: bytes, max_dim: int = 1024, quality: int = 70) -> dict:
    """Resize an image so its longest side is at most *max_dim* pixels and
    encode as JPEG at the given *quality* (1-100).

    Returns a dict with ``image_base64`` (str) and ``mime_type`` (``"image/jpeg"``).
    """
    img = Image.open(BytesIO(image_bytes))

    # Convert RGBA/P to RGB for JPEG
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Downscale maintaining aspect ratio
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim / w, max_dim / h)
        img = img.resize((int(w * ratio), int(h * ratio)))

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"image_base64": b64, "mime_type": "image/jpeg"}


# Path helpers

def contractuser(path_str: str | Path) -> str:
    """
    Contracts the user home directory in a string to ~
    """
    home = Path.home()
    path = Path(path_str).resolve()
    try:
        relative = path.relative_to(home)
        return str(Path("~", relative))
    except ValueError:
        # Path is not under the home directory — return unchanged
        return str(path)


# Command-tree helpers (for NestedCompleter)

separator_re = re.compile(r"[-\s]+")

def add_command(tree: dict, command: str) -> None:
    # Keep leading "/" so NestedCompleter keys match the raw input.
    # Split on "-" or spaces.
    cmd = command.strip()
    parts = [
        part
        for part in separator_re.split(cmd)
        if part
    ]

    if not parts:
        return

    node = tree

    for part in parts[:-1]:
        # If this command was previously a leaf, convert it to a nested dict.
        if node.get(part) is None:
            node[part] = {}

        node = node[part]

    # Do not overwrite an existing nested dict.
    node.setdefault(parts[-1], None)

def collapse_none_dicts(obj):
    if not isinstance(obj, dict):
        return obj

    # First recursively process children.
    collapsed = {
        key: collapse_none_dicts(value)
        for key, value in obj.items()
    }

    # If all values are None and there are multiple keys, convert to a set.
    # Single-key dicts are kept as dicts so NestedCompleter.from_nested_dict
    # accepts them (it requires dict branches, not sets).
    if collapsed and all(value is None for value in collapsed.values()):
        if len(collapsed) > 1:
            return set(collapsed.keys())
        # Single leaf: keep as dict so NestedCompleter works.

    return collapsed


# Time helpers

def pretty_timedelta(delta):
    """
    Pretty printing a `timedelta` object form `datetime` Python module

    Acknowledgements:
    @thatalextaylor for his earlier version:
    https://gist.github.com/thatalextaylor/7408395
    That I used to modify the script.
    Args:
        delta -- `datatime` Python time delta object
    Returns:
        None -- just a printing function -- works better with Jupyter Notebook
    """
    timedelta_seconds = delta.total_seconds()

    # Seconds will be int-s, timedelta_seconds stores also decimal places
    seconds = timedelta_seconds

    # Can be negative
    sign_string = '-' if seconds < 0 else ''

    seconds = abs(int(seconds))

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd %dh %dm %ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh %dm %ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm %ds' % (sign_string, minutes, seconds)
    elif seconds > 0:
        return '%s%ds' % (sign_string, seconds)
    else:
        'no time'


# Prompt UI abstraction layer

class PromptUi(Protocol):
    """Abstract interface for user prompting.

    Implementations provide the same prompting primitives that ``rich.prompt``
    and ``prompt_toolkit`` offer, but routed through whichever UI is active
    (classic REPL or full-screen TUI).
    """

    def ask_string(self, message: str, default: str = "") -> str:
        """Prompt the user for a free-form string."""

    def ask_float(self, message: str, default: float = 0.0) -> float:
        """Prompt the user for a floating-point number."""

    def ask_choice(
        self,
        message: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        """Present a list of *options* as ``(value, label)`` tuples and return
        the selected *value*."""

    def ask_confirm(self, message: str, default: bool = False) -> bool:
        """Ask a yes/no question and return the boolean answer."""
