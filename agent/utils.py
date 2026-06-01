import re
from pathlib import Path

def contractuser(path_str: str) -> str:
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

separator_re = re.compile(r"[-\s]+")
def add_command(tree: dict, command: str) -> None:
    # Remove leading "/" and split on "-" or spaces.
    parts = [
        part
        for part in separator_re.split(command.strip().lstrip("/"))
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

    # If all values are None, convert this dict to a set of keys.
    if collapsed and all(value is None for value in collapsed.values()):
        return set(collapsed.keys())

    return collapsed
