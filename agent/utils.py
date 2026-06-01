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
