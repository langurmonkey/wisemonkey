"""File listing, reading, and writing tools.

Allows the agent to operate with files and directories in the file system.
"""

import os
import tempfile

from agent.tools import tool
from agent.console import console

@tool(
    name="read_file",
    description=(
        "Read the full contents of a file.\n"
        "Use this when the user asks to "
        "see file contents, check a file's content, or read any file. "
        "Takes a 'path' argument (absolute or relative path to the file). "
        "Optional 'show_line_numbers' (bool) to prepend line numbers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to read (e.g., '/home/user/file.txt')",
            },
            "show_line_numbers": {
                "type": "boolean",
                "description": "If True, prepend line numbers to each line (e.g. '1: line1').",
            },
        },
        "required": ["path"],
    },
)
def read_file_handler(args):
    """Read a file and return its content.

    Args:
        path: File path to read.
        show_line_numbers: If True, prepend line numbers (e.g. "1: line1\n").

    Returns:
        Dict with 'path' and 'content'.
    """
    path = args.get("path", "")
    show_line_numbers = args.get("show_line_numbers", False)

    if not path:
        return {"error": "No file path provided"}

    # Expand ~ and relative paths
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        return {"error": f"The file does not exist: {path}"}

    if not os.path.isfile(path):
        return {"error": f"The path exists but does not point to a file: {path}"}

    with open(path, "r") as file:
        console.print(f"  Reading [white on #444444]{path}[/white on #444444]")
        content = file.read()

    # Optionally add line numbers
    if show_line_numbers:
        lines = content.split("\n")
        max_width = len(str(len(lines)))
        numbered = "\n".join(
            f"{i+1:>{max_width}}: {line}" for i, line in enumerate(lines)
        )
        content = numbered

    output = {
        "path": path,
        "content": content
    }
    return output


@tool(
    name="list_dir",
    description=(
        "List all files and subdirectories in a directory.\n"
        "Use this when the "
        "user asks to see what's in a folder, list directory contents, or "
        "explore a directory structure. Takes a 'path' argument (absolute or "
        "relative path to the directory)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The directory path to list (e.g., '/home/user/projects')",
            },
        },
        "required": ["path"],
    },
)
def list_dir_handler(args):
    """List contents of a directory."""
    path = args.get("path", "")

    if not path:
        return {"error": "No path provided"}

    # Expand ~ and relative paths
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        return {"error": f"The directory does not exist: {path}"}

    if os.path.isfile(path):
        return {"error": f"The path exists but does not point to a directory: {path}"}

    console.print(f"  Listing [white on #444444]{path}[/white on #444444]")

    content = os.listdir(path)
    # Format as a readable listing
    dirs = [f for f in content if os.path.isdir(os.path.join(path, f))]
    files = [f for f in content if os.path.isfile(os.path.join(path, f))]
    lines = []
    if dirs:
        lines.append(f"Directories ({len(dirs)}):")
        for d in sorted(dirs):
            lines.append(f"- {d}/")
    if files:
        lines.append(f"Files ({len(files)}):")
        for f in sorted(files):
            lines.append(f"- {f}")

    return {
        "path": path,
        "content": "\n".join(lines) if lines else "(empty directory)",
        "dirs": dirs,
        "files": files,
    }


@tool(
    name="write_file",
    description=(
        "Write or completely overwrite a file with new content.\n"
        "Creates parent directories if they don't exist. "
        "Use this when creating a new file or rewriting an entire file.\n"
        "For targeted edits, use 'patch_file' instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to write (e.g., '/home/user/file.txt')",
            },
            "content": {
                "type": "string",
                "description": "The full content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file_handler(args):
    """Write or overwrite a file with new content. Creates parent directories."""
    path = args.get("path", "")
    content = args.get("content", "")

    if not path:
        return {"error": "No file path provided"}

    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
        console.print(f"  Created directory [white on #444444]{parent}[/white on #444444]")

    fd, tmp_path = tempfile.mkstemp(dir=parent if parent else None, prefix=".patched-")
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)

    console.print(f"  Wrote [white on #444444]{path}[/white on #444444] ({len(content)} chars)")

    return {"path": path, "success": True, "message": f"Wrote {len(content)} bytes to {path}"}


@tool(
    name="patch_file",
    description=(
        "Apply a targeted edit to a file by replacing exact text.\n"
        "Provide the EXACT text to find (old_string) and the replacement (new_string). "
        "The edit only succeeds if old_string appears exactly once in the file. "
        "Use this for surgical edits like changing a variable name, fixing a bug, "
        "or updating a function body. For large changes, prefer 'write_file'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to edit (e.g., '/home/user/file.py')",
            },
            "old_string": {
                "type": "string",
                "description": "EXACT text to be replaced. Must match the file contents precisely.",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text to insert",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
)
def patch_file_handler(args):
    """Apply a targeted edit to a file using search/replace."""
    path = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

    if not path:
        return {"error": "No file path provided"}
    if not old_string:
        return {"error": "No 'old_string' provided."}

    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        return {"error": f"The file does not exist: {path}"}
    if not os.path.isfile(path):
        return {"error": f"The path is not a file: {path}"}

    console.print(f"  Patching [white on #444444]{path}[/white on #444444]")
    console.print(f"  [white on red]-{old_string}[/]")
    console.print(f"  [white on green]+{new_string}[/]")

    with open(path, "r") as f:
        file_content = f.read()

    count = file_content.count(old_string)

    if count == 0:
        return {"error": (
            f"old_string not found in {path}. "
            f"Use 'read_file' to check current contents and copy exact text."
        )}

    if count > 1:
        return {"error": (
            f"old_string found {count} times. "
            f"Include more surrounding context for unique match."
        )}

    new_content = file_content.replace(old_string, new_string, 1)

    parent = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=parent if parent else None, prefix=".patched-")
    try:
        os.write(fd, new_content.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)

    return {
        "path": path,
        "success": True,
        "message": f"Replaced text in {path}",
    }
