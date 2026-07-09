"""File listing, reading, writing, and search tools.

Allows the agent to operate with files and directories in the file system,
including searching by name and content.
"""

import fnmatch
import os
import tempfile
from pathlib import Path
from textwrap import indent

from agent.utils import contractuser
from agent.tools import tool
from agent.output import get_output

def _prompt_user(command: str, reason: str) -> bool:
    """Ask the user to confirm (or reject) a search."""
    output = get_output()
    output.newline()
    output.print("⚠️ [warn]Search outside cwd requires confirmation[/warn]", indent=2)
    output.print(f"[weak]Reason[/weak]: {reason}", indent=2)
    output.print(f"[cmd]{command}[/cmd]", indent=2)
    output.newline()

    confirmed = output.ask_confirm("[bold]Accept this search?[/bold]", default=False)

    if confirmed:
        output.ok("Confirmed", indent=2)
    else:
        output.err("Cancelled by user", indent=2)

    return confirmed

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
    output = get_output()

    if not path:
        output.err("Path not given :/")
        return {"error": "No file path provided"}

    # Expand ~ and relative paths
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        output.err(f"Path does not exist: {path}")
        return {"error": f"The file does not exist: {contractuser(path)}"}

    if not os.path.isfile(path):
        output.err(f"Path is not a file: {path}")
        return {"error": f"The path exists but does not point to a file: {contractuser(path)}"}

    with open(path, "r") as file:
        output.print(f"[weak]Reading[/weak] [path]{contractuser(path)}[/path]",
                     indent=2)
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
    output = get_output()

    if not path:
        output.err("Path not given :/")
        return {"error": "No path provided"}

    # Expand ~ and relative paths
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        output.err(f"Path does not exist: {path}")
        return {"error": f"The directory does not exist: {contractuser(path)}"}

    if os.path.isfile(path):
        output.err(f"Path must be a directory: {path}")
        return {"error": f"The path exists but does not point to a directory: {contractuser(path)}"}
    output.print(f"[weak]Listing[/weak] [path]{contractuser(path)}[/path]",
                 indent=2)

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
    output = get_output()

    if not path:
        output.err("Path not given :/")
        return {"error": "No file path provided"}

    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
        output.print(f"[weak]Created directory[/weak] [path]{parent}[/path]",
                     indent=2)

    fd, tmp_path = tempfile.mkstemp(dir=parent if parent else None, prefix=".patched-")
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)

    output.print(f"[weak]Wrote[/weak] [path]{contractuser(path)}[/path] ({len(content)} chars)",
                 indent=2)

    return {"path": path, "success": True, "message": f"Wrote {len(content)} bytes to {contractuser(path)}"}


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
    output = get_output()

    if not path:
        output.err("Path not given :/")
        return {"error": "No file path provided"}
    if not old_string:
        output.err("Replace 'old_string' not provided :/")
        return {"error": "No 'old_string' provided."}

    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        output.err(f"Path does not exist: {path}")
        return {"error": f"The file does not exist: {contractuser(path)}"}
    if not os.path.isfile(path):
        output.err(f"Path is not a file: {path}")
        return {"error": f"The path is not a file: {contractuser(path)}"}
    
    from rich.markup import escape
    output.print(f"[weak]Patching[/weak] [path]{contractuser(path)}[/path]",
                 indent=2)
    output.print(f"[patch-remove]{escape(indent(old_string, '  - '))}[/patch-remove]")
    output.print(f"[patch-add]{escape(indent(new_string, '  + '))}[/patch-add]")

    with open(path, "r") as f:
        file_content = f.read()

    count = file_content.count(old_string)

    if count == 0:
        return {"error": (
            f"old_string not found in {contractuser(path)}. "
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
        "message": f"Replaced text in {contractuser(path)}",
    }


@tool(
    name="find_files",
    description=(
        "Find files by name using glob/wildcard patterns. "
        "Recursively searches a directory for files whose names match the given pattern. "
        "Use this instead of the shell 'find' command. "
        "Examples: '*.py' finds all Python files, '*.md' finds all Markdown files. "
        "Returns file paths relative to the search root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": "The directory to search, preferably in the current working directory (i.e. ./assets/doc).",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match filenames (e.g., '*.py', 'test_*', '*.{txt,md}'). Uses Unix shell-style wildcards.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum recursion depth. Default: unlimited (-1). Use 0 for root only, 1 for immediate children, etc.",
            },
        },
        "required": ["root", "pattern"],
    },
)
def find_files_handler(args):
    """
    Recursively find files by name pattern. This method asks the user for confirmation if the
    agent tries to find files outside the current working directory.
    """
    root = args.get("root", "")
    pattern = args.get("pattern", "")
    max_depth = args.get("max_depth", -1)
    output = get_output()

    if not root or not pattern:
        output.err("Both 'path' and 'pattern' are required")
        return {"error": "Both 'root' and 'pattern' are required"}

    root = os.path.expanduser(root)
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        output.err(f"Directory does not exist: {contractuser(root)}")
        return {"error": f"Directory does not exist: {contractuser(root)}"}

    # Ask for confirmation if not in cwd
    cwd = Path(os.getcwd())
    target = Path(root)
    if not os.path.samefile(cwd, target) and cwd not in target.parents:
        command = f"find_files {pattern} {target}"
        confirmed = _prompt_user(command, "Target not in current working directory")
        if not confirmed:
            return {
                "error": (
                    "'find_files' was cancelled by the user. "
                    "You asked to search for files outside the current working directory "
                    "and the user declined. The file you are trying to find is "
                    "probably in the current working directory. Try that instead."
                ),
                "user_cancelled": True,
                "command": command,
            }

    output.print(f"[weak]Searching for[/weak] [path]{pattern}[/path] [weak]in[/weak] [path]{contractuser(root)}[/path]",
                 indent=2)

    matches = []
    root_path = Path(root)

    for current_root, dirs, files in os.walk(root):
        # Compute current depth
        rel_path = Path(current_root).relative_to(root_path)
        depth = 0 if rel_path == Path(".") else len(rel_path.parts)

        if max_depth >= 0 and depth > max_depth:
            # Prevent os.walk from going deeper
            dirs.clear()
            continue

        for f in files:
            if fnmatch.fnmatch(f, pattern):
                full_path = os.path.join(current_root, f)
                rel = os.path.relpath(full_path, root)
                matches.append(rel)

    matches.sort()

    # Source of truth for format in description
    result_lines = [f"Found {len(matches)} file(s) matching '{pattern}' in {contractuser(root)}:"]
    if not matches:
        result_lines = [f"No files matching '{pattern}' in {contractuser(root)}"]
    else:
        for m in matches:
            result_lines.append(f"  - {m}")

    return {
        "root": root,
        "pattern": pattern,
        "count": len(matches),
        "files": matches,
        "content": "\n".join(result_lines),
    }


@tool(
    name="search_content",
    description=(
        "Search file contents for a text string. "
        "Recursively searches all text files in a directory for lines containing the given query string. "
        "Use this instead of the shell 'grep' command. "
        "Returns matching file paths with line numbers and context. "
        "Binary files are automatically skipped."
    ),
    parameters={
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": "The directory to search in (e.g., '/home/user/projects')",
            },
            "query": {
                "type": "string",
                "description": "The text to search for in file contents.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "If True, search is case-sensitive. Default: False.",
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines to show before and after each match. Default: 0.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum recursion depth. Default: unlimited (-1).",
            },
            "include_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of glob patterns to filter files by name (e.g., ['*.py', '*.md']). If not provided, all text files are searched.",
            },
        },
        "required": ["root", "query"],
    },
)
def search_content_handler(args):
    """Recursively search file contents for a text string."""
    root = args.get("root", "")
    query = args.get("query", "")
    case_sensitive = args.get("case_sensitive", False)
    context_lines = args.get("context_lines", 0)
    max_depth = args.get("max_depth", -1)
    include_patterns = args.get("include_patterns", None)

    if not root or not query:
        return {"error": "Both 'root' and 'query' are required"}

    root = os.path.expanduser(root)
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        return {"error": f"Directory does not exist: {contractuser(root)}"}

    output = get_output()
    output.print(f"[weak]Searching for[/weak] [path]'{query}'[/path] [weak]in[/weak] [path]{contractuser(root)}[/path]",
                 indent=2)

    root_path = Path(root)
    matches = []  # list of {"file": str, "line": int, "line_content": str, "context": list[str]}

    # Common binary extensions/text extensions heuristic
    _text_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".rst", ".html",
        ".css", ".scss", ".less", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".cfg", ".conf", ".xml", ".svg", ".sh", ".bash", ".zsh", ".fish",
        ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".java", ".kt", ".go",
        ".rs", ".rb", ".php", ".pl", ".lua", ".r", ".sql", ".env", ".gitignore",
        ".dockerfile", ".editorconfig", ".prettierrc", ".eslintrc",
    }

    def is_text_file(path_str):
        """Heuristic: known extension or sniff first few bytes."""
        ext = os.path.splitext(path_str)[1].lower()
        if ext in _text_extensions:
            return True
        # Try reading a small chunk
        try:
            with open(path_str, "rb") as f:
                chunk = f.read(8192)
            # If no null bytes, likely text
            return b"\0" not in chunk
        except Exception:
            return False

    def should_include(filename):
        if not include_patterns:
            return True
        for pat in include_patterns:
            if fnmatch.fnmatch(filename, pat):
                return True
        return False

    for current_root, dirs, files in os.walk(root):
        rel_path = Path(current_root).relative_to(root_path)
        depth = 0 if rel_path == Path(".") else len(rel_path.parts)

        if max_depth >= 0 and depth > max_depth:
            dirs.clear()
            continue

        for f in files:
            if not should_include(f):
                continue

            full_path = os.path.join(current_root, f)

            if not is_text_file(full_path):
                continue

            try:
                with open(full_path, "r", errors="replace") as fh:
                    lines = fh.readlines()
            except Exception:
                continue

            for i, line in enumerate(lines, start=1):
                check_line = line if case_sensitive else line.lower()
                check_query = query if case_sensitive else query.lower()

                if check_query in check_line:
                    file_rel = os.path.relpath(full_path, root)
                    entry: dict[str, str | int | list[str]] = {
                        "file": file_rel,
                        "line": i,
                        "line_content": line.rstrip("\n"),
                        "context": [],
                    }

                    if context_lines > 0:
                        ctx: list[str] = []
                        start_ctx = max(0, i - 1 - context_lines)
                        end_ctx = min(len(lines), i + context_lines)
                        for ci in range(start_ctx, end_ctx):
                            prefix = ">" if ci == i - 1 else " "
                            ctx.append(f"{prefix} {ci + 1}: {lines[ci].rstrip(chr(10))}")
                        entry["context"] = ctx

                    matches.append(entry)

    # Build output
    if not matches:
        result_lines = [f"No matches for '{query}' in {contractuser(root)}"]
    else:
        result_lines = [
            f"Found {len(matches)} match(es) for '{query}' in {contractuser(root)}:"
        ]
        current_file = None
        for m in matches:
            if m["file"] != current_file:
                current_file = m["file"]
                result_lines.append("")
                result_lines.append(f"  {current_file}:")
            result_lines.append(f"    {m['line']}: {m['line_content']}")
            context = m.get("context")
            if isinstance(context, list):
                for ctx_line in context:
                    result_lines.append(f"      {ctx_line}")

    return {
        "root": root,
        "query": query,
        "count": len(matches),
        "results": matches[:500],  # cap results
        "content": "\n".join(result_lines),
    }
