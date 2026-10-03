"""File listing, reading, writing, and search tools.

Allows the agent to operate with files and directories in the file system,
including searching by name and content.
"""

import fnmatch
import os
import re
import tempfile
from pathlib import Path
from textwrap import indent

from agent.output import get_output_or_ipc
from agent.tools import tool
from agent.utils import contractuser


def _prompt_user(command: str, reason: str) -> bool:
    """Ask the user to confirm (or reject) a search."""
    output = get_output_or_ipc()
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
        "Read the contents of a file.\n"
        "Use this when the user asks to "
        "see file contents, check a file's content, or read any file. "
        "Takes a 'path' argument (absolute or relative path to the file). "
        "Optional 'show_line_numbers' (bool) to prepend line numbers. "
        "Optional 'max_lines' (int) to read only the first N lines from the "
        "top (like the 'head' command) \u2014 use this for large files. "
        "Optional 'offset' (int, 1-based) to start at a given line, so a "
        "region of a big file can be read directly instead of shelling out to "
        "'sed'. Combine it with 'max_lines' to read a window; line numbers and "
        "the 'start_line'/'end_line' metadata always refer to real positions in "
        "the file, not to the window."
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
            "max_lines": {
                "type": "integer",
                "description": "Read at most N lines (like the 'head' command). Combined with 'offset' this reads a window; omit or set to 0 for no limit.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from (like 'tail -n +N'). Omit or set to 0 to start at the top of the file.",
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
        max_lines: If > 0, read at most this many lines (like 'head').
        offset: If > 0, start at this 1-based line (like 'tail -n +N').

    Returns:
        Dict with 'path' and 'content', plus 'start_line' and 'end_line'. When
        the file was not returned whole it also carries 'truncated',
        'total_lines' and 'shown_lines'.
    """
    path = args.get("path", "")
    show_line_numbers = args.get("show_line_numbers", False)
    max_lines = args.get("max_lines", 0)
    offset = args.get("offset", 0)
    output = get_output_or_ipc()

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

    max_lines = _as_int(max_lines)
    offset = _as_int(offset)

    # `offset` is 1-based for the caller, like `tail -n +N`. A negative value
    # is not a range from the end -- that is `tail -n -N` territory, and it is
    # far more likely to be a miscount than an intent.
    if offset < 0:
        offset = 0

    # Read the window without pulling the whole file into memory: `offset`
    # exists precisely for the large-file case, and `readlines()` would defeat
    # it. The loop still walks the skipped prefix, but it holds one line at a
    # time instead of the whole file, and stops as soon as the window is full.
    selected: list[str] = []
    total_lines = 0
    with open(path, "r") as file:
        for index, line in enumerate(file, start=1):
            total_lines = index
            if index < offset:
                continue
            if max_lines and len(selected) >= max_lines:
                # The window is full, but `total_lines` is reported to the
                # caller and has to be the real length of the file, so the
                # rest is counted rather than iterated away.
                for _ in file:
                    total_lines += 1
                break
            selected.append(line)

    window_lines = len(selected)
    first_line = offset if offset > 1 else 1
    end_line = first_line + window_lines - 1 if window_lines else first_line
    # "Truncated" means lines existed that the caller did not see. Reaching the
    # end of a short file while looking for more is not that: the window asked
    # for 100 lines of a 2-line file, and got the whole file. Only a window that
    # starts past line 1 or that stops before the last line is truncated.
    truncated = first_line > 1 or end_line < total_lines

    where = (
        f"[weak](lines {first_line}-{end_line} of {total_lines} lines)[/weak]"
        if first_line > 1
        else ""
    )
    if where:
        output.print(
            f"[weak]Reading[/weak] [path]{contractuser(path)}[/path] {where}",
            indent=2,
        )
    else:
        output.print(f"[weak]Reading[/weak] [path]{contractuser(path)}[/path]",
                     indent=2)

    content = "".join(selected)

    # Optionally add line numbers. These are the real positions in the file, so
    # the width is computed from `total_lines`: numbering a window relative to
    # itself would make the numbers useless as an anchor for a later edit, and
    # would change the alignment between two reads of the same file.
    if show_line_numbers:
        lines = content.split("\n")
        # A trailing newline yields an empty final element; it is not a line.
        if lines and lines[-1] == "":
            lines.pop()
        max_width = len(str(total_lines))
        numbered = "\n".join(
            f"{first_line + i:>{max_width}}: {line}" for i, line in enumerate(lines)
        )
        content = numbered

    result = {
        "path": path,
        "content": content,
        "start_line": first_line,
        "end_line": end_line,
    }
    if truncated:
        result["truncated"] = True
        result["total_lines"] = total_lines
        result["shown_lines"] = window_lines
    return result


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
    output = get_output_or_ipc()

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
    output = get_output_or_ipc()

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


def _as_int(value, default: int = 0) -> int:
    """Best-effort int from a JSON argument, *default* when it is not one.

    Models do send numbers as strings when they mean to send numbers, so a bad
    value has to degrade rather than raise: an unparseable line number is
    treated as "not given", which keeps a malformed argument from turning a
    patch into a wrong-place edit.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _match_key(line: str) -> str:
    """The form of *line* used for whitespace-tolerant matching.

    Only trailing whitespace is dropped. Leading indentation is preserved
    because it carries meaning in every language we edit, while trailing
    whitespace is the one difference that reliably comes from a model's
    retyping of a block rather than from the file itself.
    """
    return line.rstrip()


def _find_line_blocks(file_lines: list[str], needle_lines: list[str]) -> list[int]:
    """Start indices of every contiguous occurrence of *needle_lines*.

    Matching is line-based and ignores trailing whitespace, so it needs the
    same line count in both: a block that differs only in indentation width
    inside it is a real difference and is *not* forgiven.
    """
    keys = [_match_key(line) for line in file_lines]
    needle = [_match_key(line) for line in needle_lines]
    span = len(needle)
    if span == 0 or span > len(keys):
        return []
    return [
        i
        for i in range(len(keys) - span + 1)
        if keys[i : i + span] == needle
    ]


def _apply_line_edit(
    file_lines: list[str],
    start: int,
    count: int,
    new_text: str,
) -> tuple[list[str], int, int]:
    """Replace *count* lines at *start* with *new_text*.

    Returns the new line list and the 1-based first line of the inserted text.

    The trailing-newline convention of the *replaced* block is honoured, not the
    one in *new_text*: a model that writes a replacement without a final
    newline must not silently join the following line onto the last inserted
    one, which is the kind of corruption that only shows up as a syntax error
    much later.
    """
    replaced = file_lines[start : start + count]
    had_newline = bool(replaced) and replaced[-1].endswith("\n")

    if new_text == "":
        inserted: list[str] = []
    else:
        inserted = new_text.splitlines(keepends=True)
        if had_newline and not inserted[-1].endswith("\n"):
            inserted[-1] += "\n"

    return file_lines[:start] + inserted + file_lines[start + count :], start + 1, start + len(inserted)


@tool(
    name="patch_file",
    description=(
        "Apply a targeted edit to a file. Two modes, both line-based.\n"
        "\n"
        "1. Text mode (default): give 'old_string' (exact text to find) and "
        "'new_string' (the replacement). Matching is whitespace-tolerant -- "
        "trailing whitespace is ignored, indentation is not -- so a block "
        "retyped by hand usually still matches. The edit only succeeds if the "
        "block appears exactly once; if it appears more than once, narrow it "
        "with 'start_line'/'end_line' to disambiguate.\n"
        "\n"
        "2. Line mode: give 'start_line' and 'end_line' (1-based, inclusive) "
        "with no 'old_string', and 'new_string' becomes the text replacing "
        "exactly those lines. Use this when you know the line numbers from "
        "'read_file' but not the exact text. Omit 'end_line' to replace a "
        "single line.\n"
        "\n"
        "Use this for surgical edits like changing a variable name, fixing a "
        "bug, or updating a function body. For large changes, prefer "
        "'write_file'. Both modes write atomically and report the line numbers "
        "they changed."
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
                "description": "EXACT text to be replaced. Must match the file contents precisely, modulo trailing whitespace. Omit to use line mode.",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text to insert. An empty string deletes the replaced lines.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based first line of the region to edit. With 'old_string' it narrows the search (use it to disambiguate a block that appears more than once); without it, this selects the start of the lines to replace.",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based last line of the region to edit, inclusive. Only used in line mode; omit to replace just 'start_line'.",
            },
        },
        "required": ["path", "new_string"],
    },
)
def patch_file_handler(args):
    """Apply a targeted edit to a file, by matching text or by line range.

    Args:
        path: The file to edit.
        old_string: Text to find. Enables text mode.
        new_string: The replacement text; empty deletes the replaced lines.
        start_line: 1-based line bound. In text mode it narrows the search; in
            line mode (no old_string) it is the first line to replace.
        end_line: 1-based inclusive end, line mode only.

    Returns:
        Dict with 'path', 'success' and 'start_line'/'end_line' of the
        inserted text, or 'error' describing what did not match.
    """
    path = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    start_line = _as_int(args.get("start_line", 0))
    end_line = _as_int(args.get("end_line", 0))
    output = get_output_or_ipc()

    if not path:
        output.err("Path not given :/")
        return {"error": "No file path provided"}
    if "new_string" not in args:
        output.err("Replace 'new_string' not provided :/")
        return {"error": "No 'new_string' provided."}

    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        output.err(f"Path does not exist: {path}")
        return {"error": f"The file does not exist: {contractuser(path)}"}
    if not os.path.isfile(path):
        output.err(f"Path is not a file: {path}")
        return {"error": f"The path is not a file: {contractuser(path)}"}

    with open(path, "r") as f:
        file_content = f.read()
    # `splitlines(keepends=True)` keeps a file that ends without a newline
    # distinguishable from one that does, which matters for the write-back.
    file_lines = file_content.splitlines(keepends=True)
    total_lines = len(file_lines)

    from rich.markup import escape

    # --- resolve the edit -------------------------------------------------
    if old_string:
        needle_lines = old_string.splitlines()
        if not needle_lines:
            output.err("'old_string' has no content :/")
            return {"error": "'old_string' is empty; nothing to match."}

        # The window is half-open in 0-based indices, and `end_line` is
        # inclusive and 1-based, so it converts to `end_line` with no shift:
        # a match starting at 0-based `i` ends at `i + len(needle)`, which is
        # the first index *past* its last line. Treating `end_line` as an
        # exclusive 1-based bound would cut the last matching line off.
        lo = start_line - 1 if start_line > 0 else 0
        hi = end_line if end_line > 0 else len(file_lines)
        if lo < 0:
            lo = 0
        if hi > len(file_lines):
            hi = len(file_lines)
        candidates = _find_line_blocks(file_lines, needle_lines)
        in_window = [i for i in candidates if lo <= i and i + len(needle_lines) <= hi]

        if not in_window and start_line > 0:
            # Scoping can exclude the real match; say so rather than reporting
            # a bare "not found", since the text may well be in the file.
            return {
                "error": (
                    f"'old_string' not found within lines {start_line}-{end_line} "
                    f"of {contractuser(path)}"
                    + (
                        f". It occurs at line(s) "
                        f"{', '.join(str(i + 1) for i in candidates)} "
                        f"({len(candidates)} occurrence(s) in the file) -- widen "
                        f"'start_line'/'end_line' or move the edit there."
                        if candidates
                        else ". Use 'read_file' with offset to check the region."
                    )
                )
            }

        if not in_window:
            return {
                "error": (
                    f"old_string not found in {contractuser(path)}. "
                    f"Use 'read_file' to check current contents and copy exact "
                    f"text, or use 'start_line'/'end_line' to patch by line."
                )
            }

        if len(in_window) > 1:
            return {
                "error": (
                    f"old_string found {len(in_window)} times "
                    f"(lines {', '.join(str(i + 1) for i in in_window)}). "
                    f"Narrow it with 'start_line'/'end_line', or include more "
                    f"surrounding context."
                )
            }

        at = in_window[0]
        verb = "Patched"
        removed = needle_lines
        new_lines, new_start, new_end = _apply_line_edit(
            file_lines, at, len(needle_lines), new_string
        )
    else:
        # --- line mode -----------------------------------------------------
        if start_line <= 0:
            output.err("Neither 'old_string' nor 'start_line' given :/")
            return {
                "error": (
                    "Provide 'old_string' to patch by text, or 'start_line' "
                    "(with optional 'end_line') to patch by line."
                )
            }
        if end_line > 0 and end_line < start_line:
            return {
                "error": (
                    f"'end_line' ({end_line}) is before 'start_line' "
                    f"({start_line})."
                )
            }
        if start_line > total_lines + 1:
            return {
                "error": (
                    f"'start_line' {start_line} is past the end of "
                    f"{contractuser(path)}, which has {total_lines} lines. "
                    f"Use {total_lines + 1} to append."
                )
            }
        if start_line == total_lines + 1:
            # Append. The line after the last one is a legitimate place to
            # insert, and refusing it would push the caller into text mode for
            # the one edit where the line numbers are already unambiguous.
            # Falls through to the shared write below with an empty
            # `removed`, so there is only one write path in this tool.
            at = total_lines
            removed = []
            new_lines, new_start, new_end = _apply_line_edit(
                file_lines, at, 0, new_string
            )
            verb = "Appended"
        else:
            last = end_line if end_line > 0 else start_line
            # A stale line count from an earlier read is normal; clamping is
            # more useful than refusing an edit that is otherwise well-formed.
            last = min(last, total_lines)
            at = start_line - 1
            removed = file_lines[at:last]
            new_lines, new_start, new_end = _apply_line_edit(
                file_lines, at, last - start_line + 1, new_string
            )
            verb = "Patched"

    # --- report and write --------------------------------------------------
    if new_lines == file_lines:
        output.print(
            f"[weak]No change[/weak] [path]{contractuser(path)}[/path] "
            f"[weak](replacement is identical to lines {new_start}-{new_end})[/weak]",
            indent=2,
        )
        return {
            "path": path,
            "success": True,
            "changed": False,
            "start_line": new_start,
            "end_line": new_end,
            "message": f"No change: the replacement matched the existing text in {contractuser(path)}",
        }

    output.print(f"[weak]{verb}[/weak] [path]{contractuser(path)}[/path]", indent=2)
    if removed:
        output.print(
            f"[weak]lines[/weak] [path]{at + 1}-{at + len(removed)}[/path] "
            f"[weak]→[/weak] [path]{new_start}-{new_end}[/path]"
        )
    if removed:
        output.print(f"[patch-remove]{escape(indent(''.join(removed), '  - '))}[/patch-remove]")
    if new_string:
        output.print(f"[patch-add]{escape(indent(new_string, '  + '))}[/patch-add]")

    parent = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=parent if parent else None, prefix=".patched-")
    try:
        os.write(fd, "".join(new_lines).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)

    return {
        "path": path,
        "success": True,
        "changed": True,
        "start_line": new_start,
        "end_line": new_end,
        "removed_lines": len(removed),
        "added_lines": new_end - new_start + 1,
        "message": (
            f"Replaced lines {at + 1}-{at + len(removed)} with "
            f"{new_end - new_start + 1} line(s) in {contractuser(path)}"
        ),
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
    output = get_output_or_ipc()

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
        "Search file contents. Recursively searches all text files in a "
        "directory and returns matching file paths with line numbers and "
        "context. Use this instead of the shell 'grep' command. Binary files "
        "are automatically skipped.\n"
        "\n"
        "By default 'query' is a literal substring. Set 'regex': true to treat "
        "it as a Python regular expression -- that is how to search for "
        "patterns like '^def ', 'self\\.\\w+ = ', or '@cmd\\(' instead of "
        "shell-quoting a regex into grep. With 'regex' false a query containing "
        "regex metacharacters is matched literally, so no escaping is needed "
        "for ordinary text.\n"
        "\n"
        "Regexes are applied per line with multiline anchors active, so '^' "
        "and '$' match the start and end of a line rather than the file. Set "
        "'multiline': true to match across newlines (for a pattern that spans "
        "lines), at the cost of holding the whole file in memory."
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
                "description": "The text to search for. A literal substring by default, or a Python regular expression when 'regex' is true.",
            },
            "regex": {
                "type": "boolean",
                "description": "If True, 'query' is a regular expression rather than a literal substring. Default: False.",
            },
            "multiline": {
                "type": "boolean",
                "description": "If True (requires 'regex'), allow matches spanning multiple lines. Default: False.",
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
            "max_results": {
                "type": "integer",
                "description": "Maximum matches to return. Default: 500. 'truncated' in the result says whether matches were dropped.",
            },
        },
        "required": ["root", "query"],
    },
)
def search_content_handler(args):
    """Recursively search file contents for a literal string or a regex.

    The literal path is the original behaviour and stays the default, because
    a query with regex metacharacters in it ("config.yaml", "a(b)c") must keep
    matching itself rather than silently becoming a pattern. Regex is opt-in
    via `regex`, and an invalid pattern is an error rather than a fallback to
    a literal search: a query that looks like a regex and fails to compile is
    a mistake worth reporting, not something to quietly reinterpret.
    """
    root = args.get("root", "")
    query = args.get("query", "")
    use_regex = bool(args.get("regex", False))
    multiline = bool(args.get("multiline", False))
    case_sensitive = args.get("case_sensitive", False)
    context_lines = _as_int(args.get("context_lines", 0))
    max_depth = args.get("max_depth", -1)
    include_patterns = args.get("include_patterns", None)
    max_results = _as_int(args.get("max_results", 500), default=500)
    if max_results <= 0:
        max_results = 500

    if not root or not query:
        return {"error": "Both 'root' and 'query' are required"}

    root = os.path.expanduser(root)
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        return {"error": f"Directory does not exist: {contractuser(root)}"}

    pattern: re.Pattern[str] | None = None
    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        # MULTILINE so '^' and '$' are line anchors, which is what a per-line
        # search means and what makes '^def ' work. Without it those anchors
        # would only ever match the start and end of the whole file, so a
        # pattern written the way it looks in grep would match nothing.
        if not multiline:
            flags |= re.MULTILINE
        try:
            pattern = re.compile(query, flags)
        except re.error as exc:
            return {
                "error": (
                    f"invalid regex {query!r}: {exc}. "
                    "Fix the pattern, or drop 'regex' to search for it literally."
                )
            }
    elif multiline:
        return {
            "error": "'multiline' requires 'regex': true -- a literal search has no pattern to span lines with."
        }

    output = get_output_or_ipc()
    kind = "regex" if use_regex else "text"
    output.print(
        f"[weak]Searching for[/weak] [path]'{query}'[/path] "
        f"[weak]({kind}) in[/weak] [path]{contractuser(root)}[/path]",
        indent=2,
    )

    root_path = Path(root)
    matches: list[dict[str, str | int | list[str]]] = []
    dropped = 0

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

    def make_entry(file_rel: str, lineno: int, text: str, lines: list[str],
                   span: int) -> dict[str, str | int | list[str]]:
        """One result, with context lines around the match.

        `lineno` is 1-based and `span` is how many lines the match covers, so a
        multiline match is anchored at its first line rather than at whatever
        line the scan happened to be on.
        """
        entry: dict[str, str | int | list[str]] = {
            "file": file_rel,
            "line": lineno,
            "line_content": text.rstrip("\n"),
            "context": [],
        }
        if context_lines > 0:
            ctx: list[str] = []
            start_ctx = max(0, lineno - 1 - context_lines)
            end_ctx = min(len(lines), lineno + span - 1 + context_lines)
            for ci in range(start_ctx, end_ctx):
                prefix = ">" if lineno <= ci + 1 < lineno + span else " "
                ctx.append(f"{prefix} {ci + 1}: {lines[ci].rstrip(chr(10))}")
            entry["context"] = ctx
        return entry

    for current_root, dirs, files in os.walk(root):
        rel_path = Path(current_root).relative_to(root_path)
        depth = 0 if rel_path == Path(".") else len(rel_path.parts)

        if max_depth >= 0 and depth > max_depth:
            dirs.clear()
            continue

        for f in sorted(files):
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

            file_rel = os.path.relpath(full_path, root)

            if multiline:
                # `multiline` is only reachable with `regex` (it is refused
                # otherwise, above), so a compiled pattern is guaranteed here.
                assert pattern is not None
                # The whole file is already in memory (readlines above); a
                # match may span lines, so its position has to be recovered
                # from a character offset rather than tracked per line.
                content = "".join(lines)
                for m in pattern.finditer(content):
                    lineno = content.count("\n", 0, m.start()) + 1
                    # Cut the reported text at the first newline so the
                    # rendered line stays one line, but keep the span so the
                    # covered lines are marked and given context.
                    first_line_end = content.find("\n", m.start())
                    if first_line_end == -1 or first_line_end > m.end():
                        text = content[m.start():m.end()]
                    else:
                        text = content[m.start():first_line_end] + " ..."
                    span = content.count("\n", m.start(), m.end()) + 1
                    if len(matches) < max_results:
                        matches.append(make_entry(file_rel, lineno, text, lines, span))
                    else:
                        dropped += 1
                continue

            for i, line in enumerate(lines, start=1):
                if pattern is not None:
                    found = pattern.search(line)
                else:
                    check_line = line if case_sensitive else line.lower()
                    check_query = query if case_sensitive else query.lower()
                    found = check_query in check_line
                if found:
                    if len(matches) < max_results:
                        matches.append(
                            make_entry(file_rel, i, line.rstrip("\n"), lines, 1)
                        )
                    else:
                        dropped += 1

    # Build output
    if not matches:
        result_lines = [f"No matches for '{query}' in {contractuser(root)}"]
    else:
        result_lines = [
            f"Found {len(matches)} match(es) for '{query}' in {contractuser(root)}:"
        ]
        if dropped:
            # The cap used to silently slice: a caller reading "Found 500"
            # could not tell 500 from 5000, and would conclude the pattern
            # only occurs 500 times.
            result_lines.append(
                f"  [{dropped} further match(es) omitted: raise 'max_results']"
            )
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
        "mode": kind,
        "count": len(matches),
        "omitted": dropped,
        "results": matches,
        "content": "\n".join(result_lines),
    }
