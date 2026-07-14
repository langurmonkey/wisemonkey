"""Memory tool.

Allows the agent to operate on memory:

- Load/save memory
- Add notes
- Set user profile
- Session plans
"""
import os
import datetime

from agent.memory import Memory
from agent.tools import tool
from agent.output import get_output


def _get_plans_dir():
    """Get the plans directory for the current session, creating it if needed."""
    mem = Memory()
    plans_dir = mem.session_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    return plans_dir


def _parse_frontmatter(content):
    """Extract YAML frontmatter and body from plan content."""
    import re
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not match:
        return {}, content.strip()
    meta = {}
    for line in match.group(1).strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, match.group(2).strip()


@tool(
    name="save_session_plan",
    description=(
        "Save a session plan to the session's plans directory.\n"
        "The file is saved as $SESSION_DIR/plans/$NAME_$TIMESTAMP.md.\n"
        "Use this to create and persist plans for the current session."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name for the plan (e.g., 'REFACTOR', 'FEATURE_X')."
            },
            "content": {
                "type": "string",
                "description": "The full plan content in markdown."
            },
        },
        "required": ["name", "content"],
    },
)
def save_session_plan_handler(args):
    """Save a session plan to the plans directory."""
    name = args.get("name", "").strip().upper().replace(" ", "_")
    content = args.get("content", "")
    output = get_output()
    
    if not name:
        output.err("Plan name is required")
        return {"saved": False, "error": "Plan name is required"}
    if not content:
        output.err("Plan content is required")
        return {"saved": False, "error": "Plan content is required"}

    output.print(f"[weak]Saving plan[/weak] [path]{name}[/path]", indent=2)

    plans_dir = _get_plans_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.md"
    filepath = plans_dir / filename

    # Add YAML frontmatter with status
    now_iso = datetime.datetime.now().isoformat()
    frontmatter = f"---\nname: {name}\ncreated: {now_iso}\nstatus: active\ncompleted: ~\n---\n\n"
    filepath.write_text(frontmatter + content, encoding="utf-8")

    return {"saved": True, "file": str(filepath), "name": name, "status": "active"}


@tool(
    name="list_session_plans",
    description=(
        "List all saved session plans in the current session's plans directory.\n"
        "Returns each plan's name, file path, and last modified time."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)
def list_session_plans_handler(args):
    """List all saved session plans."""
    plans_dir = _get_plans_dir()
    files = sorted(plans_dir.glob("*.md"), reverse=True)
    if not files:
        return {"plans": [], "message": "No session plans found"}

    plans = []
    for f in files:
        name = f.stem  # e.g. "REFACTOR_20260626_120000"
        mod_time = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        # Parse frontmatter for status
        content = f.read_text()
        meta, _ = _parse_frontmatter(content)
        status = meta.get("status", "unknown")
        plans.append({
            "name": name,
            "file": str(f),
            "modified": mod_time.isoformat(),
            "status": status,
        })
    return {"plans": plans, "count": len(plans)}


@tool(
    name="update_session_plan_status",
    description=(
        "Update the status of a saved session plan.\\n"
        "Valid statuses: active, done, obsolete.\\n"
        "Provide the plan name or prefix as returned by list_session_plans."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The full name or prefix of the plan to update (e.g., 'REFACTOR_20260626_120000' or 'REFACTOR')."
            },
            "status": {
                "type": "string",
                "description": "New status: 'active', 'done', or 'obsolete'."
            },
        },
        "required": ["name", "status"],
    },
)
def update_session_plan_status_handler(args):
    """Update the status of a session plan in its frontmatter."""
    name = args.get("name", "").strip().upper().replace(" ", "_")
    status = args.get("status", "").strip().lower()
    if not name:
        return {"updated": False, "error": "Plan name is required"}
    if status not in ("active", "done", "obsolete"):
        return {"updated": False, "error": f"Invalid status '{status}'. Valid: active, done, obsolete"}

    plans_dir = _get_plans_dir()
    # Try exact match first
    exact = plans_dir / f"{name}.md"
    target = None
    if exact.exists():
        target = exact
    else:
        matches = sorted(plans_dir.glob(f"{name}_*.md"), reverse=True)
        if matches:
            target = matches[0]

    if not target:
        return {"updated": False, "error": f"No plan found matching '{name}'"}

    content = target.read_text()
    meta, body = _parse_frontmatter(content)
    meta["status"] = status
    if status == "done":
        meta["completed"] = datetime.datetime.now().isoformat()
    elif status == "active":
        meta["completed"] = "~"

    new_frontmatter = "---\n" + "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n---\n\n"
    target.write_text(new_frontmatter + body)

    return {"updated": True, "file": str(target), "name": target.stem, "status": status}


@tool(
    name="read_session_plan",
    description=(
        "Read the full contents of a saved session plan by name.\n"
        "Provide the plan name as returned by list_session_plans (e.g., 'REFACTOR_20260626_120000').\n"
        "If multiple plans match the name prefix, the most recent one is returned."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The full name or prefix of the plan to read (e.g., 'REFACTOR_20260626_120000' or 'REFACTOR')."
            },
        },
        "required": ["name"],
    },
)
def read_session_plan_handler(args):
    """Read the contents of a single session plan by name or prefix."""
    name = args.get("name", "").strip().upper().replace(" ", "_")
    if not name:
        return {"found": False, "error": "Plan name is required"}

    plans_dir = _get_plans_dir()
    # Try exact match first
    exact = plans_dir / f"{name}.md"
    if exact.exists():
        content = exact.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(content)
        return {"found": True, "name": name, "file": str(exact), "content": content, "status": meta.get("status", "unknown")}

    # Try prefix match — find all files starting with name
    matches = sorted(plans_dir.glob(f"{name}_*.md"), reverse=True)
    if matches:
        f = matches[0]
        content = f.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(content)
        return {"found": True, "name": f.stem, "file": str(f), "content": content, "status": meta.get("status", "unknown")}

    return {"found": False, "error": f"No plan found matching '{name}'"}


@tool(
    name="update_session_plan_content",
    description=(
        "Update the content of an existing session plan while preserving its frontmatter "
        "(name, created, status, completed). "
        "Provide the plan name as returned by list_session_plans."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The full name or prefix of the plan to update (e.g., 'REFACTOR_20260626_120000' or 'REFACTOR')."
            },
            "content": {
                "type": "string",
                "description": "The new markdown content for the plan body (frontmatter is preserved automatically)."
            },
        },
        "required": ["name", "content"],
    },
)
def update_session_plan_content_handler(args):
    """Update the body content of a session plan, preserving its frontmatter."""
    name = args.get("name", "").strip().upper().replace(" ", "_")
    content = args.get("content", "")
    if not name:
        return {"updated": False, "error": "Plan name is required"}
    if not content:
        return {"updated": False, "error": "Plan content is required"}

    plans_dir = _get_plans_dir()
    # Try exact match first
    exact = plans_dir / f"{name}.md"
    target = None
    if exact.exists():
        target = exact
    else:
        matches = sorted(plans_dir.glob(f"{name}_*.md"), reverse=True)
        if matches:
            target = matches[0]

    if not target:
        return {"updated": False, "error": f"No plan found matching '{name}'"}

    old_content = target.read_text(encoding="utf-8")
    meta, _ = _parse_frontmatter(old_content)

    # Rebuild with preserved frontmatter + new body
    new_frontmatter = "---\n" + "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n---\n\n"
    target.write_text(new_frontmatter + content, encoding="utf-8")

    return {"updated": True, "file": str(target), "name": target.stem, "status": meta.get("status", "unknown")}


@tool(
    name="save_note",
    description=(
      "Save a persistent note.\n"
      "Notes are stored persistently and survive across sessions. "
      "Use notes to remember things long-term. Notes are not added to the context, but"
      "can be retrieved with the tool 'get_memory'"
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Note content"},
            "category": {"type": "string", "description": "Note category (default: general)"},
        },
        "required": ["content"],
    },
)
def save_note_handler(args):
    """Save a note to persistent memory."""
    mem = Memory()
    note = mem.add_note(args.get("content", ""), category=args.get("category", "general"))
    return {"saved": True, "note_id": note["id"], "category": note["category"]}

@tool(
    name="save_memory",
    description=(
        "Explicitly persist all memory to disk.\n"
        "Call this after making changes to memory (e.g., saving notes)"
        " to ensure they are saved."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)
def save_memory_handler(args):
    """Explicitly persist all memory to disk."""
    mem = Memory()
    mem.save()
    return {"saved": True, "message": "Memory persisted to disk"}

@tool(
    name="get_memory",
    description=(
        "Read the agent's current memory\n"
        "Use this when you need to access long-term memory, like "
        "persistent notes or the user profile."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)
def get_memory_handler(args):
    """Read the agent's current memory (profile + notes)."""
    mem = Memory()
    text = mem.get_formatted()
    if text:
        return {"memory": text}
    return {"memory": None, "message": "No memory yet"}

@tool(
    name="set_user_profile",
    description=(
        "Set the user profile with key-value pairs.\n"
        "Call save_memory after to persist to disk."
    ),
    parameters={
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "Dictionary of profile key-value pairs",
            },
        },
        "required": ["data"],
    },
)
def set_user_profile_handler(args):
    """Set the user profile. Call save_memory after to persist."""
    mem = Memory()
    # Arguments can be a dict of key-value pairs, or a 'data' key with a dict
    if "data" in args and isinstance(args["data"], dict):
        profile = args["data"]
    else:
        profile = {k: v for k, v in args.items() if k != "save"}
    mem.set_user_profile(profile)
    return {"saved": True, "profile": profile}

@tool(
    name="get_session_info",
    description=(
        "Get the session name, working directory, and session directory (configuration and agent data).\n"
        "Use this if you are unsure of where the current project files are."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)
def get_session_info_handler(args):
    """Get information on the current session."""
    mem = Memory()
    return {
        "session_name": mem.session,
        "working_directory": os.getcwd(),
        "session_directory": str(mem.session_dir)
    }
