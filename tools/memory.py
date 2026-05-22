"""Memory tool.

Allows the agent to operate on memory:

- Load/save memory
- Add notes
- Set user profile
"""
from agent.memory import Memory
from agent.tools import tool

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
        "Use this when you need to acces long-term memory, like "
        "persistent notes or the user profile.",
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
    # args can be a dict of key-value pairs, or a 'data' key with a dict
    if "data" in args and isinstance(args["data"], dict):
        profile = args["data"]
    else:
        profile = {k: v for k, v in args.items() if k != "save"}
    mem.set_user_profile(profile)
    return {"saved": True, "profile": profile}
