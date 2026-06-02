"""Basic tools.

Basic example tools for Wisemonkey.

Contains 2 tools:

- Echo: echoes a message
- List-skills: returns a list with all the skills
"""

from agent.tools import tool

@tool(
    name="echo",
    description="Echo back the provided text.\nUseful for testing.",
    parameters={
      "type": "object",
      "properties": {
          "text": {"type": "string", "description": "Text to echo back"},
      },
      "required": ["text"],
    })
def echo_handler(args):
    """Echo back the input text."""
    return {"result": args.get("text", "")}

@tool(
    name="datetime",
    description="Get the current date and time in the format '2025-10-25 07:18:28.213080'. Without parameters, it returns the time in the user's time zone. You can also pass in a time zone string to get the time in that timezone.",
    parameters={
      "type": "object",
      "properties": {
          "timezone": {
              "type": "string",
              "description": "Time zone string in the format of Python's ZoneInfo: 'UTC', 'Asia/Tokyo', 'Europe/Berlin', etc."
          }
      },
      "required": [],
    })
def get_datetime(args):
    """Get current date and time."""
    import datetime
    if args:
        from zoneinfo import ZoneInfo
        return {"result": str(datetime.datetime.now(tz=ZoneInfo(args.get("timezone"))))}
        
    else:
        return {"result": str(datetime.datetime.now())}

@tool(
    name="list_skills",
    description="List all available skills loaded by the agent.",
    parameters={
        "type": "object",
        "properties": {},
    },
)
def list_skills_handler(args):
    """List all available skills."""
    from agent.skills import SkillLoader
    loader = SkillLoader()
    skills = loader.list_skills()
    return {"skills": skills}

