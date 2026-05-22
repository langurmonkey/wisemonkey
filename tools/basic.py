"""Basic tools.

Basic example tools for the Langur agent.

Contains 2 tools:

- Echo: echoes a message
- List-skills: returns a list with all the skills
"""

from agent.tools import tool, register_tool

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
    name="list_skills",
    description="List all available skills loaded by the agent.",
    parameters={
        "type": "object",
        "properties": {},
    },
)
def list_skills_handler(args):
    """List all available skills."""
    from langur.skills import SkillLoader
    loader = SkillLoader()
    skills = loader.list_skills()
    return {"skills": skills}

