"""Centralized slash command registry for langur-agent."""

from __future__ import annotations

import ast
from rich import print
from dataclasses import dataclass, field
from typing import Callable, Optional


def smart_cast(value, target_type):
    """Cast a string value to a given type, handling booleans correctly."""
    if target_type is bool:
        return ast.literal_eval(value)   # 'False' → False
    return target_type(value)            # '42' → 42, '3.14' → float, etc.

@dataclass(frozen=True)
class Command:
    """A single slash command definition."""
    name: str
    description: str = ""
    handler: Callable = None  # (agent, params: list[str]) -> str | None
    aliases: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    can_complete: bool = False


class CommandRegistry:
    """Module-level singleton registry for slash commands."""

    def __init__(self):
        self._commands: dict[str, Command] = {}  # primary name -> Command

    def register(self, cmd: Command) -> None:
        """Register a command and all its aliases."""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd  # alias points to same Command

    def lookup(self, name: str) -> Optional[Command]:
        """Look up a command by name or alias (case-insensitive)."""
        return self._commands.get(name.lower())

    def execute(self, agent, name: str, params: list[str]):
        """Execute a command. Returns (result_string | None, should_exit)."""
        cmd = self.lookup(name)
        if cmd is None:
            return None, False

        if (
            params and
            (params[0].lower() == "-h" or params[0].lower() == "help")
        ):
            msg = self._command_str(cmd)
            return msg, False
                
        result = cmd.handler(agent, params)
        should_exit = name.lower() in ("/quit", "/exit", "/q")
        return result, should_exit

    def list_commands(self) -> list[Command]:
        """Return all unique commands (deduplicated by primary name)."""
        seen = set()
        result = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd)
        return result

    def names(self) -> list[str]:
        """Return all command names (primary + aliases) for completion."""
        return list(self._commands.keys())

    def get_commands_str(self):
        """ Prints all defined commands to the output"""
        all = ""
        for cmd in self.list_commands():
            all += f"{self._command_str(cmd)}"

        return all

    def _command_str(self, cmd):
        aliases = ", ".join(f"[green]{a}[/]" for a in cmd.aliases)
        primary = f"[green]{cmd.name}[/]"
        if aliases:
            names = f"{primary}, {aliases}"
        else:
            names = primary
        result = ""
        result += f"⬤ {names} → {cmd.description}\n"

        if cmd.examples:
            result += f"[grey50]  Examples:[/]\n"

        for example in cmd.examples:
            result += f"[grey30]    {example}[/]\n"

        return result
        


# Module-level singleton
registry = CommandRegistry()

# Decorator for commands
def cmd( name: str,
        description: str = "",
        aliases: list[str] = [],
        examples: list[str] = [],
        can_complete: bool = False):
    """Decorator to register commands."""

    def command(handler):
        registry.register(Command(name,
                                  description,
                                  handler,
                                  aliases,
                                  examples,
                                  can_complete))

    return command


# --- Built-in command handlers ---

@cmd(
      "/quit",
      "Exit the agent",
      aliases=["/exit", "/q"]))
)
def _cmd_quit(agent, params):
    return "EXIT"


@cmd(
      "/showthinking",
       "Show or hide the model thinking tokens",
       examples=["/showthinking (on|off)"]
)
def _cmd_showthinking(agent, params):
    if params:
        state = params[0].lower() == "on"
        return _cmd_config(agent, ["model.show_thinking", state])
    return "[red]ERROR:[/] /showthinking command needs a parameter (on/off): '/showthinking on', '/showthinking off'"


@cmd(
      "/note",
      "Save a note to memory",
      examples=["/note Sometimes the wind blows from the East"]
)
def _cmd_note(agent, params):
    if params:
        agent.memory.add_note(" ".join(params))
        return "[green]OK:[/] note added successfully"
    return "[red]ERROR:[/] please, provide a note: '/note This is my note'"


@cmd(
      "/notes",
      "List all notes",
)
def _cmd_notes(agent, params):
    notes = agent.memory.get_notes()
    return "\n".join(f"⬤ [blue]{note['id']}[/blue] ({note['category']}): {note['content']}" for note in notes)

def _agent_memory(agent):
    return f"[red]===== AGENT MEMORY =====[/red]\n\n{agent.memory.get_formatted()}\n\n"

def _chat_memory(agent):
    chat = agent.memory.get_formatted_chat()
    chars = len(chat)
    max = agent.config.get("agent.max_chat_history")
    chat = f"[red]===== CHAT MEMORY ===== [/red]\n\n{chat}\n\n[red underline]CHAT MEM SIZE:[/red underline] {chars}/{max} ({float(chars) * 100.0/float(max):.2f}%)\n\n"
    return chat
    

@cmd(
      "/memory",
      "List memory contents",
      examples=[
          "/memory  # list all contents",
          "/memory agent [dim]# only list agent memory (profile and notes)[/dim]",
          "/memory chat  [dim]# only list chat memory[/dim]"
      ]
)
def _cmd_memory(agent, params):
    if params:
        if params[0].lower() == "agent":
            return _agent_memory(agent)
        elif params[0].lower() == "chat":
            return _chat_memory(agent)
        else:
            return f"[red]ERROR:[/] unrecognized subcommand '{params[0]}', only 'agent' and 'chat' accepted"
    else:
        # Print all
        return _agent_memory(agent) + _chat_memory(agent)


@cmd(
      "/tools",
      "List available tools ⚙",
)
def _cmd_tools(agent, params):
    from agent.tools import get_tools_str
    return get_tools_str()


@cmd(
      "/skills",
      "List loaded skills ⚔",
)
def _cmd_skills(agent, params):
    return agent.skills.get_skills_str()


@cmd(
      "/config",
       "Set configuration values",
       examples=[
           "/config  [dim]# list configuration[/dim]",
           "/config model.show_thinking False  [dim]# do not show reasoning[/dim]",
           "/config agent.temperature 0.8      [dim]# set temperature[/dim]"
       ]
)
def _cmd_config(agent, params):
    if params:
        # Set configuration values
        if len(params) != 2:
            return "[red]ERROR:[/] /config needs either zero or two parameters: key and value"
        else:
            from agent.config import get_config
            config = get_config()
            key = params[0]
            value = params[1]
            if config.has(key):
                t = type(config.get(key))
                new_value = smart_cast(value, t)
                config.set(key, new_value)

                if hasattr(agent, key):
                    setattr(agent, key, new_value)

                return f"[green]OK:[/] {key}: {value}"
                
            else:
                return f"[red]ERROR:[/] Key '{key}' does not exist in the configuration"
            
    else:
        from agent.config import log_config
        log_config()
        return None


@cmd(
      "/vi",
       "Enable or disable vi mode input",
       examples=["/vi (on|off)"]))
)
def _cmd_vi(agent, params):
    if params:
        state = params[0].lower() == "on"
        result = _cmd_config(agent, ["agent.vi_mode", str(state)])
        if result.startswith("[green]OK"):
            agent._create_prompt_session()
        return result
    return "[red]ERROR:[/] /vi command needs a parameter (on/off): '/vi on', '/vi off'"

@cmd(
    "/help",
    "Show command help",
    aliases=["/commands"],
    can_complete=True
)
def _cmd_help(agent, params):
    return registry.get_commands_str()

