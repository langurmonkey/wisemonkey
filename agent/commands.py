"""Centralized slash command registry for langur-agent."""

from __future__ import annotations

import ast
from rich.prompt import Prompt
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent.console import console

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


class CommandRegistry:
    """Module-level singleton registry for slash commands."""

    def __init__(self):
        self._commands: dict[str, Command] = {}  # primary name -> Command

    def register(self, cmd: Command) -> None:
        """Register a command and all its aliases."""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd  # alias points to same Command

    def lookup(self, tokens: list[str]):
        """Look up a command by name or alias (case-insensitive)."""
        if not tokens:
            return None, None

        n = len(tokens)
        for words in range(1, n+1):
            name = '-'.join(tokens[i] for i in range(words))
            command = self._commands.get(name.lower())
            if command:
                return command, tokens[words:]

        return None, None

    def execute(self, agent, cmd: Command, params: list[str]):
        """Execute a command. Returns (result_string | None, should_exit)."""
        if (
            params and
            (params[0].lower() == "-h" or params[0].lower() == "help")
        ):
            msg = self._command_str(cmd)
            return msg, False
                
        result = cmd.handler(agent, params)
        should_exit = result in ("EXIT", "exit")
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
        primary = f"[green]{cmd.name.replace('-', ' ')}[/]"
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
def cmd(name: str,
        description: str = "",
        aliases: list[str] = [],
        examples: list[str] = []):
    """Decorator to register commands."""

    def command(handler):
        registry.register(Command(name,
                                  description,
                                  handler,
                                  aliases,
                                  examples))

    return command


# --- Built-in command handlers ---
@cmd(
      "/quit",
      "Exit the agent",
      aliases=["/exit", "/q"]
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
        cmd, pars = registry.lookup(["/config-set", "model.show_thinking", str(state)])
        result, _ = registry.execute(agent, cmd, pars)
        return result
    return "[red]ERROR:[/] /showthinking command needs a parameter (on/off)"


@cmd(
      "/notes-list",
      "List all notes",
)
def _cmd_notes_list(agent, params):
    notes = agent.memory.get_notes()
    buff = ""
    for note in notes:
        buff += f"📋︎  [blue]{note['id']}[/blue] ({note['category']}):\n"
        buff += f"[grey39]{note['content']}[/]\n\n"
    return buff

@cmd(
      "/notes-add",
      "Add a note to memory",
      examples=[
          "/notes add This is my note   # Add a new note",
      ]
)
def _cmd_notes_add(agent, params):
    if params:
        agent.memory.add_note(" ".join(params))
        return "[green]OK:[/] note added successfully"
           
    return "[red]ERROR:[/] please, provide a note"

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
        subcommand = params[0].lower()
        if subcommand == "agent":
            return _agent_memory(agent)
        elif subcommand == "chat":
            return _chat_memory(agent)
        else:
            return f"[red]ERROR:[/] unrecognized subcommand '{subcommand}'"
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
      "/models",
      "Set the model to use."
)
def _cmd_models(agent, params):
    models = agent.get_models()
    pr = ""
    idx = -1
    for (i, model) in enumerate(models):
        pr += f"{i}: [green]{model.id}[/]\n"
        idx = i
    console.print(pr)

    console.print()
    index_str = Prompt.ask("Choose a model", choices=[f"{i}" for i in range(idx+1)], default='0', case_sensitive=False)
    
    model = models.data[int(index_str)]

    try:
        agent.set_model(model.id)
    except NameError as e:
        return f"[red]ERROR:[/] {e}"
        
    return f"[green]OK:[/] model: {model.id}"
        

@cmd(
      "/config-list",
       "Show current configuration",
)
def _cmd_config_list(agent, params):
    from agent.config import log_config
    return log_config()

@cmd(
      "/config-set",
       "Set configuration values",
       examples=[
           "/config set model.show_thinking False  [dim]# do not show reasoning[/dim]",
           "/config set agent.temperature 0.8      [dim]# set temperature[/dim]"
       ]
)
def _cmd_config_set(agent, params):
    if params:
        # Set configuration values
        if len(params) != 2:
            return "[red]ERROR:[/] /config needs two parameters: key and value"
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
        return f"[red]ERROR[/] /config needs two parameters: key and value"
            

@cmd(
      "/vi",
       "Enable or disable vi mode input",
       examples=["/vi (on|off)"]
)
def _cmd_vi(agent, params):
    if params:
        state = params[0].lower() == "on"
        cmd, pars = registry.lookup(["/config-set", "agent.vi_mode", str(state)])
        result, _ = registry.execute(agent, cmd, pars)
        if result.startswith("[green]OK"):
            agent._create_prompt_session()
        return result
    return "[red]ERROR:[/] /vi command needs a parameter (on/off)"

@cmd(
    "/help",
    "Show command help",
    aliases=["/commands"],
)
def _cmd_help(agent, params):
    return registry.get_commands_str()

