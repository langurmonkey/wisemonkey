"""
Centralized slash command registry.

Contains all slash commands and the scaffolding necessary to register
and execute them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable

from rich.prompt import Prompt
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.formatted_text import HTML

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
        for words in reversed(range(1, n+1)):
            name = '-'.join(tokens[i] for i in range(words))
            command = self._commands.get(name.lower())
            if command:
                return command, tokens[words:]

        return None, None

    def execute(self, agent, cmd: Command, params: list[str]):
        """
        Execute a command.

        Returns:
        - ok: bool          - status of the operation
        - msg: str          - short message with an informative message
        - content: str      - long text in python rich format
        - markdown: str     - long text in Markdown format
        - should_exit: bool - boolean indicating whether the agent must exit
        """
        if (
            params and
            (params[0].lower() == "-h" or params[0].lower() == "help")
        ):
            return True, None, self._command_str(cmd), None, False
                
        ok, msg, content, markdown = cmd.handler(agent, params)
        should_exit = msg in ("EXIT", "exit")
        return ok, msg, content, markdown, should_exit

    def run_command(self, agent, command: str):
        """
        Shortcut to run a command from a string.

        Returns:
        - ok: bool          - status of the operation
        - msg: str          - short message with an informative message
        - content: str      - long text in python rich format
        - markdown: str     - long text in Markdown format
        - should_exit: bool - boolean indicating whether the agent must exit
        """
        cmd, params = self.lookup(command.split())
        if cmd:
            return self.execute(agent, cmd, params)
        else:
            raise RuntimeError(f"command not found: {command}")


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
        result += f"• {names} → {cmd.description}\n"

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
    """
    Decorator to register commands.

    Commands return:
    - ok:bool      - False if there was an error
    - msg:str      - A one-line status message
    - content:str  - Multi-line content in rich formatting
    - markdown:str - Multi-line content in markdown
    """

    def command(handler):
        registry.register(Command(name,
                                  description,
                                  handler,
                                  aliases,
                                  examples))

    return command


# Built-in command handlers
@cmd(
      "/quit",
      "Exit the agent",
      aliases=["/exit", "/q"]
)
def _cmd_quit(agent, params):
    return True, "EXIT", None, None


@cmd(
      "/reasoning",
       "Configure model reasoning",
)
def _cmd_reasoning(agent, params):
    from agent.config import get_config

    if params:
        return False, "this command does not take any arguments", None, None

    try:
        config = get_config()

        # Reasoning effort
        opts = [
            ("high", "High"),
            ("medium","Medium"),
            ("low","Low"),
            ("none", "Disable model reasoning"),
            ]
        defa = config.get("model.reasoning_effort", "medium")

        effort = choice(
            message="Choose the reasoning effort:",
            options=opts,
            default=defa,
            bottom_toolbar=HTML(
                " <b>↑</b>/<b>↓</b>: select | <b>Enter</b>: accept"
            ),
        )
        config.set("model.reasoning_effort", effort)

        # Reasoning visible
        opts = [("true", "Yes"), ("false", "No")]
        defa = config.get("model.reasoning_visible")

        visible = choice(
            message="Display reasoning:",
            options=opts,
            default=defa,
            bottom_toolbar=HTML(
                " <b>↑</b>/<b>↓</b>: select | <b>Enter</b>: accept"
            ),
        )
        visible_bool = visible == "true"
        console.print(f"{visible}, {visible_bool}")
        config.set("model.reasoning_visible", visible_bool)
    except Exception as e:
        console.print(e)

    return True, f"reasoning effort: {effort}, show reasoning: {visible}", None, None


@cmd(
      "/notes",
      "List all notes",
)
def _cmd_notes(agent, params):
    notes = agent.core.memory.get_notes()
    buff = ""
    for note in notes:
        buff += f"📋︎  [blue]{note['id']}[/blue] ({note['category']}):\n"
        buff += f"[grey39]{note['content']}[/]\n\n"
    return True, None, buff, None

@cmd(
      "/notes-add",
      "Add a note to memory",
      examples=[
          "/notes add This is my note   # Add a new note",
      ]
)
def _cmd_notes_add(agent, params):
    if params:
        agent.core.memory.add_note(" ".join(params))
        return True, "note added successfully", None, None
           
    return False, "please, provide a note", None, None

@cmd(
    "/session",
    "Print session information",
)
def _cmd_session(agent, params):
    if params:
        return False, "this command does not take any arguments", None, None

    from agent.utils import contractuser
    mem = agent.core.memory
    name = mem.session
    session_dir = mem.session_dir
    created = mem.session_created
    accessed = mem.session_accessed
    result = ""
    result += f"Name:           [accent-bold]{name}[/accent-bold]\n"
    result += f"Location:       {contractuser(session_dir)}\n"
    result += f"Created:        {created}\n"
    result += f"Last accessed:  {accessed}"
    return True, None, result, None
    
    
@cmd(
    "/session-clear",
    "Clear the current session chat memory",
    examples=[
        "/session clear        # Clear chat memory for this sesson",
        "/session clear 10     # Clear the 10 oldest chat exchanges of this session",
    ]
)
def _cmd_session_clear(agent, params):
    n = 0
    if params:
        try:
            n = int(params[0])
        except ValueError:
            return False, f"the parameter must be an integer: '{params[0]}'", None, None

    cleared = agent.core.memory.clear_chat(n)
    return True, f"{cleared} exchanges cleared", None, None
    
def _chat_memory(core, max_exchanges):
    return core.memory.get_chat_formatted(max_exchanges, timestamps=True)

@cmd(
      "/session-agent",
      "Show the session agent memory contents (user profile and notes)",
)
def _cmd_session_agent(agent, params):
    if params:
        return False, "this command does not take any arguments", None, None

    mem =  agent.core.memory.get_formatted()
    # Format in Markdown
    return True, None, None, mem

@cmd(
      "/session-chat",
      "Show the session chat memory contents",
)
def _cmd_session_chat(agent, params):
    if params:
        return False, "this command does not take any arguments", None, None

    mem =  agent.core.memory.get_formatted()
    chars, max, rate = agent.core.memory.get_chat_formatted(0)
    stats = f"Memory status: {chars}/{max} ({rate:.2f}%)"
    # Format in Markdown
    return True, stats, None, mem

@cmd(
    "/session-compact",
    "Compact the session chat history by summarizing it into a shorter form."   
)
def _cmd_session_compact(agent, params):
    if params:
        return False, "this command does not take any arguments", None, None

    history_text = agent.core.memory.get_chat_formatted()
    len_before = len(history_text) if history_text else 0

    content = (
        "You are a technical editor. Summarize and compact this conversation as a dense agent briefing. Follow these guidelines:\n"
        "- Include what was being worked on\n"
        "- Remove greetings, small talk, pleasantries, repeated interactions, and filler words\n"
        "- Remember files created or modified and how\n"
        "- Preserve file names, technical constraints, and logical reasoning\n"
        "- Add facts worth remembering long term\n"
        "- Output format is markdown, use bullet points if needed\n"
        "CONVERSATION:\n"
    )
    content += history_text
    
    messages=[{
        "role": "user",
        "content": content
    }]

    try:
        spinner_compact = console.status("⏳ Compacting chat history...")
        spinner_compact.start()

        response = agent.core.llm_chat_raw(messages)

        spinner_compact.stop()

        summary = response.choices[0].message.content
        len_after = len(summary)

        agent.core.memory.reset_chat_memory(content=[{"role": "summary", "content": summary}])

        return True, f"memory compacted successfully from {len_before} to {len_after}", None, None
    except Exception as e:
        return False, f"memory compact operation failed: {e}", None, None


@cmd(
      "/tools",
      "List available tools ⚙",
)
def _cmd_tools(agent, params):
    from agent.tools import get_tools_str
    return True, None, get_tools_str(), None


@cmd(
      "/skills",
      "List loaded skills ⚔",
)
def _cmd_skills(agent, params):
    return True, None, agent.core.skills.get_skills_str(), None

@cmd(
      "/models",
      "Configure the model to use."
)
def _cmd_models(agent, params):
    try:
        models = agent.core.get_models()
    except Exception as e:
        return False, f"{e}", None, None
        
    opts = [(f"{model.id}", f"{model.id}") for (_, model) in enumerate(models)]
    defa = models.data[0].id

    result = choice(
        message="Choose a model:",
        options=opts,
        default=defa,
        bottom_toolbar=HTML(
            " <b>↑</b>/<b>↓</b>: select | <b>Enter</b>: accept"
        ),
    )

    try:
        agent.core.set_model(result)
    except NameError as e:
        return False, f"{e}", None, None
        
    return True, f"model: {result}", None, None
        
@cmd(
      "/config",
      "Configure the agent",
)
def _cmd_config(agent, params):
    from agent.config import get_config
    if params:
        return False, "this command does not take any parameters", None, None

    config = get_config()
    base_url = config.get("model.base_url")

    new_url = Prompt.ask("Base URL", default=base_url or "")

    config.set("model.base_url", new_url)

    ok, msg = agent.core.initialize_client()
    if not ok:
        return False, f"{msg}", None, None

    return True, "configuration saved successfully", None, None
    

@cmd(
      "/config-list",
       "Show current configuration",
)
def _cmd_config_list(agent, params):
    from agent.config import log_config
    return True, None, log_config(), None

@cmd(
      "/config-set",
       "Set configuration values",
       examples=[
           "/config set model.reasoning_effort low     [dim]# set reasoning effort to low[/dim]",
           "/config set model.reasoning_visible False  [dim]# do not show reasoning[/dim]",
           "/config set agent.temperature 0.8          [dim]# set temperature[/dim]"
       ]
)
def _cmd_config_set(agent, params):
    if params:
        # Set configuration values
        if len(params) != 2:
            return False, "/config needs two parameters: key and value", None, None
        else:
            from agent.config import get_config
            config = get_config()
            key = params[0]
            value = params[1]
            if config.has(key):
                t = type(config.get(key))
                new_value = smart_cast(value, t)
                config.set(key, new_value)

                if hasattr(agent.core, key):
                    setattr(agent.core, key, new_value)

                return True, f"{key}: {value}", None, None
            
            else:
                return False, f"Key '{key}' does not exist in the configuration", None, None
    else:
        return False, "/config needs two parameters: key and value", None, None
            

@cmd(
      "/vi",
       "Enable or disable vi input mode",
       examples=["/vi (on|off)"]
)
def _cmd_vi(agent, params):
    if params:
        state = params[0].lower() == "on"
        ok, msg, _, _, _ = registry.run_command(agent, "/config-set agent.vi_mode " + str(state))
        if ok:
            agent._create_prompt_session()
        return ok, msg, None, None
    return False, "/vi command needs a parameter (on/off)", None, None

@cmd(
    "/help",
    "Show command help",
    aliases=["/commands"],
)
def _cmd_help(agent, params):
    return True, None, registry.get_commands_str(), None

