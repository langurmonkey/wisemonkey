"""
Centralized slash command registry.

Contains all slash commands and the scaffolding necessary to register
and execute them.
"""

from __future__ import annotations

import ast
import os

from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from pubsub import pub

from agent.console import console, err
from agent.utils import resize_image, PromptUi

# Global error messages for commands
no_params_error = "This command does not take any parameters"

def smart_cast(value, target_type):
    """Cast a string value to a given type, handling booleans correctly."""
    if target_type is bool:
        return ast.literal_eval(value)   # 'False' → False
    return target_type(value)            # '42' → 42, '3.14' → float, etc.

def empty():
    pass

@dataclass(frozen=True)
class Command:
    """A single slash command definition."""
    name: str
    description: str = ""
    handler: Callable = empty # (agent, params: list[str], prompt_ui: PromptUi | None) -> str | None
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

    def lookup(self, tokens: list[str]) -> tuple[Command | None, list[str] | None]:
        """
        Look up a command by name or alias (case-insensitive).

        Returns:
        - command: Command or None     - command instance
        - tokens: list[str] or None    - command arguments as a token list
        -
        """
        if not tokens:
            return None, None

        n = len(tokens)
        for words in reversed(range(1, n+1)):
            name = '-'.join(tokens[i] for i in range(words))
            command = self._commands.get(name.lower())
            if command:
                return command, tokens[words:]

        return None, None

    def execute(self, core, cmd: Command, params: list[str] | None, prompt_ui: PromptUi | None = None) -> tuple[bool, str | None, str | None, str | None, bool]:
        """
        Execute a command.

        Parameters
        ----------
        core : Core
            The agent core instance.
        cmd : Command
            The command to execute.
        params : list[str] | None
            Command arguments.
        prompt_ui : PromptUi | None
            UI abstraction for interactive prompts.  When *None*, commands
            that need user input will fall back to the classic rich.prompt
            behaviour.

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

        ok, msg, content, markdown = cmd.handler(core, params, prompt_ui)
        should_exit = msg in ("EXIT", "exit")
        return ok, msg, content, markdown, should_exit

    def run_command(self, core, command: str, prompt_ui: PromptUi | None = None) -> tuple[bool, str | None, str | None, str | None, bool]:
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
            return self.execute(core, cmd, params, prompt_ui)
        else:
            raise RuntimeError(f"command not found: {command}")


    def list_commands(self) -> list[Command]:
        """
        Return all unique commands (deduplicated by primary name) and
        in alphabetical order.
        """
        import collections
        seen = set()
        result = []
        commands = collections.OrderedDict(sorted(self._commands.items()))
        for cmd in commands.values():
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


def _fallback_prompt_ui() -> PromptUi:
    """Return a RichPromptUi for when no prompt_ui is explicitly provided."""
    from agent.prompt_ui import RichPromptUi
    return RichPromptUi()


# Built-in command handlers
@cmd(
      "/quit",
      "Exit the agent",
      aliases=["/exit", "/q"]
)
def _cmd_quit(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    return True, "EXIT", None, None


@cmd(
      "/reasoning",
       "Configure model reasoning",
)
def _cmd_reasoning(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:

    if params:
        return False, no_params_error, None, None

    ui = prompt_ui or _fallback_prompt_ui()
    from agent.config import get_config
    config = get_config()

    try:
        # Reasoning effort
        opts = [
            ("high", "High"),
            ("medium","Medium"),
            ("low","Low"),
            ("none", "Disable model reasoning"),
            ]
        defa = core.router.thinking_effort

        effort = ui.ask_choice(
            message="Choose the reasoning effort:",
            options=opts,
            default=defa,
        )
        core.router.thinking_effort = effort

        # Reasoning display
        opts = [("true", "Yes"), ("false", "No")]
        defa = config.get("model.thinking.display")

        visible = ui.ask_choice(
            message="Display reasoning:",
            options=opts,
            default=str(defa).lower(),
        )
        visible_bool = visible == "true"
        config.set("model.thinking.display", visible_bool)
    except Exception as e:
        err(e)

    return True, f"reasoning effort: {effort}, show reasoning: {visible}", None, None


@cmd(
      "/notes",
      "List all notes",
)
def _cmd_notes(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    notes = core.memory.get_notes()
    buff = ""
    for note in notes:
        buff += f"📋️  [blue]{note['id']}[/blue] ({note['category']}):\n"
        buff += f"[grey39]{note['content']}[/]\n\n"
    return True, None, buff, None

@cmd(
      "/notes-add",
      "Add a note to memory",
      examples=[
          "/notes add This is my note   # Add a new note",
      ]
)
def _cmd_notes_add(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        core.memory.add_note(" ".join(params))
        return True, "note added successfully", None, None

    return False, "please, provide a note", None, None

@cmd(
    "/session",
    "Print session information",
)
def _cmd_session(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.utils import contractuser
    mem = core.memory
    name = mem.session
    session_dir = mem.session_dir
    working_dir = contractuser(Path(os.getcwd()))
    created = mem.session_created
    accessed = mem.session_accessed
    result = ""
    result += f"Name:           [accent-bold]{name}[/accent-bold]\n"
    result += f"Location:       {contractuser(session_dir)}\n"
    result += f"Working dir:    {working_dir}\n"
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
def _cmd_session_clear(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    n = 0
    if params:
        try:
            n = int(params[0])
        except ValueError:
            return False, f"the parameter must be an integer: '{params[0]}'", None, None

    cleared = core.memory.clear_chat(n)
    return True, f"{cleared} exchanges cleared", None, None


def _chat_memory(core, max_exchanges):
    return core.memory.get_chat_formatted(max_exchanges, timestamps=True)

@cmd(
      "/session-agent",
      "Show the session agent memory contents (user profile and notes)",
)
def _cmd_session_agent(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    mem =  core.memory.get_formatted()
    if mem:
        # Format in Markdown
        return True, None, None, mem
    else:
        return False, "agent memory is empty", None, None

@cmd(
      "/session-chat",
      "Show the session chat memory contents",
      examples=[
          "/session chat     # Print entire session chat",
          "/session chat 2   # Print last 2 interactions"
      ]
)
def _cmd_session_chat(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    n = 0
    if params:
        try:
            n = int(params[0])
        except ValueError:
            return False, f"Parameter must be integer: {params[0]}", None, None

    mem = core.memory.get_chat_formatted(num_exchanges=n, timestamps=True)
    chars, max, rate = core.memory.get_chat_stats()
    stats = f"Memory status: {chars}/{max} ({rate:.2f}%)"
    # Format in Markdown
    return True, stats, None, mem

@cmd(
    "/session-compact",
    "Compact the session chat history by summarizing it into a shorter form."
)
def _cmd_session_compact(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    history_text = core.memory.get_chat_formatted()
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

    spinner_compact = console.status("⏳ Compacting chat history...")
    try:
        spinner_compact.start()
        response = core.llm_chat_raw(messages)
        spinner_compact.stop()

        summary = response.choices[0].message.content
        len_after = len(summary)

        core.memory.reset_chat_memory(content=[{"role": "summary", "content": summary}])

        return True, f"Memory compacted successfully from {len_before} to {len_after}", None, None
    except Exception as e:
        spinner_compact.stop()
        return False, f"Memory compact operation failed: {e}", None, None


@cmd(
    "/embed",
    "Embed a document into the session vector store",
    examples=[
        "/embed ~/documents/research_paper.pdf",
        "/embed ./notes.md",
    ]
)
def _cmd_embed(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if not params:
        return False, "please provide a file path", None, None

    file_path = " ".join(params)
    spinner_embed = console.status(f"⏳ Embedding: {file_path}...")
    file_path = os.path.expanduser(file_path)

    # Lazily initialize the vector store on first use
    if core.memory.vectorstore is None:
        from agent.memory import _load_vectorstore
        core.memory.vectorstore = _load_vectorstore(core.memory.session_dir)

    if core.memory.vectorstore is None:
        spinner_embed.stop()
        return False, "Vector store is not available. Check that chromadb and tiktoken are installed, and embedding config is correct.", None, None

    try:
        spinner_embed.start()
        count = core.memory.vectorstore.ingest(file_path)
        spinner_embed.stop()
        return True, f"Successfully embedded {count} chunks from '{file_path}'", None, None
    except Exception as e:
        spinner_embed.stop()
        return False, f"Embedding failed: {e}", None, None


@cmd(
      "/tools",
      "List all available tools ⚙",
)
def _cmd_tools(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str
    return True, None, get_tools_str(), None

@cmd(
      "/tools-native",
      "List native tools ⚙",
)
def _cmd_tools_native(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str
    return True, None, get_tools_str(prefix="mcp_", contains=False), None

@cmd(
      "/tools-mcp",
      "List MCP tools ⚙",
)
def _cmd_tools_mcp(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str
    return True, None, get_tools_str(prefix="mcp_", contains=True), None

@cmd(
      "/skills",
      "List loaded skills ⚔",
)
def _cmd_skills(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    return True, None, core.skills.get_skills_str(), None

@cmd(
      "/model",
      "Configure the model to use",
      aliases=["/models"]
)
def _cmd_models(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    ui = prompt_ui or _fallback_prompt_ui()
    from agent.config import get_config
    from agent.router import Provider
    config = get_config()

    # Provider selection
    current_provider = config.get("model.provider", "")
    provider_opts = [
        (Provider.OPENAI.value, "OpenAI"),
        (Provider.ANTHROPIC.value, "Anthropic"),
        (Provider.OLLAMA.value, "Ollama"),
        (Provider.LMSTUDIO.value, "LM Studio"),
        (Provider.GENERIC.value, "Generic (OpenAI-compatible)"),
    ]

    selected_provider = ui.ask_choice(
        message="Choose a provider:",
        options=provider_opts,
        default=current_provider if current_provider else Provider.GENERIC.value,
    )
    config.set("model.provider", selected_provider)

    # Reinitialize router with new provider
    ok, msg = core.initialize_router()
    if not ok:
        return False, f"Failed to initialize {selected_provider}: {msg}", None, None


    # Model selection
    try:
        models = core.get_models()
    except Exception as e:
        return False, f"{e}", None, None

    if not models:
        return True, f"Provider set to {selected_provider} (no models listed)", None, None

    opts = [(m["id"], m["id"]) for m in models]
    defa = core.router.model_name

    result = ui.ask_choice(
        message="Choose a model:",
        options=opts,
        default=defa,
    )

    try:
        success = core.set_model(result)
        if success:
            pub.sendMessage("prompt-update")
            return True, f"Provider: [accent]{selected_provider}[/accent]— Model: [accent]{result}[/accent]", None, None
        else:
            return False, "Model could not be set", None, None
    except NameError as e:
        return False, f"{e}", None, None



@cmd(
      "/url",
      "Configure the base URL",
)
def _cmd_url(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    ui = prompt_ui or _fallback_prompt_ui()
    from agent.config import get_config
    config = get_config()
    base_url = config.get("model.base_url")

    new_url = ui.ask_string(" Enter the endpoint URL", default=base_url or "")

    config.set("model.base_url", new_url)

    ok, msg = core.initialize_router()
    if not ok:
        return False, f"{msg}", None, None

    return True, "Configuration saved successfully", None, None


@cmd(
      "/config-show",
       "Show current configuration",
)
def _cmd_config_show(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import log_config
    return True, None, log_config(), None

@cmd(
      "/config-edit",
       "Edit the configuration file with $EDITOR or $VISUAL",
)
def _cmd_config_edit(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import edit_base_config_visual
    result = edit_base_config_visual()
    ok = result.returncode == 0
    if ok:
        ok, msg = core.initialize_router()
        return ok, "Configuration edited successfully", None, None
    else:
        return ok, result.stderr, None, None

@cmd(
    "/config",
    "Configure the agent interactively",
    aliases = ["/configure"],
)
def _cmd_config(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    # URL, model, reasoning, temperature, vi
    commands = ["/url", "/model", "/reasoning", "/temperature", "/vi"]

    for command in commands:
        ok, msg, _, _, _ = registry.run_command(core, command, prompt_ui)
        if not ok:
            return False, msg, None, None

    return True, "Configuration updated", None, None

@cmd(
      "/mcp-edit",
       "Edit the mcp.json configuration file with $EDITOR or $VISUAL",
)
def _cmd_mcp_edit(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import edit_mcp_config_visual
    result = edit_mcp_config_visual()
    ok = result.returncode == 0
    if ok:
        return ok, "MCP configuration edited successfully", None, None
    else:
        return ok, result.stderr, None, None

@cmd(
      "/mcp",
      "Show the current MCP configuration file",
      aliases = ["/mcp-show"],
)
def _cmd_mcp_show(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    from agent.config import get_mcp_config_path
    path = get_mcp_config_path()
    with path.open() as file:
        content = file.read()

    if content:
        return True, f"MCP configuration: {path}", content, None
    else:
        return False, "Could not load MCP configuration file", None, None

@cmd(
      "/mcp-tools",
      "List MCP tools ⚙",
)
def _cmd_mcp_tools(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    from agent.tools import get_tools_str
    return True, None, get_tools_str(prefix="mcp_", contains=True), None

@cmd(
    "/temperature",
    "Set the inference temperature parameter in 0..2",
    aliases=["/temp", "/t"],
)
def _cmd_temperature(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    ui = prompt_ui or _fallback_prompt_ui()
    new_temp = ui.ask_float(" Enter the temperature [0..2]", default=core.router.temperature)
    if new_temp < 0 or new_temp > 2:
        return False, f"Temperature out of [0..2] range: {new_temp}", None, None

    core.router.temperature = new_temp

    return True, f"Temperature: {new_temp}", None, None


@cmd(
      "/vi",
       "Enable/disable vi input mode",
)
def _cmd_vi(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if params:
        return False, no_params_error, None, None

    ui = prompt_ui or _fallback_prompt_ui()
    from agent.config import get_config
    config = get_config()

    opts = [("true", "On"), ("false", "Off")]
    defa = config.get("agent.vi_mode")

    state = ui.ask_choice(
        message="Vi input mode:",
        options=opts,
        default=str(defa).lower(),
    )
    state_bool = state == "true"
    config.set("agent.vi_mode", state_bool)
    pub.sendMessage("prompt-update")
    return True, f"Vi mode: {state_bool}", None, None


@cmd(
    "/attachimage",
    "Attach an image to the next prompt",
    examples=[
        "/attachimage path/to/screenshot.png",
        "/attachimage ~/Pictures/photo.jpg",
    ]
)
def _cmd_attach_image(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    if not params:
        return False, "please provide an image file path", None, None

    file_path = os.path.expanduser(' '.join(params))
    path = Path(file_path)

    if not path.is_file():
        return False, f"file not found: {file_path}", None, None

    try:
        raw_bytes = path.read_bytes()
        result = resize_image(raw_bytes)
        core._pending_image = result
        return True, f"🖼️  Image loaded and attached — will be sent with your next message. ([dim]{path.name}[/dim], {len(result['image_base64'])} bytes base64)", None, None
    except Exception as e:
        return False, f"failed to load image: {e}", None, None


@cmd(
    "/help",
    "Show command help",
    aliases=["/commands"],
)
def _cmd_help(core, params, prompt_ui=None) -> tuple[bool, str | None, str | None, str | None]:
    return True, None, registry.get_commands_str(), None
