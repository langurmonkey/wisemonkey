"""Core agent loop.

The agent:
1. Builds a system prompt (personality + skills + tools + memory)
2. Sends messages to the LLM
3. Handles tool calls or returns the final text response
4. Loops until max turns or a text response
"""

import json
import os
import sys
import time
# Counting tokens
import tiktoken

from rich import print, box, inspect
from rich.prompt import Prompt
from rich.panel import Panel
from rich.align import Align
from pathlib import Path
from xdg_base_dirs import xdg_data_home

import openai

from agent.config import load_config, log_config
from agent.memory import Memory
from agent.skills import SkillLoader
from agent.tools import get_tool_schemas, execute_tool, log_tools
from agent.commands import registry

# Try to import prompt_toolkit for rich input; fall back to plain input.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import FuzzyWordCompleter
    from prompt_toolkit.formatted_text import HTML
    _HAS_PROMPT_TOOLKIT = True

except ImportError:
    print("[red]ERROR:[/red] could not initialize promtp toolkit")
    _HAS_PROMPT_TOOLKIT = False

class TurnCancelled(Exception):
    """Raised when the user cancels an LLM turn mid-stream."""
    pass

class Agent:
    """Simple LLM agent with tools, skills, and memory."""

    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        model_cfg = self.config.get("model", {})
        agent_cfg = self.config.get("agent", {})

        # Initialize OpenAI client
        api_key = model_cfg.get("api_key") or os.environ.get("LANGUR_API_KEY", "")
        base_url = model_cfg.get("base_url")
        # Auto-append /v1 for LM Studio / local API servers
        if base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        try:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url or None)
        except openai.OpenAIError as err:
            print(f"[red]ERROR[/red]: OpenAI endpoint creation: {err}")
            raise Exception(f"{err}")

        # Agent settings
        self.model = model_cfg.get("name", "qwen/qwen3.6-35b-a3b")
        self.max_turns = agent_cfg.get("max_turns", 50)
        self.personality = agent_cfg.get("system_prompt", "You are a helpful assistant, expert in many areas of science. Respond concisely and to the point. No fluff.")
        self.stream = agent_cfg.get("stream", True)
        max_chat_history = agent_cfg.get("max_chat_history", 128000)

        # Initialize subsystems
        self.memory = Memory(max_chat_history=max_chat_history)
        self.skills = SkillLoader()

        # Conversation history
        self.messages = []

        # Initialize tokenizer for token-counting
        encoding_name = "cl100k_base"
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"[red]ERROR[/red]: Error loading tokenizer: {e}")
            raise Exception(f"{e}")

    def _build_system_prompt(self):
        """Build the system prompt with personality, skills, and memory."""
        parts = [self.personality]

        # Add formatted memory
        memory_text = self.memory.get_formatted()
        if memory_text:
            parts.append(memory_text)

        # Add chat history
        chat_text = self.memory.get_formatted_chat()
        if chat_text:
            parts.append(chat_text)

        # Add skills
        skills_text = self.skills.load_all()
        if skills_text:
            parts.append(skills_text)

        return "\n".join(parts)

    def _send_to_llm(self, stream=False):
        """Send messages to the LLM and get a response.

        Args:
            stream: If True, print tokens as they arrive and return a
                    dict with 'text' and 'tool_calls' keys. If False,
                    return the raw message object.
        """
        tools = get_tool_schemas()
        start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tools if tools else None,
                tool_choice="auto",
                stream=stream,
            )
        except Exception as e:
            raise RuntimeError(
                f"LLM API error (model={self.model}, base_url={self.client.base_url}): {e}"
            ) from e

        if stream:
            # Collect streamed tokens and tool calls
            full_text = ""
            tool_calls = {}  # indexed by position to merge partial deltas
            first_chunk_time = None

            try:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    now = time.time()

                    # Track when first chunk arrives (excludes request send time)
                    if first_chunk_time is None:
                        first_chunk_time = now

                    # Collect text
                    if delta.content:
                        full_text += delta.content

                        # Print text
                        print(delta.content, end="", flush=True)

                    # Collect tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc.function.name:
                                tool_calls[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[idx]["function"]["arguments"] += tc.function.arguments

            except KeyboardInterrupt:
                # Close the response stream to stop the API call
                response.close()
                print("\n[bold yellow]⏹  Turn cancelled[/bold yellow]")
                raise TurnCancelled() from None

            # Newline
            print()

            # Count tokens using tiktoken (LM Studio streaming doesn't include usage)
            tokens = 0
            if self.encoding and full_text:
                tokens = len(self.encoding.encode(full_text))

            # Elapsed time: from first chunk to last chunk (generation time only)
            if first_chunk_time is not None:
                gen_elapsed = now - first_chunk_time
            else:
                gen_elapsed = time.time() - start
            if gen_elapsed <= 0:
                gen_elapsed = 1  # avoid division by zero
            
            # No debug output

            # Convert indexed dict to list
            tc_list = list(tool_calls.values()) if tool_calls else None

            return ({"text": full_text, "tool_calls": tc_list}, tokens, gen_elapsed)
        # Non-streaming: return message object
        if not response.choices:
            raise RuntimeError(
                f"LLM returned no choices. Model: {self.model}, "
                f"base_url: {self.client.base_url}. "
                f"Check that the model name matches a loaded model in LM Studio. "
                f"Response: {response}"
            )

        tokens = response.choices[0].message.usage.completion_tokens
        gen_elapsed = time.time() - start
        return (response.choices[0].message, tokens, gen_elapsed)

    def run(self, user_input):
        """Run a turn interaction with a user message.

        Args:
            user_input: The user's message string.

        Returns:
            The final text response from the LLM.
        """
        # Record start time
        start = time.time()

        # Record user input in chat memory
        self.memory.add_chat_exchange("user", user_input)

        # Initialize with system prompt
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        # Total token count
        total_tokens = 0
        # Total generation time (excludes network latency, prompt building, etc.)
        total_gen_time = 0
        # Tool usages
        ntools = 0
        for turn in range(self.max_turns):
            # Send to LLM
            try:
                (result, tokens, gen_elapsed) = self._send_to_llm(stream=self.stream)
            except TurnCancelled:
                # User cancelled: don't persist anything, return immediately
                return ("[Cancelled]", 0, 0, 0.0)

            total_tokens += tokens
            total_gen_time += gen_elapsed

            # Normalize tool calls from both streaming (plain dicts) and
            # non-streaming (OpenAI API objects) into a common format
            if isinstance(result, dict):
                # Streaming mode: result is {"text": ..., "tool_calls": ...}
                response_text = result.get("text", "")
                raw_tool_calls = result.get("tool_calls")
            else:
                # Non-streaming mode: result is a message object
                response_text = result.content or ""
                raw_tool_calls = result.tool_calls

            # Normalize tool calls to plain dicts
            tool_calls = []
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    if isinstance(tc, dict):
                        # Already a plain dict from streaming
                        tool_calls.append(tc)
                    else:
                        # OpenAI API object — convert to dict
                        tool_calls.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })

            # Handle tool calls
            if tool_calls:
                # Append the assistant message with tool calls as plain dicts
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })

                # Execute each tool call
                for tc in tool_calls:
                    ntools = ntools + 1
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]

                    print(f"[black on #66aa99] ⚙ Activating tool: {tool_name} [/black on #66aa99]")
                    result = execute_tool(tool_name, json.loads(tool_args) if isinstance(tool_args, str) else tool_args)

                    # Append tool result
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                continue  # Loop back to LLM with tool results

            # No tool calls - this is the final response
            self.messages.append({"role": "assistant", "content": response_text})
            
            # Record assistant output in chat memory
            self.memory.add_chat_exchange("assistant", response_text)
            # Persist memory
            self.memory.save()
            
            return (response_text, total_tokens, ntools, total_gen_time)

        # Max turns reached!
        # Persist memory
        self.memory.save()
        return "I've reached the maximum number of turns. Please rephrase your request."

    def _statusline(self, total_tokens, ntools, total_gen_time):
        print(f"[black on #777777]  ⬤  {total_gen_time:.1f}s  ⬤  {total_tokens} tokens  ⬤  {ntools} tools  [/black on #777777]")
        print()

    def print_help(self):
        for cmd in registry.list_commands():
            aliases = ", ".join(f"[green]{a}[/]" for a in cmd.aliases)
            primary = f"[green]{cmd.name}[/]"
            if aliases:
                names = f"{primary}, {aliases}"
            else:
                names = primary
            print(f"⬤ {names} → {cmd.description}")
        print()
        
    def _create_prompt_session(self):
        # Key bindings: 
        kb = KeyBindings()
        @kb.add('enter')
        def _(event):
            """Enter submits the input."""
            event.current_buffer.validate_and_handle()
        @kb.add('escape', 'enter')
        def _(event):
            """Alt+Enter inserts a newline."""
            event.current_buffer.insert_text('\n')

        # Create prompt session now
        style = Style.from_dict({
            "prompt": "ansiyellow",
        })

        # Vi mode
        vi_mode = self.config.get("agent").get("vi_mode", False)

        # Slash commands autocompleter
        commands = [cmd.name for cmd in registry.list_commands()]
        slash_completer = FuzzyWordCompleter(commands)

        # History path
        history_path = xdg_data_home() / "langur-agent" / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Toolbar
        prompt_toolbar = lambda : HTML(" <b>Alt</b>+<b>Enter</b>: new line | <b>Enter</b>: submit prompt")

        self._session = PromptSession(
                    style=style,
                    message="⩥ You ⩤\n❯ ",
                    history=FileHistory(str(history_path)),
                    show_frame=True,
                    multiline=True,
                    key_bindings=kb,
                    vi_mode=vi_mode,
                    enable_open_in_editor=vi_mode,
                    complete_while_typing=True,        
                    complete_in_thread=True,
                    completer=slash_completer,
                    auto_suggest=AutoSuggestFromHistory(),
                    bottom_toolbar=prompt_toolbar,
        )
        
    def run_interactive(self):
        """Run the agent in interactive mode."""
        title = Align.center("[bold blue]LANGUR AGENT[/bold blue]", vertical='middle')
        print(Panel(title, box=box.HEAVY, border_style="yellow"))
        print()
        self.print_help()


        if _HAS_PROMPT_TOOLKIT:
            self._create_prompt_session()
        
        if self._session:
            style = Style.from_dict({
                "prompt": "ansiyellow",
            })
            get_input = lambda: str(
                self._session.prompt()
            ).strip()
        else:
            get_input = lambda: Prompt.ask("[yellow]⩥ You ⩤[/yellow]\n❯ ")

        while True:
            try:
                user_input = get_input()
            except (EOFError, KeyboardInterrupt):
                print(f"\n[bold blue]Goodbye![/]")
                break

            if not user_input:
                continue

            # SPECIAL COMMANDS
            if user_input.startswith("/"):
                tokens = user_input.split()
                command = tokens[0]
                params = tokens[1:] if len(tokens) > 1 else []

                result, should_exit = registry.execute(self, command, params)
                if should_exit:
                    print(f"\n[bold blue]Goodbye![/]")
                    break
                if result:
                    print(result)
                    print()
                continue
            else:

                print(f"\n[magenta]⩥ Agent ⩤ [/magenta]  ⦗[blue]{self.model}[/blue]⦘\n", end="", flush=True)
                (response, total_tokens, ntools, total_gen_time) = self.run(user_input)
                print()
                if response == "[Cancelled]":
                    continue  # skip status line, go straight back to prompt
                self._statusline(total_tokens, ntools, total_gen_time)

        # Persist memory on session exit
        self.memory.save()
