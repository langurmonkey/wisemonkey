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
import tiktoken
import openai

from rich import box, inspect
from rich.prompt import Prompt
from rich.panel import Panel
from rich.align import Align
from rich.markdown import Markdown

from pathlib import Path
from xdg_base_dirs import xdg_data_home

from agent.config import get_config
from agent.memory import Memory
from agent.skills import SkillLoader
from agent.tools import get_tool_schemas, execute_tool
from agent.commands import registry
from agent.console import console


# Try to import prompt_toolkit for rich input; fall back to plain input.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import FuzzyWordCompleter
    from prompt_toolkit.clipboard import InMemoryClipboard
    from prompt_toolkit.formatted_text import HTML
    _HAS_PROMPT_TOOLKIT = True

except ImportError:
    console.print("[red]ERROR:[/red] could not initialize promtp toolkit")
    _HAS_PROMPT_TOOLKIT = False

class TurnCancelled(Exception):
    """Raised when the user cancels an LLM turn mid-stream."""
    pass

class Agent:
    """Simple LLM agent with tools, skills, and memory."""

    def __init__(self, config_path=None):
        self.config = get_config()
        self.config.load(config_path)

        # Initialize client
        self._initialize_client()

        # Agent settings
        self.system_prompt = self.config.get("agent.system_prompt", "You are a helpful assistant, expert in many areas of science. Respond concisely and to the point. No fluff.")
        max_chat_history = self.config.get("max_chat_history", 128000)
        self.markdown = self.config.get("agent.markdown", False)

        # Initialize subsystems
        self.memory = Memory(max_chat_history=max_chat_history)
        self.skills = SkillLoader()

        # Conversation history
        self.messages = []

        # Status
        self.thinking = False
        self.generating = False

        # Initialize tokenizer for token-counting
        encoding_name = "cl100k_base"
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            console.print(f"[red]ERROR[/red]: Error loading tokenizer: {e}")
            raise Exception(f"{e}")

    def _initialize_client(self):
        """Initializes the client given the current configuration."""

        # Initialize OpenAI client
        api_key = self.config.get("model.api_key") or os.environ.get("LANGUR_API_KEY", "")
        base_url = self.config.get("model.base_url", "http://127.0.0.1:1234/v1")
        # Auto-append /v1 for LM Studio / local API servers
        if base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        try:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url or None)
        except openai.OpenAIError as err:
            console.print(f"[red]ERROR[/red]: OpenAI endpoint creation: {err}")
            raise Exception(f"{err}")

    def _build_system_prompt(self):
        """Build the system prompt with personality, skills, and memory."""
        parts = [self.system_prompt]

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

    def _stream_handler(self,
                        response,
                        reasoning_callback=None,
                        content_callback=None):
        self.thinking_buffer = ""
        self.response_buffer = ""

        tool_calls = {}
        first_chunk_time = None
        thinking_end = False

        for chunk in response:
            # Stop prompt spinner
            if self.spinner_prompt:
                self.spinner_prompt.stop()
                self.spinner_prompt = None
                console.print("[green]✓[/] ⏳ Prompt processed")
                
            delta = chunk.choices[0].delta
            now = time.time()

            # Track when first chunk arrives (excludes request send time)
            if first_chunk_time is None:
                first_chunk_time = now

            show_thinking = self.config.get("model.show_thinking", False)
            # Thinking
            if (
                hasattr(delta, 'reasoning_content')
                and delta.reasoning_content
            ):
            
                if not self.thinking:
                    console.rule(style="grey39")
                    if show_thinking:
                        console.print("[orange1]⇨[/] 💡 Thinking...")
                        if reasoning_callback:
                            reasoning_callback("start")
                    else:
                        self.spinner_thinking = console.status("💡 Thinking...")
                        self.spinner_thinking.start()

                self.thinking_buffer += delta.reasoning_content

                if show_thinking:
                    # Only print, do not save to main buffer
                    console.print(f"[grey39]{delta.reasoning_content}[/]", end="")
                    if reasoning_callback:
                        reasoning_callback("body", delta.reasoning_content)

                self.thinking = True

            # Collect response text
            if delta.content:
                if self.thinking and not thinking_end:
                    # End thinking
                    self.thinking = False
                    thinking_end = True
                    if show_thinking:
                        if reasoning_callback:
                            reasoning_callback("end")
                    elif self.spinner_thinking:
                        self.spinner_thinking.stop()
                        self.spinner_thinking = None
                    console.print("[green]✓[/] 💡 Done thinking")
                    console.rule(style="grey39")
                
                self.response_buffer += delta.content
                if content_callback:
                    content_callback(delta.content)

                self.generating = True

                # Print text
                console.print(delta.content, end="")

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

        return (first_chunk_time, tool_calls)

    def _send_to_llm(self,
                    reasoning_callback=None,
                    content_callback=None):
        """Send messages to the LLM and get a response.
        """
        tools = get_tool_schemas()
        start = time.time()
        model_name = self.config.get("model.name", "qwen/qwen3.6-35b-a3b")
        try:
            self.spinner_prompt = console.status("⏳ Processing prompt...")
            self.spinner_prompt.start()

                
            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages,
                temperature=self.config.get("model.temperature", 0.8),
                tools=tools if tools else None,
                tool_choice="auto",
                stream=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"API connection error. Please, check the endpoint [model={model_name}, base_url={self.client.base_url}]: {e}"
            ) from e

        try:
            (first_chunk_time, tool_calls) = self._stream_handler(response,
                                                                  reasoning_callback,
                                                                  content_callback)
        except KeyboardInterrupt:
            # Close the response stream to stop the API call
            response.close()
            console.print("\n[bold yellow]⏹  Turn cancelled[/bold yellow]")
            raise TurnCancelled() from None

        # We are done!
        console.print()

        # Print markdown if needed
        if self.markdown and self.response_buffer:
            md_panel = Panel(
                Markdown(self.response_buffer),
                title="[bold]📝 Markdown version[/bold]",
                title_align="left",
                border_style="magenta",
                padding=(1, 2),
            )
            console.print(md_panel)
        
        self.thinking = False
        self.generating = False
        now = time.time()

        # Count tokens using tiktoken (LM Studio streaming doesn't include usage)
        tokens = 0
        if self.encoding and self.response_buffer:
            tokens = len(self.encoding.encode(self.response_buffer))

        # Elapsed time: from first chunk to last chunk (generation time only)
        if first_chunk_time is not None:
            gen_elapsed = now - first_chunk_time
        else:
            gen_elapsed = time.time() - start
        if gen_elapsed <= 0:
            gen_elapsed = 1  # avoid division by zero
        
        # Convert indexed dict to list
        tc_list = list(tool_calls.values()) if tool_calls else None

        return ({"text": self.response_buffer, "tool_calls": tc_list}, tokens, gen_elapsed)

    def run(self,
            user_input,
            reasoning_callback=None,
            content_callback=None):
        """Run a turn interaction with a user message.

        Args:
            user_input: The user's message string.
            reasoning_callback: Callback for reasoning updates.
            content_callback: Callback for content updates.

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
        for turn in range(self.config.get("agent.max_turns", 50)):
            # Send to LLM
            try:
                (result, tokens, gen_elapsed) = self._send_to_llm(reasoning_callback,
                                                                  content_callback)
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

                    console.print(f"[black on #66aa99] ⚙ Activating tool: {tool_name} [/black on #66aa99]")
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
        console.print(f"[black on #777777]  ⬤  {total_gen_time:.1f}s  ⬤  {total_tokens} tokens  ⬤  {ntools} tools  [/black on #777777]")
        console.print()

        
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
            "frame.border": "ansiyellow",
        })

        # Vi mode
        vi_mode = self.config.get("agent.vi_mode", False)

        # Slash commands autocompleter
        commands = [cmd.name for cmd in registry.list_commands()]
        slash_completer = FuzzyWordCompleter(commands)

        # History path
        history_path = xdg_data_home() / "langur-agent" / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Toolbar
        prompt_toolbar = lambda : HTML(" <b>Alt</b>+<b>Enter</b>: new line | <b>Enter</b>: submit prompt | <b>Ctrl</b>+<b>C</b>: quit")

        self._session = PromptSession(
                    style=style,
                    message="⩥ You ⩤\n❯ ",
                    history=FileHistory(str(history_path)),
                    show_frame=True,
                    multiline=True,
                    key_bindings=kb,
                    vi_mode=vi_mode,
                    clipboard=InMemoryClipboard(),
                    enable_open_in_editor=vi_mode,
                    complete_while_typing=True,        
                    complete_in_thread=True,
                    completer=slash_completer,
                    auto_suggest=AutoSuggestFromHistory(),
                    bottom_toolbar=prompt_toolbar,
        )
        
    def run_interactive(self):
        """Run the agent in interactive mode."""
        import shutil
        term_size = shutil.get_terminal_size((80, 20))
        if term_size.columns < 80:
            languragent="LANGUR AGENT"
        else:
            languragent = '''
██      ▄▄▄  ▄▄  ▄▄  ▄▄▄▄ ▄▄ ▄▄ ▄▄▄▄    ▄████▄  ▄▄▄▄ ▄▄▄▄▄ ▄▄  ▄▄ ▄▄▄▄▄▄
██     ██▀██ ███▄██ ██ ▄▄ ██ ██ ██▄█▄   ██▄▄██ ██ ▄▄ ██▄▄  ███▄██   ██  
██████ ██▀██ ██ ▀██ ▀███▀ ▀███▀ ██ ██   ██  ██ ▀███▀ ██▄▄▄ ██ ▀██   ██  
            '''
        title = Align.center(f"[bold blue]{languragent}[/bold blue]", vertical='middle')
        console.print(Panel(title, box=box.HEAVY, border_style="blue"))
        console.print()
        console.print(registry.get_commands_str())

        if _HAS_PROMPT_TOOLKIT:
            self._create_prompt_session()
        
        if self._session and False:
            style = Style.from_dict({
                "prompt": "ansiyellow",
            })
            get_input = lambda: str(
                self._session.prompt()
            ).strip()
        else:
            get_input = lambda: Prompt.ask(prompt="[yellow]⩥ You ⩤[/yellow]\n❯",
                                           console=console)

        while True:
            try:
                user_input = get_input()
            except (EOFError, KeyboardInterrupt):
                console.print(f"\n[bold blue]Goodbye![/]")
                break

            if not user_input:
                continue

            # SPECIAL COMMANDS
            if user_input.startswith("/"):
                tokens = user_input.split()
                command, params = registry.lookup(tokens)

                if command:
                    result, should_exit = registry.execute(self, command, params)
                    if should_exit:
                        console.print(f"\n[bold blue]Goodbye![/]")
                        break
                    if result:
                        console.print(result)
                        console.print()
                else:
                    console.print(f"[red]ERROR:[/red] command not found: {user_input}")
                    
                continue

            else:

                console.print(f"\n[magenta]⩥ Agent ⩤ [/magenta]  ⦗[blue]{self.config.get('model.name')}[/blue]⦘")
                console.print("  [dim][bold]Ctrl[/bold]+[bold]C[/bold]: Cancel turn[/dim]\n")
                (response, total_tokens, ntools, total_gen_time) = self.run(user_input)
                console.print()
                if response == "[Cancelled]":
                    continue  # skip status line, go straight back to prompt
                self._statusline(total_tokens, ntools, total_gen_time)

        # Persist memory on session exit
        self.memory.save()

    def get_models(self):
        """Gets a list with all the available models."""
        return self.client.models.list()

    def set_model(self, model_name):
        """Sets the model to use."""
        models = self.get_models()
        for model in models:
            if model_name == model.id:
                # Match, set and return
                self.config.set("model.name", model_name)
                return

        raise NameError(f"the model '{model_name}' does not exsit")
        
