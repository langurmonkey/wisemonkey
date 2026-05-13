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

# Try to import prompt_toolkit for rich input; fall back to plain input.
try:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


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
            print(f"[red]Error[/red]: OpenAI endpoint creation: {err}")
            raise Exception(f"{err}")

        # Agent settings
        self.model = model_cfg.get("name", "qwen/qwen3.6-35b-a3b")
        self.max_turns = agent_cfg.get("max_turns", 50)
        self.personality = agent_cfg.get("system_prompt", "You are a helpful assistant, expert in many areas of science. Respond concisely and to the point. No fluff.")
        self.stream = agent_cfg.get("stream", True)

        # Initialize subsystems
        self.memory = Memory()
        self.skills = SkillLoader()

        # Conversation history
        self.messages = []

    def _build_system_prompt(self):
        """Build the system prompt with personality, skills, and memory."""
        parts = [self.personality]

        # Add formatted memory
        memory_text = self.memory.get_formatted()
        if memory_text:
            parts.append(memory_text)

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

            for chunk in response:
                delta = chunk.choices[0].delta

                # Collect text
                if delta.content:
                    full_text += delta.content
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

            print()  # newline after streaming

            # Convert indexed dict to list
            tc_list = list(tool_calls.values()) if tool_calls else None

            return {"text": full_text, "tool_calls": tc_list}

        # Non-streaming: return message object
        if not response.choices:
            raise RuntimeError(
                f"LLM returned no choices. Model: {self.model}, "
                f"base_url: {self.client.base_url}. "
                f"Check that the model name matches a loaded model in LM Studio. "
                f"Response: {response}"
            )

        return response.choices[0].message

    def _handle_tool_calls(self, message):
        """Process tool calls from the LLM response."""
        tool_calls = message.tool_calls
        if not tool_calls:
            return None

        # Append the assistant message
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })

        # Execute each tool call
        for call in tool_calls:
            print
            tool_name = call.function.name
            tool_args = json.loads(call.function.arguments)

            print(f"{col.BGGRAY}Activating tool: {tool_name}{col.END}")
            result = execute_tool(tool_name, tool_args)

            # Append tool result
            self.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

        return True

    def run(self, user_input):
        """Run the agent loop with a user message.

        Args:
            user_input: The user's message string.

        Returns:
            The final text response from the LLM.
        """
        # Initialize with system prompt
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        for turn in range(self.max_turns):
            # Send to LLM
            result = self._send_to_llm(stream=self.stream)

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
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]

                    print(f"[gray]☛ Activating tool: {tool_name}[/gray]")
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
            self.memory.save()
            return response_text

        # Max turns reached
        self.memory.save()
        return "I've reached the maximum number of turns. Please rephrase your request."

    def print_help(self):
        print(f"⬤ [green]/q[/], [green]/quit[/], [green]/exit[/]   → exit")
        print(f"⬤ [green]/tools[/]             → list tools")
        print(f"⬤ [green]/skills[/]            → list skills")
        print(f"⬤ [green]/config[/]            → print configuration")
        print(f"⬤ [green]/help[/], [green]/commands[/]   → print command help")
        print()
        

    def run_interactive(self):
        """Run the agent in interactive mode."""
        title = Align.center("[bold blue]LANGUR AGENT[/bold blue]", vertical='middle')
        print(Panel(title, box=box.HEAVY, subtitle="The dead-simple AI agent for local workflows"))
        print()
        self.print_help()

        # Set up prompt_toolkit if available
        history_path = xdg_data_home() / "langur-agent" / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)


        if _HAS_PROMPT_TOOLKIT:
            style = Style.from_dict({
                "prompt": "ansiyellow",
            })
            get_input = lambda: str(
                prompt(style=style,
                       message=":: You ::\n❯ ",
                       history=FileHistory(str(history_path)),
                       complete_while_typing=True)
            ).strip()
        else:
            get_input = lambda: Prompt.ask("[yellow]:: You ::[/]\n❯ ")

        while True:
            try:
                user_input = get_input()
            except (EOFError, KeyboardInterrupt):
                print(f"\n[bold blue]Goodbye![/]")
                break

            if not user_input:
                continue

            # SPECIAL COMMANDS
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print(f"\n[bold blue]Goodbye![/]")
                break
            elif user_input.lower() in ("/tools"):
                log_tools()
                continue
            elif user_input.lower() in ("/skills"):
                self.skills.log_skills()
                continue
            elif user_input.lower() in ("/config"):
                log_config()
                continue
            elif user_input.lower() in ("/help", "/commands"):
                self.print_help()
                continue

            print(f"\n[magenta]:: Agent :: [/magenta]\n", end="", flush=True)
            response = self.run(user_input)
            print()

        # Persist memory on session exit
        self.memory.save()
