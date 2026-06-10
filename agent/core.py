"""Core logic of the Wisemonkey agent.

The core:
1. Builds a system prompt (personality + skills + tools + memory)
2. Sends messages to the LLM using the router
3. Handles tool calls or returns the final text response
4. Loops until max turns or a text response
"""

import json
import time
import tiktoken

from enum import Enum
from pathlib import Path

from agent.config import get_config, get_mcp_config_path
from agent.memory import Memory
from agent.skills import SkillLoader
from agent.mcp import MCPClient
from agent.router import ModelRouter
from agent.tools import get_tool_schemas, execute_tool

class TurnCancelled(Exception):
    """Raised when the user cancels an LLM turn during inference."""
    pass

class Stage(Enum):
    START = 0
    PROCESS = 1
    STOP = 2

class Core:
    """Agent core, which manages tools, skills, memory, and API communication."""

    def __init__(self, config_path=None, session='default', full_startup=True):
        self.config = get_config()
        self.config.load(config_path)

        # Initialize API router
        ok, msg = self.initialize_router()
        if not ok:
            raise RuntimeError(msg)

        # Agent settings
        self.system_prompt = self.config.get("agent.system_prompt", "You are a helpful assistant, expert in many areas of science. Respond concisely and to the point. No fluff.")

        # In case of onboarding we do not need a full startup
        if full_startup:
            # Initialize MCP
            self.mcp = MCPClient()
            self.mcp.load_config(get_mcp_config_path())
            self.mcp.start_all()

            # Initialize memory
            max_chat_history = self.config.get("agent.max_chat_history", 300000)
            self.memory = Memory(max_chat_history=max_chat_history, session=session)

            # Initialize skills
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
                raise Exception(f"Error loading tokenizer: {e}")

    def initialize_router(self):
        """Initialize the model router."""
        self.router = ModelRouter(self.config)
        return self.router.initialize()

    def shutdown(self):
        """Shutdown the agent core."""
        if self.mcp:
            self.mcp.stop_all()

    def _find_workspace_root(self, start: Path | None = None) -> Path:
        """Search for context files starting from start (or cwd) and walking up.

        Returns the deepest directory that contains at least one configured
        context file, or ``start`` (or cwd) if none are found.
        """
        context_files = self.config.get("agent.context_files", ["AGENTS.md"])
        current = (start or Path.cwd()).resolve()
        for parent in [current, *current.parents]:
            for name in context_files:
                if (parent / name).is_file():
                    return parent
        return current

    def _load_context_files(self) -> str:
        """Load and cache the contents of all configured context files.

        Returns the concatenated file contents separated by headings, or an
        empty string if none are found.
        """
        if hasattr(self, "_context_files_cache"):
            return self._context_files_cache

        context_files = self.config.get("agent.context_files", ["AGENTS.md"])
        workspace_root = self._find_workspace_root()

        parts: list[str] = []
        for name in context_files:
            path = workspace_root / name
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if content:
                    parts.append(f"## Workspace Instructions ({name})\n{content}")

        self._context_files_cache = "\n\n".join(parts) if parts else ""
        return self._context_files_cache

    def _build_system_prompt(self):
        """Build the system prompt with personality, skills, and memory."""
        parts = [self.system_prompt]

        # Add workspace context files
        context = self._load_context_files()
        if context:
            parts.append(context)

        # Add formatted memory
        # Only user profile, no notes
        memory_text = self.memory.get_formatted(notes=False)
        if memory_text:
            parts.append(memory_text)

        # Add chat history
        chat_text = self.memory.get_chat_formatted(timestamps=False)
        if chat_text:
            parts.append(chat_text)

        # Add skills
        skills_text = self.skills.load_all()
        if skills_text:
            parts.append(skills_text)

        return "\n".join(parts)

    def _stream_handler(self,
                        response,
                        prompt_callback=None,
                        reasoning_callback=None,
                        content_callback=None,
                        error_callback=None):
        self.thinking_buffer = ""
        self.response_buffer = ""

        tool_calls = {}
        first_chunk_time = None
        thinking_end = False
        prompt_stopped = False

        thinking_display = self.config.get("model.thinking.display", False)

        for chunk in response:
            # Stop prompt spinner
            if not prompt_stopped and prompt_callback:
                prompt_callback(Stage.STOP)
                prompt_stopped = True

            delta = chunk.choices[0].delta
            now = time.time()

            # Track when first chunk arrives (excludes request send time)
            if first_chunk_time is None:
                first_chunk_time = now

            # Thinking
            if (
                hasattr(delta, 'reasoning_content')
                and delta.reasoning_content
            ):
            
                if not self.thinking:
                    if reasoning_callback:
                        reasoning_callback(Stage.START, None, thinking_display)

                self.thinking_buffer += delta.reasoning_content

                if reasoning_callback:
                    reasoning_callback(Stage.PROCESS, delta.reasoning_content, thinking_display)

                self.thinking = True

            # Collect response text
            if delta.content:
                if self.thinking and not thinking_end:
                    # End thinking
                    self.thinking = False
                    thinking_end = True
                    if reasoning_callback:
                        reasoning_callback(Stage.STOP, None, thinking_display)
                
                self.response_buffer += delta.content
                if content_callback:
                    content_callback(delta.content)

                self.generating = True

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
                        existing = tool_calls[idx]["function"]["arguments"]
                        incoming = tc.function.arguments
                        if isinstance(incoming, dict):
                            # Ollama delivers the full dict in one chunk
                            tool_calls[idx]["function"]["arguments"] = json.dumps(incoming)
                        else:
                            # OpenAI/Anthropic stream partial JSON strings
                            tool_calls[idx]["function"]["arguments"] = existing + incoming

        return (first_chunk_time, tool_calls)

    def llm_chat_raw(self,
                    messages,
                    error_callback=None):
        """
        Send a single interaction to the LLM and get the non-streaming response.
        """
        
        model_name = self.config.get("model.name", "qwen/qwen3.6-35b-a3b")
        try:
            return self.router.chat(
                messages=messages,
                stream=False,
                tools=None,
            )
        except Exception as e:
            if error_callback:
                error_callback(e, f"API connection error. Please, check the endpoint (model={model_name}, base_url={self.router._base_url}): {e}")
            else:
                raise RuntimeError(f"API connection error. Please, check the endpoint (model={model_name}, base_url={self.router._base_url}): {e}") from e
                

    def _cancel_prompts(self,
                        prompt_callback=None,
                        reasoning_callback=None):
        if prompt_callback:
            prompt_callback(Stage.STOP)
        if reasoning_callback:
            reasoning_callback(Stage.STOP)

    def _send_to_llm(self,
                     prompt_callback=None,
                     reasoning_callback=None,
                     content_callback=None,
                     cancel_callback=None,
                     error_callback=None):
        """
        Send messages to the LLM and get a response.
        """
        tools = get_tool_schemas()
        start = time.time()
        model_name = self.config.get("model.name", "qwen/qwen3.6-35b-a3b")
        try:

            if prompt_callback:
                prompt_callback(Stage.START)
                
            response = self.router.chat(
                messages=self.messages,
                stream=True,
                tools=tools if tools else None,
            )
        except Exception as e:
            self._cancel_prompts(prompt_callback, reasoning_callback)
            
            if error_callback:
                error_callback(e, f"API connection error. Please, check the endpoint (model={model_name}, base_url={self.router._base_url}): {e}")
            else:
                raise RuntimeError(f"API connection error. Please, check the endpoint (model={model_name}, base_url={self.router._base_url}): {e}") from e

        try:
            (first_chunk_time, tool_calls) = self._stream_handler(response,
                                                                  prompt_callback,
                                                                  reasoning_callback,
                                                                  content_callback,
                                                                  error_callback)
        except KeyboardInterrupt as e:
            # Close the response stream to stop the API call
            response.close()
            self._cancel_prompts(prompt_callback, reasoning_callback)
            if cancel_callback:
                cancel_callback(e)

        return self._finish_inference(start, first_chunk_time, tool_calls)

    def _finish_inference(self, start, first_chunk_time, tool_calls):
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


    def run_turn(self,
            user_input,
            prompt_callback=None,
            reasoning_callback=None,
            content_callback=None,
            tool_callback=None,
            cancel_callback=None,
            error_callback=None):
        """
        Run a turn interaction with a user message. All callbacks are Agent methods, so
        they take the agent as the first parameter.

        Args:
            user_input: The user's message string.
            prompt_callback: Callback for prompt processing. Gets Stage.
            reasoning_callback: Callback for reasoning updates. Gets Stage and optional message.
            content_callback: Callback for content updates. Gets message.
            tool_callback: Callback to run during tool activations. Gets tool name and args.
            cancel_callback: Callback for user-canceled inference.
            error_callback: Callback on error.

        Returns:
            The final text response from the LLM.
        """


        # Initialize with system prompt
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        # Total token count
        total_tokens = 0
        # Total generation time (excludes network latency, prompt building, etc.)
        total_gen_time = 0
        # Number of tool calls
        n_tools = 0
        for turn in range(self.config.get("agent.max_turns", 50)):
            # Send to LLM
            try:
                (result, tokens, gen_elapsed) = self._send_to_llm(prompt_callback,
                                                                  reasoning_callback,
                                                                  content_callback,
                                                                  cancel_callback,
                                                                  error_callback)
            except TurnCancelled:
                # User canceled turn: don't persist anything, return immediately
                return ("[Canceled]", 0, 0, 0.0)

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
                n_tools += self._tool_calls(tool_calls, tool_callback)
                continue  # Loop back to LLM with tool results

            # No tool calls - this is the final response
            self.messages.append({"role": "assistant", "content": response_text})
            
            # Record exchange to chat history
            self.memory.add_chat_exchange(self, "user", user_input)
            self.memory.add_chat_exchange(self, "assistant", response_text)

            # Persist memory
            self.memory.save()
            
            return (response_text, total_tokens, n_tools, total_gen_time)

        # Max turns reached!
        # Persist memory
        self.memory.save()
        return "I've reached the maximum number of turns. Please rephrase your request."

    def _tool_calls(self, tool_calls, tool_callback=None):
        """Handle tool calls"""
        # Append the assistant message with tool calls as plain dictionaries
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })

        n_tools = 0
        # Execute each tool call
        for tc in tool_calls:
            n_tools += 1
            tool_name = tc["function"]["name"]
            tool_args = tc["function"]["arguments"]

            if tool_callback:
                tool_callback(tool_name, tool_args)

            result = execute_tool(tool_name, json.loads(tool_args) if isinstance(tool_args, str) else tool_args)

            # Append tool result
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result if isinstance(result, str) else json.dumps(result),
            })

        return n_tools

    def save_memory(self):
        """Persists the memory"""
        self.memory.save()
        
    def get_models(self):
        """Gets a list with all the available models."""
        try:
            models =  self.router.list_models()
            models = sorted(models, key=lambda model: model['id'])
            return models
        except Exception as e:
            print(e)
            raise RuntimeError("Error getting models. Please check the endpoint is up and reachable.") from e

    def set_model(self, model_name):
        """Sets the model to use."""
        models = self.get_models()
        for model in models:
            if model_name == model['id']:
                # Match, set and return
                self.router.model_name = model_name
                return True

        raise NameError(f"the model '{model_name}' does not exist")
