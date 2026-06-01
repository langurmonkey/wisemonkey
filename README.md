<h3 align="center"><img src="icon.png" alt="Langur Agent" width="130px"><br>Langur Agent - <i>A dead simple CLI agent for Linux and macOS</i></h3>

---

Langur Agent is a simple, open, and hackable AI agent for the Linux and macOS terminal. It connects to any service providing an OpenAI-compatible endpoint. It features:

- session management
- memory management
    - persistent memory
    - memory compaction
    - vector store for documents
- tools
- skills
- autocompletion
- interactive configuration
- visual candy
- and much more

<p align="center">
<a href="https://asciinema.org/a/8cTlvnN0qFeyflLH" target="_blank"><img src="https://asciinema.org/a/8cTlvnN0qFeyflLH.svg" width="60%"/></a>
</p>

The sections of this document are:

- [Quickstart](#quickstart)
- [Run from source](#run-from-source)
- [Configuration](#configuration)
- [Usage and commands](#usage-and-commands)
- [Global memory](#global-memory)
- [Rolling chat memory](#rolling-chat-memory)
- [Extend agent](#extend-agent)

## Quickstart

Langur Agent has been tested on Linux and macOS only.

### Requirements

- Python 3.13+
- `uv` for dependency management

### Installation

Install the agent with:

```bash
curl -fsSL https://codeberg.org/langurmonkey/langur-agent/raw/branch/master/install.sh | bash
```

### Running

Run the agent with the default session:

```bash
langur-agent
```

If you need an API key to access the endpoint, put it in the `.env` file. Langur Agent looks for the `.env` file in the following locations, in order:

- Current directory, `./.env`
- Config directory, `$XDG_CONFIG_HOME/langur-agent/.env`
- Home directory, `$HOME/.env`

Create the `.env` file with the API key:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

> The agent uses `python-dotenv` to load `.env` at startup. The `openai` package reads `OPENAI_API_KEY` from the environment automatically. You can also set `OPENAI_API_KEY` in your shell profile.


## Run from source

```bash
# Clone the repo, then build the project:
uv build
# Set API key:
export OPENAI_API_KEY=your-api-key
# Run the agent:
uv run langur-agent
```

## Configuration

On first run, the configuration is created in `$XDG_CONFIG_HOME/langur-agent/config.yaml`.

It works with any OpenAI-compatible endpoint, so LM Studio, Ollama, OpenWebUI, or any other service you configure. Here are the default values:

```yaml
# Langur Agent Configuration
model:
  # Model name
  name: qwen/qwen3.6-35b-a3b
  # URL of OpenAI endpoint
  base_url: http://127.0.0.1:1234/v1
  # Temperature setting for inference
  temperature: 0.8
  # The reasoning effort. 'none' to disable reasoning
  reasoning_effort: medium
  # Show the model internal thinking
  reasoning_visible: False

embedding:
  # Embedding model name
  name: qwen/qwen3-embedding-0.6b-gguf
  # URL of the OpenAI endpoint for embeddings
  base_url: http://127.0.0.1:1234/v1

agent:
  max_turns: 50
  system_prompt: You are a helpful assistant, expert in many domains of science and engineering. Respond concisely and clearly. No fluff. Ask for clarification if needed. Do not invent. On first interaction, analyze the user's message for their name, role, interests, and preferences. Record them with set_user_profile.
  # Display formatted output at the end of generation
  markdown: false
  # Length of chat history kept for context, in characters
  max_chat_history: 128000
  # Enable vi mode input
  vi_mode: false
```

## Usage and commands

Run the agent, and then you can enter your prompt. You can use the following key bindings during input:

- <kbd>Alt</kbd> + <kbd>Enter</kbd>: add a new line
- <kbd>Enter</kbd>: submit the prompt
- <kbd>Ctrl</kbd> + <kbd>q</kbd>: quit

During inference, you can cancel the turn and return to the input prompt with <kbd>Ctrl</kbd> + <kbd>c</kbd>

### Sessions

Internally, Langur Agent uses sessions to separate different memory histories. Sessions are **named** by the user. By default, the agent uses the `default` session. You can start in a different session (either create a new one, or restore it if it exists) with the `--session` argument:

```bash
# Start in a specific session
langur-agent --session my-project
```

The default session's name is `default`, so the following two commands are equivalent:
```bash
# These two commands start the default session
langur-agent
langur-agent --session default
```

You can also list the existing sessions with `-ls`:

```bash
# List sessions
uv run langur-agent --ls           
Sessions:
- my-project - ~/.local/share/langur-agent/sessions/my-project
- default - ~/.local/share/langur-agent/sessions/default
```

Sessions contain:

- The input history
- Chat memory (see [chat memory](#chat-memory))
- Notes (see [session memory](#session-memory))
- User profile (see [session memory](#session-memory))

For now, the configuration file is the same for all sessions.

> Sessions are matched by the directory name in the sessions location (`~/.local/share/langur-agent/sessions`). You can rename a session by just renaming the directory! 

### `vi` mode

You can enable `vi` mode for the current session with the [command](#commands) `/vi on`, or permanently in the [configuration](#configuration).

**External editor**---In `vi` mode, exit INSERT mode (<kbd>Esc</kbd>), then press <kbd>v</kbd> to edit your prompt in an external editor (uses your `$VISUAL` or `$EDITOR` variable).

### Slash commands

There are a few commands available to use in the agent loop. You can list them with `/help`. Also, use `/[command-name] help` (e.g. `/config help`) to show additional help for a command.

## Session memory

Persistent memory follows XDG Base Directory spec in `~/.local/share/langur-agent/session/$SESSION_NAME`:

- `user_profile.json`---User information
- `notes.json`---Persistent notes (added via `save_note` tool)

**Lifecycle:**
- Memory is loaded into the system prompt each turn
- `save_note` tool adds notes during a session
- `save_memory` tool explicitly persists memory to disk
- Memory is auto-saved when the agent exits (interactive mode)

## Document embedding

Langur Agent can embed documents into a per-session vector store, allowing the agent to search and reference their contents during conversation. Use `/embed` to add a document:

```bash
/embed ~/documents/research_paper.pdf
/embed ./notes.md
```

The agent uses the `search_knowledge` tool to query embedded documents when answering questions about previously indexed files. Supported formats include PDF, Markdown, and plain text. Embeddings are powered by the configured embedding model and stored in the session directory under `vectordb/`.

## Chat memory

In addition to persistent memory, the agent maintains a **chat history** of recent user input and assistant output pairs. This provides context that survives beyond the LLM's context window. Here is how it works:

- Each user message and assistant response is stored in memory
- Reasoning is omitted from chat memory
- Automatically compacted when exceeding the configured character limit
- The user can trigger the compaction any time with `/memory compact`
- Chat memory is attached to the system prompt on each turn
- The agent displays the last 10 exchanges, with long messages truncated

**Persistence:**
- Chat history is persisted to `~/.local/share/langur-agent/session/$SESSION_NAME/chat_history.json`
- Automatically loaded on startup
- Saved after every exchange (user input or assistant response)
- Compacted history is also persisted to disk

**Configuration:**
```yaml
agent:
  max_chat_history: 128000  # Maximum history characters to keep for context
```

## Extend the agent

Langur Agent can be easily customized and extended by adding new tools, commands, and skills.

If you create a cool new tool, skill, or slash command, consider contributing it via a pull request!

### Adding tools

Create a file in `tools/` or use one of the existing ones. To create a tool,
create a method and decorate it with `@tool(name, description, params)`:

```python
from langur.tools import tool

@tool(
    name="my_tool",
    description="Does something useful. Be exhaustive here, as it is what the LLM will read to know about your tool.",
    parameters={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The input parameter."
            }
        },
        "required": ["input"],
    },
)
def my_handler(args):
    input = args.get("input", "no input provided")
    return {"result": f"{input}"}
```

Tools are auto-discovered on startup.

### Adding slash commands

The process is very similar to tools. You need to create your method, preferably in `agent/commands.py`, and decorate it with `@cmd(name, description, aliases, examples, can_complete)`.

A slash command must return, in that order, `ok:bool`, `msg:str`, `content:str`, `markdown:str`:

1. `ok`: a `bool` indicating if the command succeeded or failed.
2. `msg`: an optional short status message. It is printed with `OK` or `ERROR`.
3. `content`: an optional `str` with the Python Rich-formatted content, it is printed to the output.
4. `markdown`: an optional `str` formatted in Markdown, it is printed to the output.

```python
@cmd(
    "/my-command",
    "This is the description",
    aliases=["/mycmd"],
)
def _cmd_mine(agent, params):
    """This command returns a message but no content"""
    return True, "This is awesome!", None, None
```

Decorated commands are automatically registered, and auto-completed in the input prompt.

### Adding skills

Add a `.md` file in `skills/` with YAML front matter, following the [agentskills.io](https://agentskills.io) standard:

```markdown
---
name: my-skill
description: What this skill does
---

# My skill

## When to use

...

## Steps

1. ...
```

The front matter `name` and `description` are parsed and shown in the
skills list. The body is injected into the system prompt.

