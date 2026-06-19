<h3 align="center"><img src="icon.png" alt="Wisemonkey" width="130px"><br>Wisemonkey - <i>A dead simple CLI agent for Linux and macOS</i></h3>

<p align="center">
<a href="https://codeberg.org/langurmonkey/wisemonkey/releases"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcodeberg.org%2Fapi%2Fv1%2Frepos%2Flangurmonkey%2Fwisemonkey%2Freleases%2Flatest&query=%24.tag_name&label=latest%20release" alt="Latest release" /></a>
<a href="https://codeberg.org/langurmonkey/wisemonkey/issues"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcodeberg.org%2Fapi%2Fv1%2Frepos%2Flangurmonkey%2Fwisemonkey%2Fissues&query=%24.length&label=open%20issues" alt="Open issues" /></a>
<a href="https://opensource.org/licenses/MPL-2.0"><img src="https://img.shields.io/badge/license-MIT-brightgreen.svg" alt="License: MPL2.0" /></a>
<img alt="Static Badge" src="https://img.shields.io/badge/OS-Linux-blue">
<img alt="Static Badge" src="https://img.shields.io/badge/OS-macOS-yellow">

</p>

---

[Wisemonkey](https://tonisagrista.com/projects/wisemonkey) is a simple, open, and hackable AI agent for the Linux and macOS terminal. It connects to any service providing an OpenAI, Anthropic, or Ollama-compatible endpoint. It features **session management**, **persistent memory management**, **vector store** for document embedding, native and MCP **tools**, **skills**, and much more.

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

Wisemonkey has been tested to work on Linux and macOS.

### Requirements

- Python 3.13+
- `uv` for dependency management

### Installation

On Linux or macOS, install `uv` and run the agent:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Run wisemonkey
uvx wisemonkey
```

You can also install it with our script:

```bash
curl -fsSL https://codeberg.org/langurmonkey/wisemonkey/raw/branch/master/install.sh | bash
```
This installs wisemonkey to `~/.local/share/wisemonkey/repository`. It adds a `wisemonkey` binary to `~/.local/bin/wisemonkey`. If you have `~/.local/bin` in your `$PATH`, you can launch the onboarding process to configure the agent interactively:

```bash
wisemonkey --onboard
```

### Running

Run the agent with the default session:

```bash
# Using uvx
uvx wisemonkey
# If installed, simply do
wisemonkey
```
For the rest of this document, we assume that `wisemonkey` is in your path. You can substitute it with `uvx wisemonkey` if you use the `uvx` method.

If you need an API key to access the endpoint, put it in the `.env` file. Wisemonkey looks for the `.env` file in the following locations, in order:

- Current directory, `./.env`
- Config directory, `$XDG_CONFIG_HOME/wisemonkey/.env`
- Home directory, `$HOME/.env`

Create the `.env` file with the API key:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
echo "OLLAMA_API_KEY=your-api-key-here" > .env
```

> The agent uses `python-dotenv` to load `.env` at startup. The `openai` package reads `OPENAI_API_KEY` from the environment automatically. You can also set `OPENAI_API_KEY` in your shell profile. Same goes for `ANTHROPIC_API_KEY` and `OLLAMA_API_KEY`.


## Run from source

```bash
# Clone the repo, then build the project:
uv build
# Set API key:
export OPENAI_API_KEY=your-api-key
# Run the agent with the default session:
uv run wisemonkey
```

## Configuration

You can configure the agent interactively before the first run with `wisemonkey --onboard`. On first run, the configuration file is created in `$XDG_CONFIG_HOME/wisemonkey/config.yaml` from the default configuration (`config.yaml`) in the root of this repository.

Additionally, the configuration directory holds the `mcp.json` (see next section), and the `.updates.yml`, which holds information about the last update time and status.

### Model Context Protocol (MCP)

Wisemonkey also supports MCP. Use the following commands to manage the MCP integration:

- `/mcp`: Show the current MCP configuration
- `/mcp edit`: Edit the MCP configuration file (`~/.config/wisemonkey/mcp.json`)
- `/mcp tools`: List all MCP tools available. Alias: `/tools mcp`

MCP servers are started when the agent boots. You need to restart the agent if you add new servers.

## Usage and commands

Run the agent, and then you can enter your prompt. You can use the following key bindings during input:

- <kbd>Alt</kbd> + <kbd>Enter</kbd>: add a new line
- <kbd>Enter</kbd>: submit the prompt
- <kbd>Ctrl</kbd> + <kbd>q</kbd>: quit

During inference, you can cancel the turn and return to the input prompt with <kbd>Ctrl</kbd> + <kbd>c</kbd>

### Sessions

Internally, Wisemonkey uses sessions to separate different memory histories. Sessions are **named by the user**. By default, the agent uses the `default` session. You can start in a different session (either create a new one, or restore it if it exists) by passing its name as a positional argument:

```bash
# Start in a specific session named 'my-project'
wisemonkey my-project
```

The default session's name is `default`, so the following two commands are equivalent:
```bash
# These two commands start the 'default' session
wisemonkey
wisemonkey default
```

You can also list the existing sessions with `-ls`:

```bash
# List sessions
wisemonkey --ls           
Sessions:
- my-project - ~/.local/share/wisemonkey/sessions/my-project
- default - ~/.local/share/wisemonkey/sessions/default
```

Sessions contain:

- The input history
- Chat memory (see [chat memory](#chat-memory))
- Vector store (see [document embedding](#document-embedding))
- Notes (see [session memory](#session-memory))
- User profile (see [session memory](#session-memory))

For now, the configuration file is the same for all sessions.

> Sessions are matched by the directory name in the sessions location (`~/.local/share/wisemonkey/sessions`). You can rename a session by just renaming the directory! 

### `vi` mode

You can enable `vi` mode for the current session with the [command](#commands) `/vi on`, or permanently in the [configuration](#configuration).

**External editor**---In `vi` mode, exit INSERT mode (<kbd>Esc</kbd>), then press <kbd>v</kbd> to edit your prompt in an external editor (uses your `$VISUAL` or `$EDITOR` variable).

### Slash commands

There are a few commands available to use in the agent loop. You can list them with `/help`. Also, use `/[command-name] help` (e.g. `/config help`) to show additional help for a command.

## Session memory

Persistent memory follows XDG Base Directory spec in `~/.local/share/wisemonkey/session/$SESSION_NAME`:

- `user_profile.json`---User information
- `notes.json`---Persistent notes (added via `save_note` tool)

**Lifecycle:**
- Memory is loaded into the system prompt each turn
- `save_note` tool adds notes during a session
- `save_memory` tool explicitly persists memory to disk
- Memory is auto-saved when the agent exits (interactive mode)

## Document embedding

Wisemonkey can embed documents into a per-session vector store, allowing the agent to search and reference their contents during conversation. Use `/embed` to add a document:

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
- Chat history is persisted to `~/.local/share/wisemonkey/session/$SESSION_NAME/chat_history.json`
- Automatically loaded on startup
- Saved after every exchange (user input or assistant response)
- Compacted history is also persisted to disk

**Configuration:**
```yaml
agent:
  max_chat_history: 128000  # Maximum history characters to keep for context
```

## Structure

Wisemonkey is built to be modular and hackable. Here is an overview of the main parts and their mapping to the file system.

```
wisemonkey/
├── agent/                  # Core agent code.
│   ├── agent.py            # Main agent loop, prompt handling, key bindings.
│   ├── commands.py         # Slash commands (e.g. /embed, /quit).
│   ├── config.py           # Configuration loading and handling.
│   ├── console.py          # Rich console output with themed formatting.
│   ├── core.py             # Core agent functions, like API connection and tool calls.
│   ├── mcp.py              # MCP server support.
│   ├── memory.py           # Session memory, paste file creation.
│   ├── router.py           # API router implementation for OpenAI, Ollama, and Anthropic.
│   ├── skills.py           # Skill loading and management.
│   ├── tools.py            # Tool definitions.
│   ├── update.py           # Update management.
│   ├── utils.py            # Utility functions.
│   └── vectorstore.py      # Vector store wrapper.
├── tools/                  # Tool implementations available to the model.
│   ├── basic.py            # Basic and example tools.
│   ├── files.py            # File read/write tools.
│   ├── memory.py            # search_knowledge tool.
│   ├── network.py          # URL fetching.
│   ├── terminal.py         # Shell command execution.
│   └── vectorstore.py      # Vector store tool handler.
├── skills/                 # Skill definitions. Add new skills here.
│   ├── example.md
│   └── rolldice.md
├── tests/                  # Contains all `unittest` tests.
│   └── [...]
├── config.yaml             # Default config file.
├── README.md
├── pyproject.toml
├── install.sh              # Installer script.
└── .env.example
```

## Extend the agent

This agent is simple enough that it can be easily customized and extended by adding new tools, commands, and skills.

If you create a cool new tool, skill, or slash command, consider contributing it via a merge request!

### Adding tools

Create a file in `tools/` or use one of the existing ones. To create a tool,
create a method and decorate it with `@tool(name, description, params)`:

```python
from agent.tools import tool

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
def _cmd_my_command(agent, params) -> (bool, str, str, str):
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
