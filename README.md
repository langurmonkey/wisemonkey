<h3 align="center"><img src="icon.png" alt="Langur Agent" width="130px"><br>Langur Agent - <i>A dead simple CLI agent for Linux and macOS</i></h3>

Langur Agent is a simple, open, hackable CLI AI agent for Linux. It supports **tools**, **skills**, and **persistent memory**. It connects to any service providing an OpenAI-compatible endpoint.

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

Add your API key (if any), and run.

```bash
# Set API key:
export LANGUR_API_KEY=your-api-key
# Run the agent:
langur-agent
```

<p align="center">
<img src="screenshot.jpg" 
        alt="Welcome window"
        style="display: block; margin: 0 auto" />
</p>


## Run from source

```bash
# Clone the repo, then build the project:
uv build
# Set API key:
export LANGUR_API_KEY=your-api-key
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
  # Local API key
  api_key: ""
  base_url: "http://127.0.0.1:1234/v1"
  # Temperature setting for inference
  temperature: 0.8
  # Show the model internal thinking
  show_thinking: False

agent:
  max_turns: 50
  system_prompt: "You are a helpful assistant, expert in many domains of science and engineering. Respond concisely and clearly. No fluff. Ask for clarification if needed. Do not invent. On first interaction, analyze the user's message for their name, role, interests, and preferences. Record them with set_user_profile."
  # Display formatted output at the end of generation
  markdown: false
  # Length of chat history kept for context, in characters
  max_chat_history: 128000
  # Enable vi mode input
  vi_mode: false
```

## Usage and commands

Run the agent, and then you can enter your prompt. The input is multiline: Use <kbd>Alt</kbd> + <kbd>Enter</kbd> to add a new line. Use <kbd>Enter</kbd> to submit the prompt.

### `vi` mode

You can enable `vi` mode for the current session with the [command](#commands) `/vi on`, or permanently in the [configuration](#configuration).

**External editor**---In `vi` mode, exit INSERT mode (<kbd>Esc</kbd>), then press <kbd>v</kbd> to edit your prompt in an external editor (uses your `$VISUAL` or `$EDITOR` variable).

### Commands

There are a few commands available to use in the agent loop. You can list them with `/help`. Also, use `/[command-name] help` (e.g. `/config help`) to show additional help for a command.

Below is a listing of all the slash commands.

| Command(s) | Description | Examples |
| :--- | :--- | :--- |
| `/quit`, `/exit`, `/q` | Exit the agent | |
| `/models` | Choose the model to use, interactively | |
| `/notes` | List and add notes to memory | `/notes add The sky is blue`, `/notes list` |
| `/memory` | List memory contents | `/memory`, `/memory agent`, `/memory chat` |
| `/tools` | List available tools | |
| `/skills` | List loaded skills | |
| `/showthinking` | Show or hide thinking tokens | `/showthinking on`, `/showthinking off` |
| `/vi` | Enable or disable vi mode input | `/vi on`, `/vi off` |
| `/config` | Set configuration values or print config | `/config` (list), `/config model.temperature 0.6` |
| `/help`, `/commands` | Show command help | |

The `/config` slash command updates the in-memory configuration and persist it to disk. For properties that need further initialization (like vi mode), please use the appropriate commands (e.g. `/vi on`).

## Global memory

Persistent memory follows XDG Base Directory spec in `~/.local/share/langur-agent/memory/`:
- `user_profile.json` — user information
- `notes.json` — persistent notes (added via `save_note` tool)

**Lifecycle:**
- Memory is loaded into the system prompt each turn
- `save_note` tool adds notes during a session
- `save_memory` tool explicitly persists memory to disk
- Memory is auto-saved when the agent exits (interactive mode)

## Rolling chat memory

In addition to persistent memory, the agent maintains a **rolling chat history** of recent user input and assistant output pairs. This provides context that survives beyond the LLM's context window.

**How it works:**
- Each user message and assistant response is stored in memory
- Automatically trimmed when exceeding the configured character limit
- Attached to the system prompt on each turn
- The agent displays the last 10 exchanges, with long messages truncated

**Persistence:**
- Chat history is persisted to `~/.local/share/langur-agent/memory/chat_history.json`
- Automatically loaded on startup
- Saved after every exchange (user input or assistant response)
- Trimmed history is also persisted to disk

**Configuration:**
```yaml
agent:
  max_chat_history: 128000  # Maximum history characters to keep for context
```

Example chat history format in the prompt:
```
## Recent Conversation
### User
What is the capital of France?
---
### Assistant
The capital of France is Paris.
---
```

## Extend agent

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
                "type": "string"
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

A shash command must return `ok:bool, msg:str, content:str`:

1. `ok`: a `bool` indicating if the command succeeded or failed.
2. `msg`: an optional short message with the status. It is printed with `OK` or `ERROR`.
3. `content`: an optional `str` with the content to print directly to the output.

```python
@cmd(
    "/my-command",
    "This is the description",
    aliases=["/mycmd"],
)
def _cmd_mine(agent, params):
    """This command returns a message but no content"""
    return True, "This is awesome!", None
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

