# Wisemonkey Client/Server Protocol

This document specifies the wire protocol between a Wisemonkey **client**
(REPL, TUI, or any third-party frontend) and a **server** (daemon) that owns
the session authoritatively: `Core`, chat memory, profile/notes, vector store,
tool execution, and persistence.

The reference implementation lives in `agent/ipc.py`; the server is
`agent/server.py` and the client connection is `agent/client.py`
(`ServerConnection`).

- **Version:** `1.0` (`PROTOCOL_VERSION`)
- **Transport:** Unix domain socket at
  `$XDG_RUNTIME_DIR/wisemonkey/$SESSION.sock` (Linux) or
  `$TMPDIR/wisemonkey-$USER/$SESSION.sock` (macOS fallback).
- **Framing:** newline-delimited JSON (NDJSON). Each frame is a single line
  terminated by `\n`, UTF-8 encoded, compact separators.
- **Compatibility:** the protocol major version must match exactly
  (`check_protocol_version()`). Payload reconstruction (`payload_as()`)
  ignores unknown keys, so newer peers may add fields without breaking older
  ones.

## Message envelope

Every frame is a JSON object with these fields:

| Field      | Type   | Description                                             |
|------------|--------|---------------------------------------------------------|
| `kind`     | string | `event`, `request`, `response`, or `error`              |
| `name`     | string | Message kind name (see tables below)                    |
| `payload`  | object | Message-specific fields                                 |
| `id`       | string | Unique frame id (UUID hex, auto-generated)              |
| `reply_to` | string | `id` of the request this frame replies to (replies only)|
| `turn_id`  | string | Turn this message belongs to (events mostly)            |
| `ts`       | float  | Unix timestamp of frame creation                        |

### Frame kinds (`MessageKind`)

- `event` — unsolicited server → client message (streaming output, status).
- `request` — either direction; expects a `response` or `error` reply.
- `response` — successful reply; `reply_to` is set, `name` is `"reply"`.
- `error` — failed reply; payload contains at least `message`.

## Session lifecycle

1. **Connect** — client connects a stream socket to the session's socket path.
2. **Attach** — client sends an `attach` request
   (`AttachPayload`: `client`, `protocol_version`, `capabilities`,
   `read_only`, `client_pid`).
3. **Handshake** — server replies with a `handshake` event
   (`HandshakePayload`: `protocol_version`, `server_version`, `session`,
   `model`, `server_pid`, `started_at`, `capabilities`, `session_dir`,
   `working_dir`). A client only attaches to a server running the **same
   session name**; otherwise it starts its own independent session.
4. **Work** — prompts, commands, events, and RPCs flow (below).
5. **Detach / close** — client sends `detach` or simply closes the socket.
   The server keeps running and its state survives client restarts.

## Client → server requests

All are `request` frames; each is answered with a `response` (payload listed)
or an `error`.

| Name           | Payload           | Reply payload       | Description |
|----------------|-------------------|---------------------|-------------|
| `attach`       | `AttachPayload`   | handshake event     | Register the client; server replies with capabilities. |
| `detach`       | —                 | —                   | Client is leaving; server frees the attachment. |
| `prompt`       | `PromptPayload` (`text`, `image_base64`, `mime_type`) | `TurnEndPayload` | Run a full agent turn. While it runs the server streams `turn_start`, `stage`, `content`, `reasoning`, `tool_call`, `tool_result`, `output`, and finally `turn_end` events, all carrying the same `turn_id`. |
| `command`      | `CommandPayload` (`name`, `params`, `raw`) | `CommandResultPayload` (`command`, `params`, `ok`, `msg`, `content`, `markdown`, `should_exit`) | Execute a slash command server-side. `should_exit` means the *client* should exit (not the server). |
| `cancel`       | `CancelPayload` (`turn_id`, `reason`) | — (fire and forget) | Request cancellation of the running turn. The server sets observable cancel state; the turn ends with `cancelled: true` in `TurnEndPayload`. |
| `ping`         | `PingPayload` (`nonce`) | `PongPayload` (`nonce`) | Liveness probe. |
| `memory_stats` | —                 | `MemoryStatsPayload` (`used`, `max_tokens`, `fill_rate`) | Chat-memory usage in tokens. |
| `record`       | `RecordPayload` (`role`, `content`) | ack | Record a client-side exchange (e.g. a locally executed `!` shell command) into the server's chat history without running a turn. |
| `inject`       | `InjectPayload` (`text`, `when`, `role`) | `InjectedPayload` | Queue a message for delivery into the running or next turn. `when` is an `InjectWhen` value (see below). |
| `shutdown`     | —                 | `ShutdownPayload`   | Ask the server to stop. |

### Injection timing (`InjectWhen`)

- `between_turns` — queued until the current turn ends; delivered before the
  next prompt.
- `after_tool` — delivered at the next tool-loop seam (after the current tool
  batch, before the model's next reasoning step).
- `interrupt` — cancels the running stream, then delivers the message at the
  next seam.

## Server → client requests (RPCs)

The server may need client-local capabilities (interactive prompts, running
programs with terminal control). These are `request` frames answered by the
client with a `ReplyPayload` (`value`, `ok`):

| Name             | Payload                | Expected reply value |
|------------------|------------------------|----------------------|
| `confirm`        | `ConfirmPayload` (`message`, `default`) | boolean |
| `ask_string`     | `AskStringPayload` (`message`, `default`) | string |
| `ask_float`      | `AskFloatPayload` (`message`, `default`) | float |
| `ask_choice`     | `AskChoicePayload` (`message`, `options` as `[value, label]` pairs, `default`) | selected value |
| `run_subprocess` | `RunSubprocessPayload` (`cmd`, `cwd`) | subprocess result — **executed client-side** with full terminal control |

If a client has no handler for a server request, it replies with an `error`
frame.

## Server → client events

Unsolicited `event` frames, mostly tied to the running turn via `turn_id`:

| Name            | Payload              | Description |
|-----------------|----------------------|-------------|
| `handshake`     | `HandshakePayload`   | Reply to `attach` (sent as an event). |
| `attached` / `detached` | —            | Attachment lifecycle notices. |
| `turn_start`    | `TurnStartPayload` (`turn_id`, `prompt`) | A turn began. |
| `turn_end`      | `TurnEndPayload` (`turn_id`, `response`, `total_tokens`, `n_tools`, `gen_time`, `cancelled`, `error`) | The turn finished; also the reply to `prompt`. |
| `stage`         | `StagePayload` (`name`: `prompt`\|`reasoning`\|`generation`, `stage`: `start`\|`process`\|`stop`, `visible`) | Streaming phase lifecycle (spinners). |
| `content`       | `ContentPayload` (`text`) | Streamed content delta. |
| `reasoning`     | `ReasoningPayload` (`text`, `visible`, `stage`) | Reasoning delta or thinking-indicator lifecycle. |
| `tool_call`     | `ToolCallPayload` (`id`, `name`, `arguments`, `index`) | The model requested a tool. |
| `tool_result`   | `ToolResultPayload` (`id`, `name`, `content`, `is_error`, `duration`, `image_base64`, `mime_type`) | A tool finished. |
| `output`        | `OutputPayload` (`format`, `text`, `end`, `indent`, `level`, `title`, `subtitle`, `border_style`, `style`, `align`) | Pre-rendered UI output so the server needs no Rich console. `format` is an `OutputFormat` (`text`, `markup`, `ansi`, `markdown`, `panel`, `rule`); `level` is an `OutputLevel` (`normal`, `info`, `ok`, `err`, `warn`). |
| `command_result`| `CommandResultPayload` | Result of a slash command (also sent as an event for broadcast scenarios). |
| `status`        | `StatusPayload` (`busy`, `turn_id`, `queue_depth`, `model`, `session`, memory fields, `uptime`) | Server status snapshot. |
| `injected`      | `InjectedPayload` (`text`, `when`, `role`, `turn_id`) | Confirmation an injected message was delivered. |
| `cancelled`     | `CancelledPayload` (`turn_id`, `reason`) | A turn was cancelled. |
| `error`         | `ErrorPayload` (`message`, `kind`, `recoverable`, `turn_id`) | Unsolicited error. |
| `pong`          | `PongPayload`        | Reply to `ping` (also sent as an event). |
| `shutdown`      | `ShutdownPayload` (`reason`, `exit_code`) | The server is shutting down. |

## Capabilities

Exchanged during the handshake so peers can degrade gracefully:

- **Client** (`ClientCapability`): `ui_prompts` (can answer server RPC
  prompts), `subprocess`, `screenshot`, `clipboard`, `editor`.
- **Server** (`ServerCapability`): `commands`, `injection`, `autonomous`,
  `vectorstore`, `mcp`.

## Transports

Two implementations of the `Transport` protocol (`send`, `recv(timeout)`,
`close`, `closed`):

- `UnixTransport` — NDJSON over a connected Unix domain socket (production).
- `LoopbackTransport` / `loopback_pair()` — in-process pair used by local
  (non-daemon) mode and tests.

## Client-side concurrency rule

`ServerConnection` runs a **single reader thread** that receives all frames
and dispatches them: replies → per-request queues, events → the current
turn's `on_event` callback, server requests → RPC handlers. Never call
`recv()` from multiple threads — concurrent readers steal each other's
messages and corrupt request/reply matching.

## Example session

```
C → S  {"kind":"request","name":"attach","payload":{"client":"cli",
        "protocol_version":"1.0","capabilities":["ui_prompts",
        "subprocess"],"read_only":false,"client_pid":12345}, ...}
S → C  {"kind":"event","name":"handshake","payload":{"protocol_version":
        "1.0","server_version":"2026.9.23","session":"default",
        "model":"qwen/qwen3.6-35b-a3b","server_pid":1681519,
        "capabilities":["commands","injection"], ...}, ...}
C → S  {"kind":"request","name":"prompt","payload":{"text":"List the agent dir"}, ...}
S → C  {"kind":"event","name":"turn_start","payload":{"turn_id":"t1", ...}, "turn_id":"t1"}
S → C  {"kind":"event","name":"tool_call","payload":{"id":"1","name":"list_dir", ...}, "turn_id":"t1"}
S → C  {"kind":"event","name":"tool_result","payload":{"id":"1","name":"list_dir","content":"..."}, "turn_id":"t1"}
S → C  {"kind":"event","name":"content","payload":{"text":"Here are the files..."}, "turn_id":"t1"}
S → C  {"kind":"response","reply_to":"<prompt id>","payload":{"turn_id":"t1",
        "response":"Here are the files...","total_tokens":1523,"n_tools":1,
        "gen_time":4.2,"cancelled":false,"error":""}, ...}
```
