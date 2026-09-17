"""Wisemonkey IPC protocol (phase 0).

This module defines the wire protocol that lets a Wisemonkey *client*
(the classic REPL or the Textual TUI) talk to a Wisemonkey *server*
(``Core``, tools, memory, router) over a local transport.

Nothing here performs I/O beyond the in-process :class:`LoopbackTransport`.
The concrete Unix-domain-socket transport lands in phase 2; phase 1 wires the
existing ``Core.run_turn()`` callbacks through an emitter that speaks this
protocol.

Design
------
* **Envelope** — every frame is a :class:`Message` with a ``kind``
  (event / request / response / error), a ``name``, a JSON ``payload``, a
  unique ``id`` and an optional ``reply_to`` for correlation.
* **Direction** — the server pushes *events*; the client sends *requests*.
  The server may also send *requests* of its own (confirmations and other UI
  prompts) which the client answers with *responses*.  This makes the protocol
  fully bidirectional and lets ``OutputAdapter`` calls cross the process
  boundary unchanged.
* **Framing** — newline-delimited JSON (see :func:`encode_message` and
  :func:`decode_message`).  One frame per line, UTF-8, no embedded newlines.

Payload dataclasses are the *typed* form of the payloads.  On the wire they
are plain JSON objects, so enum members arrive as their string values and
tuples arrive as lists.
"""

from __future__ import annotations

import json
import queue
import time
import uuid

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

__all__ = [
    "PROTOCOL_VERSION",
    "ProtocolError",
    "TransportClosed",
    "MessageKind",
    "Event",
    "ClientRequest",
    "ServerRequest",
    "InjectWhen",
    "StageKind",
    "OutputFormat",
    "OutputLevel",
    "ClientCapability",
    "ServerCapability",
    "Message",
    "HandshakePayload",
    "TurnStartPayload",
    "TurnEndPayload",
    "StagePayload",
    "ContentPayload",
    "ReasoningPayload",
    "ToolCallPayload",
    "ToolResultPayload",
    "OutputPayload",
    "StatusPayload",
    "ErrorPayload",
    "InjectedPayload",
    "CancelledPayload",
    "PongPayload",
    "ShutdownPayload",
    "AttachPayload",
    "PromptPayload",
    "CommandPayload",
    "CommandResultPayload",
    "InjectPayload",
    "CancelPayload",
    "PingPayload",
    "ConfirmPayload",
    "AskStringPayload",
    "AskFloatPayload",
    "AskChoicePayload",
    "RunSubprocessPayload",
    "ReplyPayload",
    "Transport",
    "LoopbackTransport",
    "loopback_pair",
    "encode_message",
    "decode_message",
    "payload_as",
    "check_protocol_version",
]


# ----------------------------------------------------------------
# Version
# ----------------------------------------------------------------

#: Wire protocol version. Bump the major number on breaking changes; clients
#: and servers refuse to talk across a major-version mismatch.
PROTOCOL_VERSION = "1.0"


class ProtocolError(Exception):
    """Raised on malformed frames or an incompatible protocol version."""


class TransportClosed(Exception):
    """Raised when reading from or writing to a closed transport."""


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a ``"major.minor"`` version string into an int tuple."""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as e:
        raise ProtocolError(f"Malformed protocol version: {version!r}") from e


def check_protocol_version(version: str) -> None:
    """Validate *version* against :data:`PROTOCOL_VERSION`.

    Only the major component must match; minor versions are compatible.
    Raises :class:`ProtocolError` on mismatch.
    """
    remote = _parse_version(version)
    local = _parse_version(PROTOCOL_VERSION)
    if not remote or not local or remote[0] != local[0]:
        raise ProtocolError(
            f"Incompatible protocol version: local={PROTOCOL_VERSION}, "
            f"remote={version}"
        )


# ----------------------------------------------------------------
# Enumerations
# ----------------------------------------------------------------

class MessageKind(StrEnum):
    """Top-level frame kind."""

    EVENT = "event"        # server -> client, unsolicited
    REQUEST = "request"    # either direction, expects a response
    RESPONSE = "response"  # reply to a request (``reply_to`` set)
    ERROR = "error"        # reply to a request that failed


class Event(StrEnum):
    """Server -> client events."""

    HANDSHAKE = "handshake"
    ATTACHED = "attached"
    DETACHED = "detached"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    STAGE = "stage"
    CONTENT = "content"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OUTPUT = "output"
    COMMAND_RESULT = "command_result"
    STATUS = "status"
    INJECTED = "injected"
    CANCELLED = "cancelled"
    ERROR = "error"
    PONG = "pong"
    SHUTDOWN = "shutdown"


class ClientRequest(StrEnum):
    """Client -> server requests."""

    ATTACH = "attach"
    DETACH = "detach"
    PROMPT = "prompt"
    COMMAND = "command"
    INJECT = "inject"
    CANCEL = "cancel"
    PING = "ping"
    SHUTDOWN = "shutdown"


class ServerRequest(StrEnum):
    """Server -> client requests (mirrors ``OutputAdapter``)."""

    CONFIRM = "confirm"
    ASK_STRING = "ask_string"
    ASK_FLOAT = "ask_float"
    ASK_CHOICE = "ask_choice"
    RUN_SUBPROCESS = "run_subprocess"


class InjectWhen(StrEnum):
    """When an injected message should be delivered."""

    BETWEEN_TURNS = "between_turns"  # queued until the current turn ends
    AFTER_TOOL = "after_tool"        # delivered at the next tool-loop seam
    INTERRUPT = "interrupt"          # cancel the stream, then deliver


class StageKind(StrEnum):
    """Lifecycle stage of a streaming phase."""

    START = "start"
    PROCESS = "process"
    STOP = "stop"


class OutputFormat(StrEnum):
    """How the client should render an :class:`OutputPayload`."""

    TEXT = "text"          # plain text, no markup
    MARKUP = "markup"      # rich markup
    ANSI = "ansi"          # pre-rendered ANSI (opaque renderables)
    MARKDOWN = "markdown"  # markdown body
    PANEL = "panel"        # markdown body wrapped in a titled panel
    RULE = "rule"          # horizontal rule


class OutputLevel(StrEnum):
    """Semantic level of an output line."""

    NORMAL = "normal"
    INFO = "info"
    OK = "ok"
    ERR = "err"
    WARN = "warn"


class ClientCapability(StrEnum):
    """Things a client can do locally on behalf of the server."""

    UI_PROMPTS = "ui_prompts"      # can answer ServerRequest prompts
    SUBPROCESS = "subprocess"      # can run a program with terminal control
    SCREENSHOT = "screenshot"      # can capture the local screen
    CLIPBOARD = "clipboard"
    EDITOR = "editor"


class ServerCapability(StrEnum):
    """Features the server advertises in the handshake."""

    COMMANDS = "commands"
    INJECTION = "injection"
    AUTONOMOUS = "autonomous"
    VECTORSTORE = "vectorstore"
    MCP = "mcp"


# ----------------------------------------------------------------
# Payloads
# ----------------------------------------------------------------


@dataclass
class HandshakePayload:
    """Server response to an ``attach`` request."""

    protocol_version: str = PROTOCOL_VERSION
    server_version: str = ""
    session: str = "default"
    model: str = ""
    server_pid: int = 0
    started_at: float = 0.0
    capabilities: list[str] = field(default_factory=list)
    session_dir: str = ""
    working_dir: str = ""


@dataclass
class AttachPayload:
    """Client -> server ``attach`` request."""

    client: str = "cli"
    protocol_version: str = PROTOCOL_VERSION
    capabilities: list[str] = field(default_factory=list)
    read_only: bool = False
    client_pid: int = 0


@dataclass
class TurnStartPayload:
    """A turn has begun."""

    turn_id: str = ""
    prompt: str = ""


@dataclass
class TurnEndPayload:
    """A turn has finished (normally, canceled or with an error)."""

    turn_id: str = ""
    response: str = ""
    total_tokens: int = 0
    n_tools: int = 0
    gen_time: float = 0.0
    cancelled: bool = False
    error: str = ""


@dataclass
class StagePayload:
    """Start/stop marker for a streaming phase."""

    name: str = ""            # "prompt" | "reasoning" | "generation"
    stage: str = "start"      # StageKind value
    visible: bool = True


@dataclass
class ContentPayload:
    """A streamed content delta."""

    text: str = ""


@dataclass
class ReasoningPayload:
    """A reasoning delta or lifecycle marker.

    ``stage`` is a :class:`StageKind` value so clients can show and dismiss
    "thinking" indicators (START/STOP) as well as render deltas (PROCESS).
    """

    text: str = ""
    visible: bool = True
    stage: str = StageKind.PROCESS


@dataclass
class ToolCallPayload:
    """The model requested a tool."""

    id: str = ""
    name: str = ""
    arguments: str = ""
    index: int = 0


@dataclass
class ToolResultPayload:
    """A tool finished executing."""

    id: str = ""
    name: str = ""
    content: str = ""
    is_error: bool = False
    duration: float = 0.0
    image_base64: str = ""
    mime_type: str = ""


@dataclass
class OutputPayload:
    """A UI output line (mirrors ``OutputAdapter`` print methods)."""

    format: str = OutputFormat.TEXT
    text: str = ""
    end: str = "\n"
    indent: int = 0
    level: str = OutputLevel.NORMAL
    title: str = ""
    subtitle: str = ""
    border_style: str = ""
    style: str = "dim"      # for RULE
    align: str = "center"   # for RULE


@dataclass
class StatusPayload:
    """Server status snapshot."""

    busy: bool = False
    turn_id: str = ""
    queue_depth: int = 0
    model: str = ""
    session: str = ""
    memory_chars: int = 0
    memory_max: int = 0
    memory_rate: float = 0.0
    uptime: float = 0.0


@dataclass
class ErrorPayload:
    """An error, either unsolicited or in reply to a request."""

    message: str = ""
    kind: str = "error"
    recoverable: bool = False
    turn_id: str = ""


@dataclass
class InjectedPayload:
    """Confirmation that an injected message was delivered."""

    text: str = ""
    when: str = InjectWhen.BETWEEN_TURNS
    role: str = "user"
    turn_id: str = ""


@dataclass
class CancelledPayload:
    """A turn was cancelled."""

    turn_id: str = ""
    reason: str = "user"


@dataclass
class PongPayload:
    """Reply to ``ping``."""

    nonce: str = ""


@dataclass
class ShutdownPayload:
    """The server is shutting down."""

    reason: str = ""
    exit_code: int = 0


@dataclass
class PromptPayload:
    """Client -> server ``prompt`` request."""

    text: str = ""
    image_base64: str = ""
    mime_type: str = ""


@dataclass
class CommandPayload:
    """Client -> server ``command`` request (slash command)."""

    name: str = ""
    params: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class CommandResultPayload:
    """Result of a slash command."""

    command: str = ""
    params: list[str] = field(default_factory=list)
    ok: bool = True
    msg: str = ""
    content: str = ""
    markdown: str = ""
    should_exit: bool = False


@dataclass
class InjectPayload:
    """Client -> server ``inject`` request."""

    text: str = ""
    when: str = InjectWhen.BETWEEN_TURNS
    role: str = "user"


@dataclass
class CancelPayload:
    """Client -> server ``cancel`` request."""

    turn_id: str = ""
    reason: str = "user"


@dataclass
class PingPayload:
    """Liveness probe."""

    nonce: str = ""


@dataclass
class ConfirmPayload:
    """Server -> client ``confirm`` request."""

    message: str = ""
    default: bool = False


@dataclass
class AskStringPayload:
    """Server -> client ``ask_string`` request."""

    message: str = ""
    default: str = ""


@dataclass
class AskFloatPayload:
    """Server -> client ``ask_float`` request."""

    message: str = ""
    default: float = 0.0


@dataclass
class AskChoicePayload:
    """Server -> client ``ask_choice`` request.

    ``options`` is a list of ``[value, label]`` pairs.
    """

    message: str = ""
    options: list[list[str]] = field(default_factory=list)
    default: str = ""


@dataclass
class RunSubprocessPayload:
    """Server -> client ``run_subprocess`` request."""

    cmd: list[str] = field(default_factory=list)
    cwd: str = ""


@dataclass
class ReplyPayload:
    """Client -> server answer to a :class:`ServerRequest`."""

    value: Any = None
    ok: bool = True


# ----------------------------------------------------------------
# Envelope
# ----------------------------------------------------------------

def _payload_dict(payload: Any) -> dict[str, Any]:
    """Normalise a payload argument into a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return asdict(payload)


@dataclass
class Message:
    """A single protocol frame."""

    kind: str = MessageKind.EVENT
    name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reply_to: str = ""
    turn_id: str = ""
    ts: float = field(default_factory=time.time)

    # ── constructors ────────────────────────────────────────────────────

    @classmethod
    def event(
        cls,
        name: str | Event,
        payload: Any = None,
        turn_id: str = "",
    ) -> "Message":
        """Build an unsolicited server -> client event."""
        return cls(
            kind=MessageKind.EVENT,
            name=str(name),
            payload=_payload_dict(payload),
            turn_id=turn_id,
        )

    @classmethod
    def request(cls, name: str | ClientRequest | ServerRequest, payload: Any = None) -> "Message":
        """Build a request (either direction)."""
        return cls(
            kind=MessageKind.REQUEST,
            name=str(name),
            payload=_payload_dict(payload),
        )

    @classmethod
    def response(cls, reply_to: str, payload: Any = None) -> "Message":
        """Build a successful reply to a request."""
        return cls(
            kind=MessageKind.RESPONSE,
            name="reply",
            payload=_payload_dict(payload),
            reply_to=reply_to,
        )

    @classmethod
    def error(cls, reply_to: str, message: str, **extra: Any) -> "Message":
        """Build a failed reply to a request."""
        payload: dict[str, Any] = {"message": message}
        payload.update(extra)
        return cls(
            kind=MessageKind.ERROR,
            name="error",
            payload=payload,
            reply_to=reply_to,
        )

    # ── serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Rebuild a message from :meth:`to_dict` output."""
        kind = data.get("kind")
        if kind not in tuple(MessageKind):
            raise ProtocolError(f"Unknown message kind: {kind!r}")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError("Message is missing a name")
        payload = data.get("payload") or {}
        if not isinstance(payload, dict):
            raise ProtocolError("Message payload must be an object")
        return cls(
            kind=kind,
            name=name,
            payload=payload,
            id=str(data.get("id") or uuid.uuid4().hex),
            reply_to=str(data.get("reply_to") or ""),
            turn_id=str(data.get("turn_id") or ""),
            ts=float(data.get("ts") or time.time()),
        )

    @property
    def is_event(self) -> bool:
        return self.kind == MessageKind.EVENT

    @property
    def is_request(self) -> bool:
        return self.kind == MessageKind.REQUEST

    @property
    def is_reply(self) -> bool:
        return self.kind in (MessageKind.RESPONSE, MessageKind.ERROR)


# ----------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------

def encode_message(message: Message) -> str:
    """Serialize *message* to a single newline-terminated JSON line."""
    return json.dumps(message.to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n"


def decode_message(line: str) -> Message:
    """Parse a single JSON line into a :class:`Message`.

    Raises :class:`ProtocolError` on malformed input.
    """
    text = line.strip()
    if not text:
        raise ProtocolError("Empty frame")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"Malformed JSON frame: {e}") from e
    if not isinstance(data, dict):
        raise ProtocolError("Frame must be a JSON object")
    return Message.from_dict(data)


T = TypeVar("T")


def payload_as(cls: type[T], message: Message) -> T:
    """Reconstruct a payload dataclass from *message*.

    Unknown keys are ignored so that newer peers can add fields without
    breaking older ones.
    """
    factory: Any = cls
    known = {f.name for f in factory.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in message.payload.items() if k in known}
    return cast(T, factory(**kwargs))


# ----------------------------------------------------------------
# Transports
# ----------------------------------------------------------------

@runtime_checkable
class Transport(Protocol):
    """A bidirectional, message-oriented channel.

    Implementations must be safe to use from a single reader thread and a
    single writer thread concurrently.  Phase 2 provides a Unix-domain-socket
    implementation; phase 1 uses :class:`LoopbackTransport`.
    """

    def send(self, message: Message) -> None:
        """Send a message.  Raises :class:`TransportClosed` if closed."""
        ...

    def recv(self, timeout: float | None = None) -> Message | None:
        """Receive a message.

        Blocks until a message arrives, or up to *timeout* seconds
        (``None`` blocks forever, ``0`` polls).  Returns ``None`` on timeout
        and raises :class:`TransportClosed` when the peer is gone.
        """
        ...

    def close(self) -> None:
        """Close the transport, waking any blocked reader."""
        ...

    @property
    def closed(self) -> bool:
        """Whether this transport has been closed."""
        ...


_CLOSED = object()


class LoopbackTransport:
    """In-process transport backed by two queues.

    Used in phase 1 to prove the protocol seam without any real I/O: the
    server and client run in the same process and exchange :class:`Message`
    objects directly.
    """

    def __init__(self, inbox: "queue.Queue[Any]", outbox: "queue.Queue[Any]") -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, message: Message) -> None:
        if self._closed:
            raise TransportClosed("Transport is closed")
        self._outbox.put(message)

    def recv(self, timeout: float | None = None) -> Message | None:
        if self._closed and self._inbox.empty():
            raise TransportClosed("Transport is closed")
        try:
            item = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _CLOSED:
            self._closed = True
            raise TransportClosed("Peer closed the transport")
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._outbox.put(_CLOSED)


def loopback_pair() -> tuple[LoopbackTransport, LoopbackTransport]:
    """Create a connected pair of in-process transports.

    Returns ``(a, b)`` such that a message sent on ``a`` is received on ``b``
    and vice versa.
    """
    a_to_b: "queue.Queue[Any]" = queue.Queue()
    b_to_a: "queue.Queue[Any]" = queue.Queue()
    return (
        LoopbackTransport(inbox=b_to_a, outbox=a_to_b),
        LoopbackTransport(inbox=a_to_b, outbox=b_to_a),
    )
