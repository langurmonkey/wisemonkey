"""Wisemonkey client (client/server phase 2).

A thin proxy that connects to a server over a Unix domain socket, sends
prompts/commands, and delivers protocol events to a consumer callback.
Interactive ``ServerRequest`` prompts from the server (tool confirmations,
questions) are surfaced to the client-side UI via ``request_handler``.

The client also implements the ephemeral co-lifetime: ``ensure_server()``
spawns a server process when none is running and waits for its socket.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from agent.ipc import (
    PROTOCOL_VERSION,
    AttachPayload,
    CancelPayload,
    ClientRequest,
    CommandPayload,
    CommandResultPayload,
    ConfirmPayload,
    Event,
    HandshakePayload,
    Message,
    MemoryStatsPayload,
    PingPayload,
    RecordPayload,
    PromptPayload,
    ReplyPayload,
    ServerRequest,
    ShutdownPayload,
    TransportClosed,
    TurnEndPayload,
    UnixTransport,
    socket_path,
    payload_as,
)


class ServerConnection:
    """Connection to a wisemonkey server over UDS."""

    def __init__(self, transport: UnixTransport) -> None:
        self.transport = transport
        self.handshake: HandshakePayload | None = None
        self._pending: dict[str, queue.Queue[Message | None]] = {}
        self._on_event: Callable[[Message], None] | None = None
        self._req_handlers: dict[ServerRequest, Callable[[Any], Any]] = {}
        self._stop = False
        # Single reader thread: all messages from the server are received
        # here and dispatched (replies -> waiters, events -> callback,
        # server requests -> handlers). Concurrent recv from multiple
        # threads would steal each other's messages.
        self._reader = threading.Thread(
            target=self._read_loop, name="wisemonkey-client-reader", daemon=True
        )
        self._reader.start()

    # ── connection management ───────────────────────────────────────────────

    @classmethod
    def connect(
        cls,
        session: str = "default",
        spawn: bool = True,
        config_path: str | None = None,
        timeout: float = 10.0,
    ) -> "ServerConnection":
        """Connect to the session's server, spawning one if needed."""
        path = socket_path(session)
        transport = None
        try:
            transport = UnixTransport.connect(path, timeout=2.0)
        except (OSError, TransportClosed):
            if not spawn:
                raise
            _spawn_server(session, config_path)
            # Wait for the socket to accept connections
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    transport = UnixTransport.connect(path, timeout=2.0)
                    break
                except (OSError, TransportClosed):
                    time.sleep(0.1)
            if transport is None:
                raise ConnectionError(
                    f"Server for session '{session}' did not start "
                    f"(waited {timeout}s for {path})"
                )
        return cls(transport)

    # ── request/response helpers ────────────────────────────────────────────

    def _request(self, name: ClientRequest, payload: Any = None) -> Message:
        """Send a request and wait for its reply."""
        request = Message.request(name, payload)
        q: queue.Queue[Message | None] = queue.Queue()
        self._pending[request.id] = q
        try:
            self.transport.send(request)
            while True:
                reply = q.get(timeout=600)
                if reply is None:
                    raise TransportClosed("Connection closed while waiting for reply")
                if reply.reply_to == request.id:
                    return reply
        finally:
            self._pending.pop(request.id, None)

    # ── reader thread ────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Receive all server messages and dispatch them."""
        while True:
            try:
                message = self.transport.recv(timeout=0.2)
            except TransportClosed:
                break
            if message is None:
                continue
            if message.reply_to and message.reply_to in self._pending:
                self._pending[message.reply_to].put(message)
            elif message.is_request:
                self._dispatch_server_request(message)
            else:
                if self._on_event is not None:
                    self._on_event(message)

        # Connection gone: wake up all waiters with None.
        for q in self._pending.values():
            q.put(None)
        self._pending.clear()

    def _dispatch_server_request(self, message: Message) -> None:
        try:
            name = ServerRequest(message.name)
        except ValueError:
            return
        handler = self._req_handlers.get(name)
        if handler is None:
            self.transport.send(
                Message.error(message.id, f"No handler for {name}")
            )
            return
        payload = _PAYLOAD_TYPES[name]
        value = handler(payload_as(payload, message))
        self.transport.send(
            Message.response(message.id, ReplyPayload(value=value))
        )

    def _on_early_event(self, message: Message) -> None:
        """Hook for events arriving while waiting for a reply."""
        if self._on_event is not None:
            self._on_event(message)

    def attach(self, client: str = "cli", capabilities: list[str] | None = None) -> HandshakePayload:
        reply = self._request(
            ClientRequest.ATTACH,
            AttachPayload(
                client=client,
                capabilities=capabilities or [],
                client_pid=os.getpid(),
            ),
        )
        if reply.kind == "error":
            raise ConnectionError(f"Attach failed: {reply.payload.get('message')}")
        self.handshake = payload_as(HandshakePayload, reply)
        return self.handshake

    def ping(self) -> bool:
        reply = self._request(ClientRequest.PING, PingPayload(nonce="x"))
        return reply.kind == "response"

    def memory_stats(self) -> tuple[int, int, float]:
        """Return (used, max, fill_rate) chat-memory tokens from the server."""
        reply = self._request(ClientRequest.MEMORY_STATS)
        if reply.kind != "response":
            return 0, 0, 0.0
        payload = payload_as(MemoryStatsPayload, reply)
        return payload.used, payload.max_tokens, payload.fill_rate

    def prompt(self, text: str, on_event: Callable[[Message], None] | None = None) -> TurnEndPayload:
        """Run a turn on the server, streaming events to *on_event*."""
        request = Message.request(ClientRequest.PROMPT, PromptPayload(text=text))
        q: queue.Queue[Message | None] = queue.Queue()
        self._pending[request.id] = q
        prev_on_event = self._on_event
        self._on_event = on_event
        try:
            self.transport.send(request)
            while True:
                message = q.get(timeout=600)
                if message is None:
                    raise TransportClosed("Connection closed during turn")
                if message.reply_to == request.id:
                    return payload_as(TurnEndPayload, message)
        finally:
            self._pending.pop(request.id, None)
            self._on_event = prev_on_event

    def command(self, raw: str) -> CommandResultPayload:
        """Run a slash command server-side."""
        parts = raw.split()
        payload = CommandPayload(
            name=parts[0] if parts else "",
            params=parts[1:],
            raw=raw,
        )
        reply = self._request(ClientRequest.COMMAND, payload)
        return payload_as(CommandResultPayload, reply)

    def cancel(self, reason: str = "user") -> None:
        """Request cancellation of the running turn (fire and forget)."""
        self.transport.send(
            Message.request(ClientRequest.CANCEL, CancelPayload(reason=reason))
        )

    def record(self, role: str, content: str) -> None:
        """Record a client-side exchange into the server's chat history."""
        self._request(ClientRequest.RECORD, RecordPayload(role=role, content=content))

    def shutdown(self) -> None:
        """Ask the server to shut down."""
        try:
            self._request(ClientRequest.SHUTDOWN)
        except TransportClosed:
            pass  # server may close the socket before replying

    def close(self) -> None:
        try:
            self.transport.send(
                Message.request(ClientRequest.DETACH)
            )
        except (TransportClosed, OSError):
            pass
        self.transport.close()

    # ── ServerRequest answering ─────────────────────────────────────────────

    def serve_requests(
        self,
        handlers: dict[ServerRequest, Callable[[Any], Any]],
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Register *handlers* for server requests until shutdown.

        Requests are dispatched by the single reader thread, so this just
        installs the handlers and blocks until the connection closes or
        *should_stop* becomes true.
        """
        self._req_handlers = handlers
        while True:
            if should_stop is not None and should_stop():
                return
            if not self._reader.is_alive():
                return
            time.sleep(0.2)


from agent.ipc import (
    AskChoicePayload,
    AskFloatPayload,
    AskStringPayload,
    RunSubprocessPayload,
)

_PAYLOAD_TYPES = {
    ServerRequest.CONFIRM: ConfirmPayload,
    ServerRequest.ASK_STRING: AskStringPayload,
    ServerRequest.ASK_FLOAT: AskFloatPayload,
    ServerRequest.ASK_CHOICE: AskChoicePayload,
    ServerRequest.RUN_SUBPROCESS: RunSubprocessPayload,
}


def _spawn_server(session: str, config_path: str | None) -> subprocess.Popen:
    """Spawn a server process for *session* (ephemeral co-lifetime)."""
    cmd = [sys.executable, "-m", "agent.server", session]
    if config_path:
        cmd += ["--config", config_path]
    env = os.environ.copy()
    env["WISEMONKEY_SERVER"] = "1"
    # Detached child; the client kills it explicitly on exit. On Linux we
    # also set PR_SET_PDEATHSIG so it dies with the parent even on crash.
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
        "start_new_session": True,
    }
    if sys.platform == "linux":
        kwargs["preexec_fn"] = _set_pdeathsig
    proc = subprocess.Popen(cmd, **kwargs)
    return proc


def _set_pdeathsig() -> None:  # pragma: no cover - runs in the forked child
    """Set PR_SET_PDEATHSIG in the child (runs after fork, before exec)."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass