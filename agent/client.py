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
import signal
import subprocess
import sys
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
    PingPayload,
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
        self.transport.send(request)
        while True:
            reply = self.transport.recv(timeout=600)
            if reply is None:
                continue
            if reply.reply_to == request.id:
                return reply
            # Not our reply (e.g. an interleaved event) — queue it for the
            # event consumer.
            self._on_early_event(reply)

    def _on_early_event(self, message: Message) -> None:
        """Hook for events arriving while waiting for a reply."""
        self._early_events.append(message)

    _early_events: list[Message] = []

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

    def prompt(self, text: str, on_event: Callable[[Message], None] | None = None) -> TurnEndPayload:
        """Run a turn on the server, streaming events to *on_event*."""
        request = Message.request(ClientRequest.PROMPT, PromptPayload(text=text))
        self.transport.send(request)
        while True:
            message = self.transport.recv(timeout=600)
            if message is None:
                continue
            if message.reply_to == request.id:
                return payload_as(TurnEndPayload, message)
            if message.is_event and on_event is not None:
                on_event(message)

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
        """Answer ServerRequest RPCs on a background thread.

        *handlers* maps request kinds to functions taking the payload and
        returning the reply value.
        """
        while True:
            if should_stop is not None and should_stop():
                return
            try:
                message = self.transport.recv(timeout=0.2)
            except TransportClosed:
                return
            if message is None:
                continue
            if not message.is_request:
                continue
            try:
                name = ServerRequest(message.name)
            except ValueError:
                continue
            handler = handlers.get(name)
            if handler is None:
                self.transport.send(
                    Message.error(message.id, f"No handler for {name}")
                )
                continue
            payload = _PAYLOAD_TYPES[name]
            value = handler(payload_as(payload, message))
            self.transport.send(
                Message.response(message.id, ReplyPayload(value=value))
            )


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