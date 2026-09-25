"""Wisemonkey server (client/server phase 2).

Owns the ``Core`` (config, memory, router) and serves clients over a Unix
domain socket. Clients connect, attach, and drive turns by sending
``ClientRequest`` messages; the server streams ``TurnEmitter`` events back
and relays interactive ``ServerRequest`` prompts (confirmations, questions)
to the client as RPCs.

Run with ``wmk --server <session>``. In ephemeral mode the client spawns this
process and kills it on exit; in daemon mode it stays up until a
``shutdown`` request or SIGTERM.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
import traceback
from typing import cast
from pathlib import Path

from agent.commands import registry
from agent.core import Core
from agent.emitter import TurnEmitter
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
    ProtocolError,
    ReplyPayload,
    ServerRequest,
    ShutdownPayload,
    TransportClosed,
    TurnEndPayload,
    UnixTransport,
    check_protocol_version,
    payload_as,
    socket_path,
)


class WisemonkeyServer:
    """Single-client server owning the agent core."""

    def __init__(self, config_path: str | None = None, session: str = "default") -> None:
        self.session = session
        self.config_path = config_path
        self.core: Core | None = None
        self.path = socket_path(session)
        self._listen_sock: socket.socket | None = None
        self._shutdown = threading.Event()
        self._turn_lock = threading.Lock()
        self._current_emitter: TurnEmitter | None = None
        self._client: UnixTransport | None = None
        self._client_lock = threading.Lock()
        self._pending_rpc: dict[str, threading.Event] = {}
        self._rpc_replies: dict[str, Message] = {}
        self._rpc_lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Create the core and bind the listening socket."""
        self.core = Core(self.config_path, self.session)

        # Stale socket from a dead server
        if self.path.exists():
            self.path.unlink()

        self.path.parent.mkdir(mode=0o700, exist_ok=True)
        self._listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listen_sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self._listen_sock.listen(1)

        signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum, frame) -> None:
        self._shutdown.set()
        # Unblock the accept loop
        try:
            if self._listen_sock is not None:
                self._listen_sock.close()
        except OSError:
            pass

    def serve_forever(self) -> None:
        """Accept client connections until shutdown."""
        assert self._listen_sock is not None
        while not self._shutdown.is_set():
            try:
                conn, _ = self._listen_sock.accept()
            except OSError:
                break  # closed by SIGTERM/shutdown
            transport = UnixTransport.from_server(conn)
            try:
                self._serve_client(transport)
            except TransportClosed:
                pass
            finally:
                transport.close()
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the socket file and shut the core down."""
        try:
            if self._listen_sock is not None:
                self._listen_sock.close()
        except OSError:
            pass
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        if self.core is not None:
            try:
                self.core.save_memory()
                self.core.shutdown()
            except Exception:
                pass

    # ── client session ──────────────────────────────────────────────────────

    def _serve_client(self, transport: UnixTransport) -> None:
        """Handle one attached client until it detaches or dies."""
        self._client = transport
        try:
            while not self._shutdown.is_set():
                try:
                    message = transport.recv(timeout=0.5)
                except TransportClosed:
                    break
                if message is None:
                    continue
                if not self._dispatch(transport, message):
                    break
        finally:
            # Cancel any running turn when the client goes away
            emitter = self._current_emitter
            if emitter is not None:
                emitter.cancel("client detached")
            self._client = None

    def _send(self, message) -> None:
        transport = self._client
        if transport is not None:
            transport.send(message)

    def _dispatch(self, transport: UnixTransport, message) -> bool:
        """Handle one client message. Returns False to end the session."""
        # Replies to our ServerRequest RPCs
        if message.is_reply and message.reply_to:
            with self._rpc_lock:
                event = self._pending_rpc.pop(message.reply_to, None)
                if event is not None:
                    self._rpc_replies[message.reply_to] = message
                    event.set()
            return True

        if not message.is_request:
            return True  # ignore stray events

        try:
            name = ClientRequest(message.name)
        except ValueError:
            self._send(Message.error(message.id, f"Unknown request: {message.name}"))
            return True

        if name == ClientRequest.ATTACH:
            self._handle_attach(transport, message)
        elif name == ClientRequest.PING:
            payload = payload_as(PingPayload, message)
            self._send(Message.response(message.id, payload))
        elif name == ClientRequest.MEMORY_STATS:
            used, max_tokens, fill_rate = cast(Core, self.core).memory.get_chat_stats()
            self._send(
                Message.response(
                    message.id,
                    MemoryStatsPayload(
                        used=used, max_tokens=max_tokens, fill_rate=fill_rate
                    ),
                )
            )
        elif name == ClientRequest.RECORD:
            payload = payload_as(RecordPayload, message)
            core = cast(Core, self.core)
            core.memory.add_chat_exchange(core, payload.role, payload.content)
            self._send(Message.response(message.id, ReplyPayload(ok=True)))
        if name == ClientRequest.PROMPT:
            # PROMPT and COMMAND can block on client RPCs (ask_* prompts,
            # subprocess runs). Handle them in worker threads so the reader
            # loop below stays free to dispatch the client's replies —
            # handling them inline would deadlock (ask_client waits for a
            # reply that only this loop could read).
            threading.Thread(
                target=self._handle_prompt, args=(message,),
                name="wisemonkey-turn", daemon=True,
            ).start()
        elif name == ClientRequest.COMMAND:
            threading.Thread(
                target=self._handle_command, args=(message,),
                name="wisemonkey-command", daemon=True,
            ).start()
        elif name == ClientRequest.CANCEL:
            payload = payload_as(CancelPayload, message)
            emitter = self._current_emitter
            if emitter is not None:
                emitter.cancel(payload.reason or "user")
            self._send(Message.response(message.id, ReplyPayload(ok=True)))
        elif name == ClientRequest.DETACH:
            self._send(Message.response(message.id, ReplyPayload(ok=True)))
            return False
        elif name == ClientRequest.SHUTDOWN:
            self._send(
                Message.response(
                    message.id, ShutdownPayload(reason="client requested shutdown")
                )
            )
            self._shutdown.set()
            return False
        return True

    # ── request handlers ────────────────────────────────────────────────────

    def _handle_attach(self, transport: UnixTransport, message) -> None:
        payload = payload_as(AttachPayload, message)
        try:
            check_protocol_version(payload.protocol_version)
        except ProtocolError as e:
            self._send(Message.error(message.id, str(e)))
            return

        assert self.core is not None
        handshake = HandshakePayload(
            server_version=_server_version(),
            session=self.session,
            model=self.core.config.get("model.name", ""),
            server_pid=os.getpid(),
            started_at=time.time(),
            session_dir=str(self.core.memory.session_dir),
            working_dir=os.getcwd(),
        )
        self._send(Message.response(message.id, handshake))

    def _handle_prompt(self, message) -> None:
        """Run a full turn, streaming events to the client."""
        assert self.core is not None
        payload = payload_as(PromptPayload, message)
        if not self._turn_lock.acquire(blocking=False):
            self._send(
                Message.error(message.id, "A turn is already in progress")
            )
            return

        emitter = TurnEmitter(transport=self._client)
        self._current_emitter = emitter
        try:
            emitter.turn_start(payload.text)
            result = self.core.run_turn(
                payload.text,
                emitter.prompt,
                emitter.reasoning,
                emitter.content,
                emitter.tool_call,
                emitter.cancelled,
                emitter.error,
                tool_result_callback=emitter.tool_result,
                poll=emitter.poll,
            )
            emitter.turn_end(
                response=result.response,
                total_tokens=result.total_tokens,
                n_tools=result.n_tools,
                gen_time=result.gen_time,
                cancelled=result.cancelled,
                error=result.error,
            )
            self._send(
                Message.response(
                    message.id,
                    TurnEndPayload(
                        turn_id=emitter.turn_id,
                        response=result.response,
                        total_tokens=result.total_tokens,
                        n_tools=result.n_tools,
                        gen_time=result.gen_time,
                        cancelled=result.cancelled,
                        error=result.error,
                    ),
                )
            )
        except TransportClosed:
            emitter.cancel("client detached")
            raise
        except Exception as e:
            self._send(Message.error(message.id, f"Turn failed: {e}"))
        finally:
            self._current_emitter = None
            self._turn_lock.release()

    def _handle_command(self, message) -> None:
        """Run a slash command server-side and return its result."""
        assert self.core is not None
        payload = payload_as(CommandPayload, message)
        from agent.output import IpcOutputAdapter

        output = IpcOutputAdapter(
            transport=self._client, ask_handler=self.ask_client
        )
        try:
            command, params = registry.lookup(payload.raw.split())
            ok_flag, msg, content, markdown, should_exit = registry.run_command(
                self.core, payload.raw, output
            )
            result = CommandResultPayload(
                command=command.name if command else payload.name,
                params=params or payload.params,
                ok=bool(ok_flag),
                msg=msg or "",
                content=content or "",
                markdown=markdown or "",
                should_exit=bool(should_exit),
            )
        except Exception as e:
            result = CommandResultPayload(
                command=payload.name, ok=False, msg=str(e)
            )
        self._send(Message.response(message.id, result))

    # ── ServerRequest RPC relay ─────────────────────────────────────────────

    def ask_client(self, name: ServerRequest, payload):
        """Send a ServerRequest to the client and wait for its reply.

        Used by the output adapter to route interactive prompts
        (confirmations, string/float/choice questions) to the attached UI.
        Returns the unwrapped reply value (e.g. bool/str/float), or None if
        the request failed or was denied.
        """
        transport = self._client
        if transport is None:
            return None

        request = Message.request(name, payload)
        event = threading.Event()
        with self._rpc_lock:
            self._pending_rpc[request.id] = event
        try:
            transport.send(request)
        except TransportClosed:
            with self._rpc_lock:
                self._pending_rpc.pop(request.id, None)
            return None

        if not event.wait(timeout=600):
            with self._rpc_lock:
                self._pending_rpc.pop(request.id, None)
            return None
        with self._rpc_lock:
            reply = self._rpc_replies.pop(request.id, None)
        if reply is None or reply.kind == "error":
            return None
        return payload_as(ReplyPayload, reply).value


def _server_version() -> str:
    try:
        from importlib.metadata import version

        return version("wisemonkey")
    except Exception:
        from agent import __version__

        return __version__


def main(config_path: str | None = None, session: str = "default") -> int:
    """Entry point for ``wmk --server``."""
    server = WisemonkeyServer(config_path, session)
    try:
        server.setup()
    except Exception as e:
        print(f"Server setup failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"[server] session={session} pid={os.getpid()} socket={server.path}")
    sys.stdout.flush()

    # Redirect stdout/stderr to a log file so nothing leaks into the
    # client's terminal.
    log_path = Path.home() / ".local" / "share" / "wisemonkey" / "logs"
    try:
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path / f"{session}.log", "ab", buffering=0)
        sys.stdout = log_file
        sys.stderr = log_file
    except OSError:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.cleanup()
    return 0


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "default"
    sys.exit(main(session=session))