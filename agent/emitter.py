"""Turn event emitter (client/server phase 1).

The emitter is the boundary between the agent core and whatever is driving it.
Instead of calling Rich/inline UI helpers directly, a driver (today: the REPL
and TUI; tomorrow: the IPC server) hands a :class:`TurnEmitter` to
``Core.run_turn()`` and receives a stream of protocol events.

Phase 1 contract
----------------
* The emitter's methods have the **exact signatures** of the callbacks
  ``Core.run_turn()`` already expects, so it can be passed positionally:
  ``run_turn(text, e.prompt, e.reasoning, e.content, e.tool_call, e.cancelled,
  e.error, poll=e.poll)``.
* Cancellation is **server-owned state**, not an exception raised into the
  core's stack.  :meth:`cancel` sets a flag and :meth:`poll` reports it, which
  is the only cancellation primitive that survives a process boundary.
  ``raise_on_cancel=True`` restores the legacy behaviour (the cancel callback
  raises ``TurnCancelled``) for callers that still rely on it.

Nothing here does I/O beyond handing messages to a :class:`agent.ipc.Transport`.
"""

from __future__ import annotations

import time
import uuid

from typing import Any

from agent.ipc import (
    CancelledPayload,
    ContentPayload,
    ErrorPayload,
    Event,
    Message,
    ReasoningPayload,
    StageKind,
    StagePayload,
    StatusPayload,
    ToolCallPayload,
    ToolResultPayload,
    Transport,
    TurnEndPayload,
    TurnStartPayload,
)

__all__ = ["TurnEmitter", "NullEmitter"]


def _stage_name(stage: Any) -> str:
    """Normalise a stage argument to a :class:`StageKind` value.

    Accepts the ``Stage`` enum from :mod:`agent.core`, a plain string, or
    anything with a ``name`` attribute.  The core is deliberately *not*
    imported so that this module stays free of a core dependency.
    """
    if isinstance(stage, str):
        return stage.lower()
    name = getattr(stage, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(stage).lower()


class TurnEmitter:
    """Translates core callbacks into protocol events on a transport.

    The same object also plays the part of the *driver*: it owns the cancel
    flag that ``poll()`` exposes, and it packages the turn's outcome into a
    :class:`TurnEndPayload`.
    """

    def __init__(
        self,
        transport: Transport | None = None,
        turn_id: str | None = None,
        *,
        raise_on_cancel: bool = False,
        reasoning_visible: bool | None = None,
        echo: bool = True,
    ) -> None:
        self.transport = transport
        self.turn_id = turn_id or uuid.uuid4().hex
        self.raise_on_cancel = raise_on_cancel
        self.reasoning_visible = reasoning_visible
        self.echo = echo
        self.started_at = time.time()
        self._cancelled = False
        self.cancel_reason = ""
        self._tool_index = 0

    # ── outbound ────────────────────────────────────────────────────────────

    def emit(self, name: str | Event, payload: Any = None) -> None:
        """Send an event, if a transport and echo are configured."""
        if self.transport is None or not self.echo:
            return
        self.transport.send(Message.event(name, payload, turn_id=self.turn_id))

    # ── lifecycle ───────────────────────────────────────────────────────────

    def turn_start(self, prompt: str = "") -> None:
        """Announce that a turn is beginning."""
        self.emit(Event.TURN_START, TurnStartPayload(turn_id=self.turn_id, prompt=prompt))

    def turn_end(
        self,
        response: str = "",
        total_tokens: int = 0,
        n_tools: int = 0,
        gen_time: float = 0.0,
        *,
        cancelled: bool = False,
        error: str = "",
    ) -> None:
        """Announce that a turn has finished, whatever the outcome."""
        self.emit(
            Event.TURN_END,
            TurnEndPayload(
                turn_id=self.turn_id,
                response=response,
                total_tokens=total_tokens,
                n_tools=n_tools,
                gen_time=gen_time,
                cancelled=cancelled,
                error=error,
            ),
        )

    def status(self, **fields: Any) -> None:
        """Emit a status snapshot (busy, memory fill, model, ...)."""
        payload = StatusPayload(turn_id=self.turn_id)
        for key, value in fields.items():
            if hasattr(payload, key):
                setattr(payload, key, value)
        self.emit(Event.STATUS, payload)

    # ── core callbacks ──────────────────────────────────────────────────────

    def prompt(self, stage: Any) -> None:
        """Callback for prompt-processing stage changes."""
        self.emit(Event.STAGE, StagePayload(name="prompt", stage=_stage_name(stage)))

    def reasoning(self, stage: Any, content: str = "", visible: bool = True) -> None:
        """Callback for reasoning/thinking deltas.

        The core passes ``content=None`` on START and STOP, where the
        ``visible`` flag carries the only real information. The stage rides
        along so clients can manage "thinking" indicators.
        """
        shown = visible if self.reasoning_visible is None else self.reasoning_visible
        self.emit(
            Event.REASONING,
            ReasoningPayload(
                text=content or "",
                visible=bool(shown),
                stage=_stage_name(stage),
            ),
        )

    def content(self, content: str = "") -> None:
        """Callback for streamed response content."""
        self.emit(Event.CONTENT, ContentPayload(text=content))

    def tool_call(self, tool_name: str, tool_args: Any) -> None:
        """Callback for the moment a tool is about to run."""
        args = tool_args if isinstance(tool_args, str) else str(tool_args)
        self.emit(
            Event.TOOL_CALL,
            ToolCallPayload(
                id="",
                name=tool_name,
                arguments=args,
                index=self._tool_index,
            ),
        )
        self._tool_index += 1

    def tool_result(
        self,
        tool_id: str,
        tool_name: str,
        content: str,
        is_error: bool = False,
        duration: float = 0.0,
        image_base64: str = "",
        mime_type: str = "",
    ) -> None:
        """Callback for a finished tool execution."""
        self.emit(
            Event.TOOL_RESULT,
            ToolResultPayload(
                id=tool_id,
                name=tool_name,
                content=content,
                is_error=is_error,
                duration=duration,
                image_base64=image_base64,
                mime_type=mime_type,
            ),
        )

    def cancelled(self, exc: BaseException | None = None) -> None:
        """Callback invoked by the core when a stream is interrupted.

        In the default (transport-friendly) mode this records the cancellation
        and returns, letting ``run_turn`` notice via :meth:`poll` and finish
        cleanly.  With ``raise_on_cancel=True`` it raises ``TurnCancelled`` so
        that the legacy unwinding behaviour is preserved.
        """
        self.cancel_reason = self.cancel_reason or "user"
        self._cancelled = True
        self.emit(Event.CANCELLED, CancelledPayload(turn_id=self.turn_id, reason=self.cancel_reason))
        if self.raise_on_cancel:
            from agent.core import TurnCancelled

            raise TurnCancelled() from exc

    def error(self, exc: BaseException | None = None, message: str = "") -> None:
        """Callback for unrecoverable errors.

        Emits an error event and re-raises, which preserves the existing
        ``run_turn`` contract (the core persists the partial turn and lets the
        exception escape).
        """
        text = message or str(exc or "Unknown error")
        self.emit(
            Event.ERROR,
            ErrorPayload(message=text, kind="inference", recoverable=False, turn_id=self.turn_id),
        )
        raise RuntimeError(text) from exc

    # ── cancellation ────────────────────────────────────────────────────────

    def cancel(self, reason: str = "user") -> None:
        """Request cancellation of the current turn.

        Safe to call from any thread; the core observes it through
        :meth:`poll` on the next streamed chunk or turn boundary.
        """
        self.cancel_reason = reason
        self._cancelled = True

    def poll(self) -> bool:
        """Cancellation probe handed to ``Core.run_turn(poll=...)``."""
        return self._cancelled

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    def reset(self, turn_id: str | None = None) -> None:
        """Prepare the emitter for a new turn."""
        self.turn_id = turn_id or uuid.uuid4().hex
        self.started_at = time.time()
        self._cancelled = False
        self.cancel_reason = ""
        self._tool_index = 0

    def elapsed(self) -> float:
        """Seconds since the current turn started."""
        return time.time() - self.started_at


class NullEmitter(TurnEmitter):
    """An emitter that discards every event.

    Useful for autonomous/headless runs, and as a default when no driver is
    attached.  Cancellation still works, so a caller can hold a reference and
    call :meth:`cancel`.
    """

    def __init__(self, turn_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(transport=None, turn_id=turn_id, **kwargs)
