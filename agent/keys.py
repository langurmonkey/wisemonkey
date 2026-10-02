"""Terminal key-protocol helpers: telling modified Enter apart from Enter.

A terminal sends the *same* byte (``0x0D``) for Enter, Shift+Enter and
Ctrl+Enter, so a line editor cannot offer Shift+Enter as "insert a newline"
unless the terminal is asked to disambiguate. Two mechanisms do that:

- **xterm ``modifyOtherKeys``** — some terminals send ``CSI 27 ; mods ; 13 ~``
  for a modified Enter with no configuration. prompt_toolkit's table already
  maps that to ``c-m``, and the raw sequence survives in ``event.data``; see
  :func:`is_modified_enter`.
- **Kitty keyboard protocol** (``CSI > flags u`` / ``CSI < u``) — a stronger
  mechanism supported by kitty, Ghostty, Alacritty, WezTerm, foot and Rio.
  We enable and disable it ourselves (no user configuration needed). Nothing
  is gated on ``TERM``: a terminal that does not understand the sequence
  ignores it, so a wrong guess costs nothing, whereas a process killed in
  enhanced mode would break the shell that takes over.

prompt_toolkit 3.0.52 does not parse the ``CSI u`` form, so
:func:`extend_ansi_sequences` teaches its parser about it. The sequences are
mapped to ``c-m`` like every other Enter; what distinguishes them is the raw
data on the event, which is why the bindings must test it rather than relying
on distinct key names.

Both the enable and the disable sequence must be sent. A terminal left in
disambiguate mode after the process dies keeps sending key codes in the
``CSI u`` form, which breaks whatever the user runs next -- so the disable is
registered with :mod:`atexit` as well as called on the way out.
"""

from __future__ import annotations

import atexit
import os
import sys

from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding.key_processor import KeyPress

# The flag we ask for: "disambiguate escape codes". This is the only flag
# needed to tell Shift/Ctrl/Alt+Enter apart from Enter.
DISAMBIGUATE_FLAG = 1

# Push/pop the keyboard protocol for the current terminal.
PUSH = f"\x1b[>{DISAMBIGUATE_FLAG}u"
POP = "\x1b[<u"

# Emulators known to implement the kitty keyboard protocol. `TERM` is the
# primary signal; `TERM_PROGRAM` covers the macOS ones that keep a generic TERM.
SUPPORTED_TERMS = frozenset({
    "alacritty",
    "contour",
    "foot",
    "foot-extra",
    "ghostty",
    "rio",
    "st",
    "wezterm",
    "xterm-ghostty",
    "xterm-kitty",
})
SUPPORTED_TERM_PROGRAMS = frozenset({
    "ghostty",
    "iTerm.app",
    "MacTerm",
    "WezTerm",
    "rio",
})

# CSI-u encodings of Enter, as sent once the disambiguate flag is on:
#
#     CSI unicode-key-code ; modifiers u
#
# with `unicode-key-code` = 13 (CR) and `modifiers` encoded as
# ``1 + bitfield`` -- shift 1, alt 2, ctrl 4, super 8, hyper 16, meta 32,
# caps_lock 64, num_lock 128. So Shift+Enter is 2, Alt+Enter 3, Ctrl+Enter 5,
# and with num_lock on (the common case, and what a real terminal sends)
# Shift+Enter is **130**, Ctrl+Enter **133**, Alt+Enter **131** -- not the
# 2/3/5 that hand-written tables tend to guess.
#
# The field is open-ended, so every value is registered rather than a
# hand-picked few: a miss is not a degraded feature, it is a literal
# "ESC[13;130u" typed into the prompt.
CSI_U_ENTER = "\x1b[13u"
_CSI_U_MODIFIER_VALUES = range(1, 258)

CSI_U_ENTER_SEQUENCES = {CSI_U_ENTER} | {
    f"\x1b[13;{mods}u" for mods in _CSI_U_MODIFIER_VALUES
}

# Modifier bit names, in bit order. Index 0 is unused (the encoded value is
# 1 + bits, so 1 itself means "no modifiers").
MODIFIER_BITS: tuple[tuple[int, str], ...] = (
    (1, "shift"),
    (2, "alt"),
    (4, "ctrl"),
    (8, "super"),
    (16, "hyper"),
    (32, "meta"),
    (64, "caps_lock"),
    (128, "num_lock"),
)


def describe_enter_sequence(sequence: str) -> str:
    """Human-readable name for an Enter sequence (debug aid).

    ``\\x1b[13u`` is a bare Enter; ``\\x1b[13;130u`` decodes to
    ``enter+shift+num_lock``.
    """
    if sequence == CSI_U_ENTER:
        return "enter"
    if not sequence.startswith("\x1b[13;") or not sequence.endswith("u"):
        return sequence
    try:
        # -1: strip the "+1 = no modifiers" encoding before decoding bits.
        bits = int(sequence[len("\x1b[13;"):-1]) - 1
    except ValueError:
        return sequence
    names = [name for bit, name in MODIFIER_BITS if bits & bit]
    return "+".join(["enter"] + names)


_extended = False


def extend_ansi_sequences() -> None:
    """Teach prompt_toolkit's parser the ``CSI u`` form of modified Enter.

    Idempotent, and a no-op for a prompt_toolkit that already knows the
    sequences, so calling it once per prompt session is safe.

    The sequences map to ``Keys.ControlM`` exactly like a plain Enter. What
    separates them is the raw text, which prompt_toolkit keeps in
    ``KeyPress.data`` and exposes as ``event.data`` -- see
    :func:`is_modified_enter`.

    ``CSI_U_ENTER_SEQUENCES`` covers the whole modifier range rather than a
    hand-listed few, because a real terminal reports whatever lock modifiers
    happen to be active: ``\\x1b[13;130u`` is Shift+Enter *with num_lock*, and
    a table that only knows ``13;2u`` makes the terminal print its own escape
    code into the prompt.
    """
    global _extended
    if _extended:
        return

    from prompt_toolkit.input import ansi_escape_sequences

    for sequence in CSI_U_ENTER_SEQUENCES:
        # Do not clobber a mapping a newer prompt_toolkit already provides.
        if sequence not in ansi_escape_sequences.ANSI_SEQUENCES:
            ansi_escape_sequences.ANSI_SEQUENCES[sequence] = Keys.ControlM

    _extended = True


def is_modified_enter(event) -> bool:
    """True when *event* is an Enter carrying a modifier.

    Both encodings end up as ``c-m``, and both are recognised here:

    - ``CSI 27 ; mods ; 13 ~`` (xterm ``modifyOtherKeys``) -- already in
      prompt_toolkit's table.
    - ``CSI 13 ; mods u`` (kitty) -- only after :func:`extend_ansi_sequences`.

    A plain Enter is a bare carriage return, so the test is simply "did the
    terminal send an escape sequence with it". Comparing against the decoded
    key name would not work: both forms decode to ``c-m``.
    """
    data = getattr(event, "data", "") or ""
    return data.startswith("\x1b[")


def terminal_supports_kitty_protocol() -> bool:
    """Whether this terminal is known to implement the kitty keyboard protocol.

    Deliberately conservative: an emulator not on either list is treated as
    unsupported, which costs the user Shift+Enter but never risks sending
    sequences to something that mis-handles them. tmux is detected via
    ``TERM`` as well and needs ``set -g extended-keys on`` plus
    ``extended-keys-format csi-u`` to pass sequences through.
    """
    term = os.environ.get("TERM", "")
    if term in SUPPORTED_TERMS:
        return True
    if term.startswith("screen") or term.startswith("tmux"):
        # Only if the user configured tmux to forward enhanced keys; we cannot
        # tell from here, and sending is harmless either way.
        return True
    program = os.environ.get("TERM_PROGRAM", "")
    return program in SUPPORTED_TERM_PROGRAMS


def _write(sequence: str) -> bool:
    """Write *sequence* to the terminal, bypassing prompt_toolkit's writer.

    Returns False when there is no usable terminal, so callers can degrade
    quietly instead of raising in a REPL.
    """
    if not sys.stdout.isatty():
        return False
    try:
        sys.stdout.write(sequence)
        sys.stdout.flush()
    except (OSError, ValueError):
        return False
    return True


def enable_kitty_keyboard() -> bool:
    """Push the disambiguate flag. Returns whether it was sent.

    A terminal that does not understand ``CSI > 1 u`` ignores it silently, so
    a False return here is not an error: it just means Shift+Enter will keep
    arriving as a plain Enter and only Alt+Enter will work.
    """
    pushed = _write(PUSH)
    if pushed:
        # The terminal must not be left in enhanced mode if we are killed.
        atexit.register(_write, POP)
    return pushed


def disable_kitty_keyboard() -> None:
    """Pop the disambiguate flag, restoring the terminal's default mode."""
    _write(POP)


def install_enter_bindings(kb) -> None:
    """Add the Enter-family bindings to a ``KeyBindings`` collection.

    One rule: **Enter submits, a modified Enter inserts a newline.**

    - ``Enter`` submits, unless the terminal sent an escape sequence with it
      (a modifier), in which case it continues the line.
    - ``Alt+Enter`` (``escape enter``) inserts a newline. This is the only
      option on terminals that cannot distinguish Shift+Enter -- notably
      macOS Terminal.app, where Option has to be configured as Meta.
    - ``Ctrl+J`` inserts a newline. It is the one newline key that works on
      every terminal in existence, and prompt_toolkit's own default for it in a
      multiline prompt is a newline, but an explicit ``enter`` binding resolves
      the same key for some terminals and a bound handler wins -- which made
      Ctrl+J silently submit the prompt instead of continuing the line.

    The bindings assume :func:`extend_ansi_sequences` has been called, which
    is what lets the kitty protocol's ``CSI u`` form be recognised at all.
    """
    extend_ansi_sequences()

    @kb.add("enter")
    def _enter(event):
        if is_modified_enter(event):
            event.current_buffer.insert_text("\n")
            return
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _alt_enter(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-j")
    def _ctrl_j(event):
        event.current_buffer.insert_text("\n")


def describe_event(event: KeyPress) -> str:
    """Debug helper: name the binding an event would take."""
    data = getattr(event, "data", "") or ""
    return describe_enter_sequence(data) if data != "\r" else "enter"