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

prompt_toolkit 3.0.52 does not parse the ``CSI u`` form at all, so
:func:`extend_ansi_sequences` teaches its parser about it: modified Enter,
Ctrl+C, and every other functional key the flag re-encodes.

The flag does not only re-encode the keys it mentions above, it changes the
whole key encoding: every non-text key now carries a modifier field, so a
plain arrow becomes ``CSI 1;129 A`` once num_lock is active rather than
``CSI A``. prompt_toolkit's table has no entry with a lock modifier set, so
*every* functional key -- arrows, Home/End, Delete, the F-keys, Backspace,
the keypad -- arrives unknown and has its own escape code typed into the
prompt. :func:`extended_sequences` registers the lot, over the whole modifier
range, so that pressing a key does the obvious thing on any terminal that
speaks the protocol.

On a terminal that does not support the protocol, ``CSI > 1 u`` is ignored
and every key arrives in its legacy encoding, which prompt_toolkit's table
already covers; the extra entries are simply never matched. Nothing depends
on the protocol being available except Shift+Enter (and, with it, Ctrl+Enter
as a distinct key) -- see :func:`install_enter_bindings` for what still works
everywhere.

Ctrl+C is the one key that must *keep* working, and the flag breaks it: the
terminal reports ``CSI 99 ; mods u`` instead of ``0x03``, so the kernel never
raises SIGINT. :func:`extend_ansi_sequences` maps those back to ``c-c`` for
prompt_toolkit, and :func:`is_ctrl_c_sequence` lets :mod:`agent.steer` -- which
reads the tty raw, with no prompt_toolkit involved -- recognise the same bytes
and raise SIGINT itself.

Both the enable and the disable sequence must be sent. A terminal left in
disambiguate mode after the process dies keeps sending key codes in the
``CSI u`` form, which breaks whatever the user runs next -- so the disable is
registered with :mod:`atexit` as well as called on the way out.
"""

from __future__ import annotations

import atexit
import os
import sys

from functools import lru_cache
from typing import cast

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
CSI_U_CTRL_C = "\x1b[99u"  # unicode-key-code 99 == 'c'
_CSI_U_MODIFIER_VALUES = range(1, 258)

CSI_U_ENTER_SEQUENCES = {CSI_U_ENTER} | {
    f"\x1b[13;{mods}u" for mods in _CSI_U_MODIFIER_VALUES
}

# Ctrl+C is the other key the disambiguate flag changes. With it on, the
# terminal no longer sends 0x03 for Ctrl+C -- it reports the *key* as
# ``CSI 99 ; mods u``, which the kernel never sees, so no SIGINT is ever
# raised. Both consumers of Ctrl+C therefore need the sequence:
#
# - the prompt (``@kb.add("c-c")`` in agent/agent.py) sees an unknown key and
#   types the raw escape code into the buffer;
# - mid-turn there is no prompt at all -- agent/steer.py reads the tty raw,
#   so it has to recognise the sequence itself and raise SIGINT itself.
#
# Same reasoning as Enter: the modifier field is open-ended, so cover it all.
#
# The bare ``CSI 99 u`` is deliberately *not* included: a modifier value of 1
# means "no modifiers", which a disambiguating terminal never reports (it
# keeps sending unmodified keys as plain bytes). Mapping it would turn the
# letter "c" into Ctrl+C.
CSI_U_CTRL_C_SEQUENCES = {
    f"\x1b[99;{mods}u" for mods in _CSI_U_MODIFIER_VALUES if mods != 1
}

# --------------------------------------------------------------------------
# Everything the disambiguate flag changes besides Enter and Ctrl+C.
# --------------------------------------------------------------------------
#
# The flag does not only change the *encoding* of a key, it changes which keys
# carry a modifier field at all. A plain arrow used to be the three bytes
# ``CSI A``; once a lock modifier is active it is ``CSI 1;129 A``, because
# num_lock (128) is reported like any other modifier. prompt_toolkit's table
# knows ``CSI 1;2A`` (Shift+Up) and ``CSI A``, but nothing with a lock bit set,
# so the arrow arrives as an *unknown* key and its own escape code is typed
# into the prompt. The same applies to every functional key: the CSI ``u``
# form for Backspace, the ``CSI number;mods~`` form for Delete and F-keys, the
# keypad, and xterm's ``modifyOtherKeys`` spelling of a modified Enter.
#
# So all of it is registered here, for the whole modifier range. The keys are
# the spec's own functional-key table; nothing is hand-picked.

# Lock modifiers. They describe the state of the keyboard rather than a key the
# user is holding, so they are stripped before deciding what the key is:
# ``CSI 1;129A`` is Up, not some shift-y variant of it.
LOCK_MODIFIER_BITS = 64 | 128

# ``CSI 1;mods <letter>`` -- the functional keys that keep the legacy letter
# form under the protocol (kitty sends them this way, it does not move them to
# the ``CSI u`` form).
CSI_LETTER_KEYS: dict[str, Keys] = {
    "A": Keys.Up,
    "B": Keys.Down,
    "C": Keys.Right,
    "D": Keys.Left,
    "F": Keys.End,
    "H": Keys.Home,
    "P": Keys.F1,
    "Q": Keys.F2,
    "S": Keys.F4,
}

# ``CSI number;mods ~`` -- the functional keys the spec keeps in this form.
CSI_TILDE_KEYS: dict[int, Keys] = {
    2: Keys.Insert,
    3: Keys.Delete,
    5: Keys.PageUp,
    6: Keys.PageDown,
    7: Keys.Home,
    8: Keys.End,
    11: Keys.F1,
    12: Keys.F2,
    13: Keys.F3,
    14: Keys.F4,
    15: Keys.F5,
    17: Keys.F6,
    18: Keys.F7,
    19: Keys.F8,
    20: Keys.F9,
    21: Keys.F10,
    23: Keys.F11,
    24: Keys.F12,
}

# ``CSI number;mods u`` for the keys the spec encodes as a Unicode codepoint.
# Values are prompt_toolkit key names, or a literal string to insert.
CSI_U_FUNCTIONAL_KEYS: dict[int, str] = {
    2: Keys.Insert.value,
    3: Keys.Delete.value,
    5: Keys.PageUp.value,
    6: Keys.PageDown.value,
    7: Keys.Home.value,
    8: Keys.End.value,
    9: Keys.ControlI.value,
    27: Keys.Escape.value,
    # Backspace has no modified key name in prompt_toolkit (no ``s-backspace``,
    # no ``c-backspace``), and needs none: every modifier variant should
    # delete backwards exactly like the unmodified key. Ctrl+Backspace is
    # conventionally a word delete, but there is no binding for that here and
    # inventing one would silently change what an existing binding means.
    127: Keys.ControlH.value,
    # Keypad. Digits and operators are sent as themselves rather than as keys,
    # so that they land in the buffer as text; the navigation keys map to their
    # non-keypad equivalents, which is what the spec says to do.
    57399: "0",  # KP_0
    57400: "1",
    57401: "2",
    57402: "3",
    57403: "4",
    57404: "5",
    57405: "6",
    57406: "7",
    57407: "8",
    57408: "9",
    57410: "/",  # KP_DIVIDE
    57411: "*",  # KP_MULTIPLY
    57412: "-",  # KP_SUBTRACT
    57413: "+",  # KP_ADD
    57414: Keys.ControlM.value,  # KP_ENTER
    57415: "=",  # KP_EQUAL
    57416: ",",  # KP_SEPARATOR
    57417: Keys.Left.value,
    57418: Keys.Right.value,
    57419: Keys.Up.value,
    57420: Keys.Down.value,
    57421: Keys.PageUp.value,
    57422: Keys.PageDown.value,
    57423: Keys.Home.value,
    57424: Keys.End.value,
    57425: Keys.Insert.value,
    57426: Keys.Delete.value,
    # KP_BEGIN has no prompt_toolkit equivalent; Home is the nearest thing
    # that moves the cursor somewhere predictable.
    57427: Keys.Home.value,
}

# F13 and up only ever exist in the ``CSI u`` form. prompt_toolkit names F1-F24
# and nothing beyond, so F25-F35 have no key to map to: they are left
# unregistered rather than mapped to something wrong. A terminal that sends
# them gets an unknown key, which prompt_toolkit ignores -- not an escape code
# in the prompt, because the parser sees the full sequence and finds no match.
CSI_U_F13_BASE = 57376
CSI_U_F35_BASE = 57398


def _f_key_name(number: int) -> str | None:
    """prompt_toolkit's name for spec key code *number*, if it has one."""
    name = f"f{number - CSI_U_F13_BASE + 13}"
    return name if any(k.value == name for k in Keys) else None


def _with_modifiers(key: str, bits: int) -> str:
    """Spell *key* with the shift/ctrl bits of *bits*, as prompt_toolkit does.

    prompt_toolkit names modified keys ``s-`` then ``c-`` prefixed (``s-left``,
    ``c-s-left``). Lock bits are dropped before we get here. Modifiers it has
    no name for -- alt, super, hyper, meta -- leave the key unmodified: there
    is nothing to bind, and mapping to the base key keeps the cursor moving
    instead of typing an escape code into the prompt.
    """
    prefix = ""
    if bits & 4:
        prefix = "c-"
    if bits & 1:
        prefix += "s-"
    return prefix + key


@lru_cache(maxsize=None)
def _resolved_key(key: str, bits: int) -> str | Keys:
    """The prompt_toolkit mapping for *key* pressed with modifier *bits*.

    Three fallbacks, in order of preference:

    1. the modified name -- ``s-left``, ``c-up``, ``c-s-delete``;
    2. the unmodified key -- there is no ``s-c-h`` (Shift+Backspace) or
       ``s-c-m`` (Shift+KP_Enter), but the base key is still the right thing
       to do rather than nothing;
    3. the key as given -- only reachable for a single character (a keypad
       digit), which prompt_toolkit's parser inserts as text.

    A third fallback to a *multi-character* string would be a bug:
    ``KeyPress`` asserts the key is a ``Keys`` or exactly one character, so a
    miss there raises AssertionError mid-parse rather than degrading.

    Cached because the tables are walked once per modifier value (256 of them)
    over the same handful of keys, and the lock bits mean the distinct
    *held* combinations are only 16 -- without the cache this is a few hundred
    thousand enum lookups at startup.
    """
    candidate = _with_modifiers(key, bits)
    try:
        return Keys(candidate)
    except ValueError:
        pass
    if bits:
        try:
            return Keys(key)
        except ValueError:
            pass
    return key


def extended_sequences() -> dict[str, str | Keys | tuple[Keys, ...]]:
    """Every sequence the disambiguate flag adds, beyond Enter and Ctrl+C.

    Values are normally a ``Keys``; a keypad digit or operator is a
    one-character ``str``, which prompt_toolkit's parser inserts as text --
    ``KeyPress.__init__`` accepts it and asserts exactly that, even though the
    table's annotation does not mention it.

    Kept as a function rather than a module-level constant because it is built
    once per modifier value over a handful of tables, and building it lazily
    keeps the cost off the import path.
    """
    sequences: dict[str, str | Keys | tuple[Keys, ...]] = {}
    for mods in _CSI_U_MODIFIER_VALUES:
        if mods == 1:
            # 1 means "no modifiers", which is the legacy encoding (the field
            # is omitted entirely). Registering it would turn the letter "c"
            # into a cancel and bare "a" into an arrow.
            continue
        bits = mods - 1
        held = bits & ~LOCK_MODIFIER_BITS

        for letter, key in CSI_LETTER_KEYS.items():
            sequences[f"\x1b[1;{mods}{letter}"] = _resolved_key(key.value, held)

        for number, key in CSI_TILDE_KEYS.items():
            sequences[f"\x1b[{number};{mods}~"] = _resolved_key(key.value, held)

        for number, key in CSI_U_FUNCTIONAL_KEYS.items():
            if number == 9 and held & 1:
                # Shift+Tab is a distinct key in prompt_toolkit, not a modified
                # tab, and the spec agrees: it is one key with two identities.
                sequences[f"\x1b[{number};{mods}u"] = Keys.BackTab
            else:
                sequences[f"\x1b[{number};{mods}u"] = _resolved_key(key, held)

        for number in range(CSI_U_F13_BASE, CSI_U_F35_BASE + 1):
            name = _f_key_name(number)
            if name is not None:
                sequences[f"\x1b[{number};{mods}u"] = _resolved_key(name, held)

        # xterm's modifyOtherKeys spelling of a modified Enter. prompt_toolkit
        # has 2/5/6; with a lock modifier active xterm reports 130 and friends,
        # which then arrive as an unknown key.
        sequences[f"\x1b[27;{mods};13~"] = Keys.ControlM
    return sequences


_extended = False

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

    from typing import cast

    from prompt_toolkit.input import ansi_escape_sequences

    # Enter and Ctrl+C are registered last, and unconditionally overwrite:
    # they are the two keys whose encoding is shared with a plain character
    # the parser must never see (CR and 0x03), so the table we build for the
    # other keys must not be able to shadow them.
    for sequence, key in extended_sequences().items():
        # Do not clobber a mapping prompt_toolkit already provides. The table
        # is *not* wrong everywhere -- for the legacy spellings it decodes as
        # an `escape` prefix (``CSI 1;3D`` -> escape, left, so an `escape left`
        # binding applies) that is right on a terminal not using the protocol,
        # and shadowing it would break Alt+Left there. Where the two readings
        # genuinely conflict -- ``CSI 1;2P`` is Shift+F1 under this protocol
        # and F13 under the xterm/linux-console convention -- upstream wins,
        # because both readings are function keys the prompt does not bind and
        # nothing is lost either way. What matters is that we never *lose* a
        # mapping: a miss is an escape code typed into the prompt.
        if sequence not in ansi_escape_sequences.ANSI_SEQUENCES:
            ansi_escape_sequences.ANSI_SEQUENCES[sequence] = cast(
                "Keys | tuple[Keys, ...]", key
            )
    for sequence in sorted(CSI_U_CTRL_C_SEQUENCES):
        ansi_escape_sequences.ANSI_SEQUENCES[sequence] = Keys.ControlC
    for sequence in sorted(CSI_U_ENTER_SEQUENCES):
        ansi_escape_sequences.ANSI_SEQUENCES[sequence] = Keys.ControlM

    _extended = True


def is_ctrl_c_sequence(sequence: str) -> bool:
    """True when *sequence* is a kitty-protocol encoding of Ctrl+C.

    For readers that do not go through prompt_toolkit -- :mod:`agent.steer`
    reads the tty byte by byte -- so they can recognise Ctrl+C themselves and
    raise SIGINT, which is what the kernel would have done had the terminal
    not intercepted the key.
    """
    return sequence in CSI_U_CTRL_C_SEQUENCES


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
    - ``Alt+Enter`` (``escape enter``) inserts a newline. This works on every
      terminal in existence, and is the only way to add a newline on macOS
      Terminal.app, where Option has to be configured as Meta.
    - ``Ctrl+J`` inserts a newline. Same reasoning: one key that works
      everywhere, and prompt_toolkit's own default for it in a multiline
      prompt is a newline -- but an explicit ``enter`` binding resolves the
      same key on some terminals and a bound handler wins, which made Ctrl+J
      silently submit the prompt instead of continuing the line.

    On a terminal that does not implement the kitty keyboard protocol,
    ``extend_ansi_sequences`` registers nothing that can match, because none of
    these sequences is ever sent: ``Alt+Enter`` and ``Ctrl+J`` remain, and
    Shift+Enter and Ctrl+Enter both arrive as a plain Enter and submit.

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