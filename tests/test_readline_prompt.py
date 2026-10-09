"""End-to-end: a Rich prompt must survive a readline redraw.

The failure this pins is entirely about *bytes on the wire*. Rich prints the
question and then calls ``input()`` with an empty prompt, so readline believes
the line buffer starts at column 1; the first redraw (Backspace) rewrites from
column 1 and erases the question. Nothing a unit test with a mocked prompt can
observe: it needs a real pty, a real readline, and a cursor tracker replaying
what reached the terminal.
"""

import os
import re
import unittest


# Exactly the question ``/model`` asks, markup and all.
PROMPT_MARKUP = (
    "[bold]Enter the endpoint URL[/bold] "
    "[dim](https://opencode.ai/zen/go/v1)[/dim]:"
)

# "hello" -> Left,Left -> "X" -> "helXlo" -> Backspace -> "hello"
#          -> Right, "!" -> "hell!o"
KEYS = [b"hello", b"\x1b[D", b"\x1b[D", b"X", b"\x7f", b"\x1b[C", b"!"]
EXPECTED = "hell!o"

# What Rich appends after the markup ("prompt_suffix").
PROMPT_SUFFIX = ": "


def prompt_visible_width() -> int:
    """Columns the question occupies once the styles are stripped."""
    text = PROMPT_MARKUP
    while "[" in text:
        start = text.index("[")
        end = text.index("]", start)
        text = text[:start] + text[end + 1 :]
    return len(text + PROMPT_SUFFIX)


def _replay_erases(data: bytes, prompt_cols: int) -> list[int]:
    """Columns at which an erase-to-end-of-line fell inside the question.

    A cursor tracker, deliberately minimal: it understands only the sequences
    readline actually emits when redrawing (``\\r``, ``BS``, ``CUB``/``CUF``,
    ``CUP``-by-column and ``EL``). Anything else is skipped, which is safe
    because an unrecognised sequence cannot move the cursor in a way that
    matters here.
    """
    col = 1
    bad: list[int] = []
    i = 0
    while i < len(data):
        byte = data[i : i + 1]
        if byte == b"\x1b":
            match = re.match(rb"\x1b\[([0-9;?]*)([a-zA-Z])", data[i : i + 16])
            if match:
                params = [int(n) for n in match.group(1).split(b";") if n.isdigit()]
                arg = params[0] if params else None
                final = match.group(2)
                if final == b"G":  # CHA: cursor horizontal absolute
                    col = arg or 1
                elif final == b"K" and arg in (None, 0):  # EL / EL0
                    if col <= prompt_cols:
                        bad.append(col)
                elif final == b"C":  # CUF
                    col += arg or 1
                elif final == b"D":  # CUB
                    col -= arg or 1
                i += match.end()
                continue
            i += 1
            continue
        if byte == b"\r":
            col = 1
        elif byte == b"\n":
            col = 1
        elif byte == b"\x08":
            col -= 1
        elif byte >= b" ":
            col += 1
        i += 1
    return bad


def _drive_pty(use_rich_prompt_class: bool) -> bytes:
    """Run one prompt under a pty, type the key sequence, return the bytes.

    ``use_rich_prompt_class=False`` drives the readline-aware prompt, i.e. what
    ``RichOutputAdapter`` does now. The flag exists so the test can also drive
    the original Rich behaviour and show it is genuinely different.
    """
    import fcntl
    import pty
    import select
    import struct
    import sys
    import termios
    import time

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child = f"""
import sys
sys.path.insert(0, {repo!r})

from agent.keys import ensure_readline
ensure_readline()

from rich.console import Console
from rich.prompt import Prompt as RichPrompt

from agent.console import monkee_theme
from agent.output import _ReadlinePrompt

console = Console(theme=monkee_theme, color_system="truecolor")
cls = RichPrompt if {use_rich_prompt_class!r} else _ReadlinePrompt
value = cls.ask({PROMPT_MARKUP!r}, console=console)
sys.stderr.write("GOT=%r\\n" % value)
"""

    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        try:
            os.execv(sys.executable, [sys.executable, "-u", "-c", child])
        finally:
            os._exit(0)

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))

    data = bytearray()

    def drain(timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            data.extend(chunk)

    drain(1.5)  # let the child start and print the question
    for key in KEYS:
        os.write(fd, key)
        drain(0.5)
    os.write(fd, b"\r")
    drain(2.0)

    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return bytes(data)


class TestPromptSurvivesRedraw(unittest.TestCase):
    """Backspace and friends must not eat the question."""

    def test_the_question_is_never_erased(self):
        data = _drive_pty(use_rich_prompt_class=False)
        self.assertIn(b"GOT=", data, f"child never answered: {data!r}")

        bad = _replay_erases(data, prompt_visible_width())
        self.assertEqual(
            bad, [], f"erase inside the question at columns {bad}: {data!r}"
        )

    def test_the_answer_is_what_was_typed(self):
        data = _drive_pty(use_rich_prompt_class=False)
        match = re.search(rb"GOT=(.*)", data)
        assert match is not None, data
        got = match.group(1).decode(errors="replace").strip().strip("'")
        self.assertEqual(got, EXPECTED)

    def test_rich_own_behaviour_really_does_erase(self):
        """Guard the guard: without the prompt handed to readline it breaks.

        If Rich ever changed to pass its prompt to ``input()`` this would stop
        reproducing, and the two tests above would pass for the wrong reason.
        """
        data = _drive_pty(use_rich_prompt_class=True)
        bad = _replay_erases(data, prompt_visible_width())
        self.assertNotEqual(
            bad, [], "Rich no longer reproduces the bug; revisit this test"
        )


if __name__ == "__main__":
    unittest.main()