from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

from agent.palette import PALETTE

# Replace default error tracebacks with better version
install()

# Style tags are named for their role in the output, not for a colour: the
# same names are resolved by every frontend (see agent/palette.py).
theme_dict: dict[str, str] = PALETTE.rich_theme_dict()

# Theme
monkee_theme = Theme(theme_dict)

# Create consoles
console = Console(theme=monkee_theme)
err_console = Console(theme=monkee_theme, stderr=True)

def newline():
    console.print()

def print(msg,
            end='\n',
            justify=None):
    console.print(msg,
                  end=end,
                  justify=justify)

def err(msg,
            end='\n',
            justify=None):
    err_console.print(f"[err]⨯[/err] {msg}",
                      end=end,
                      justify=justify)

def ok(msg,
            end='\n',
            justify=None):
    console.print(f"[ok]✓[/ok] {msg}",
                  end=end,
                  justify=justify)

def info(msg,
            end='\n',
            justify=None):
    console.print(f"[info]⇨[/info] {msg}",
                  end=end,
                  justify=justify)

def warn(msg,
            end='\n',
            justify=None):
    err_console.print(f"[warn]⚠[/warn] {msg}",
                      end=end,
                      justify=justify)
