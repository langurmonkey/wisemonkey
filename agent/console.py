from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

# Replace default error tracebacks with better version
install()

# Theme
monkee_theme = Theme({
    "title": "bold deep_sky_blue3",

    # Global
    "accent": "deep_sky_blue3",
    "accent-bold": "bold deep_sky_blue3",
    "output-frame": "gray39",

    # Turns
    "agent": "medium_orchid",
    "user": "gold1",

    # Features
    "tool": "steel_blue3",
    "status": "white on grey15",
    "weak": "grey39",
    "kbd": "light_goldenrod1 bold on grey15", 

    "list-item": "cyan",
    "list-desc": "grey39",

    # Logging
    "ok": "chartreuse4",
    "info": "dim cyan",
    "warn": "orange_red1",
    "error": "bold red",
    "err": "bold red"
})

# Create consoles
console = Console(theme=monkee_theme)
err_console = Console(theme=monkee_theme, stderr=True)

def newline():
    console.print()

def print(msg:str=None, end:str='\n', justify:str=None):
    console.print(msg,
                  end=end,
                  justify=justify)

def err(msg:str, end:str='\n', justify:str=None):
    err_console.print(f"[err]⨯[/err] {msg}",
                      end=end,
                      justify=justify)

def ok(msg:str, end:str='\n', justify:str=None):
    console.print(f"[ok]✓[/ok] {msg}",
                  end=end,
                  justify=justify)

def info(msg:str, end:str='\n', justify:str=None):
    console.print(f"[info]⇨[/info] {msg}",
                  end=end,
                  justify=justify)

def warn(msg:str, end:str='\n', justify:str=None):
    err_console.print(f"[warn]⚠[/warn] {msg}",
                      end=end,
                      justify=justify)
