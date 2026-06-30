from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

# Replace default error tracebacks with better version
install()

theme_dict: dict[str, str] = {
    "title": "bold deep_sky_blue3",

    # Global
    "accent": "deep_sky_blue3",
    "accent-bold": "bold deep_sky_blue3",
    "output-frame": "gray39",
    "time": "grey30 i",

    # Turns
    "agent": "orange3",
    "user": "deep_sky_blue3",

    # Features
    "tool": "steel_blue3",
    "status": "white on grey15",
    "path": "#999999 on #252525",
    "cmd": "indian_red",
    "code": "light_pink3",
    "prompt": "dark_olive_green3 bold",
    "weak": "grey39",
    "kbd": "grey69 bold on grey15", 
    "link": "deep_sky_blue1 underline",

    # Patching
    "patch-add": "green",
    "patch-remove": "red",

    "list-item": "cyan",
    "list-desc": "grey39",

    # Logging
    "ok": "chartreuse4",
    "info": "dim cyan",
    "warn": "orange_red1",
    "error": "bold red",
    "err": "bold red"
}

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
