"""
utils/terminal.py
Terminal UI helpers — ANSI colors, banners, progress indicators.
Zero external dependencies (pure Python).
"""
import sys
import time
import threading

# ── ANSI Color Codes ────────────────────────────────────────────
class C:
    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    DIM         = "\033[2m"

    # Foreground colors
    CYAN        = "\033[96m"
    BLUE        = "\033[94m"
    GREEN       = "\033[92m"
    YELLOW      = "\033[93m"
    RED         = "\033[91m"
    MAGENTA     = "\033[95m"
    WHITE       = "\033[97m"
    GREY        = "\033[90m"

    # Background colors
    BG_CYAN     = "\033[46m"
    BG_BLUE     = "\033[44m"
    BG_GREEN    = "\033[42m"
    BG_DARK     = "\033[40m"

def c(text: str, *codes) -> str:
    """Wrap text with ANSI color codes and auto-reset."""
    return "".join(codes) + str(text) + C.RESET

def divider(char: str = "─", width: int = 56, color: str = C.CYAN) -> str:
    return c(char * width, color)

def header_block(title: str, subtitle: str = "") -> None:
    """Print a styled header banner."""
    print()
    print(c("╔" + "═" * 54 + "╗", C.CYAN, C.BOLD))
    pad = (54 - len(title)) // 2
    print(c("║", C.CYAN, C.BOLD) + " " * pad + c(title, C.WHITE, C.BOLD) + " " * (54 - pad - len(title)) + c("║", C.CYAN, C.BOLD))
    if subtitle:
        pad2 = (54 - len(subtitle)) // 2
        print(c("║", C.CYAN, C.BOLD) + " " * pad2 + c(subtitle, C.GREY) + " " * (54 - pad2 - len(subtitle)) + c("║", C.CYAN, C.BOLD))
    print(c("╚" + "═" * 54 + "╝", C.CYAN, C.BOLD))
    print()

def section(title: str, color: str = C.CYAN) -> None:
    """Print a section label."""
    print()
    print(c(f"  ┌─ {title} ", color, C.BOLD))
    print(c("  │", color))

def row(label: str, value: str, label_color: str = C.GREY, value_color: str = C.WHITE) -> None:
    """Print a key-value row inside a section."""
    print(c("  │  ", C.GREY) + c(f"{label:<22}", label_color) + c(str(value), value_color))

def success(msg: str) -> None:
    print(c("  ✓  ", C.GREEN) + c(msg, C.WHITE))

def warn(msg: str) -> None:
    print(c("  !  ", C.YELLOW) + c(msg, C.YELLOW))

def error(msg: str) -> None:
    print(c("  x  ", C.RED) + c(msg, C.RED))

def info(msg: str) -> None:
    print(c("  >  ", C.CYAN) + c(msg, C.GREY))

def tag(text: str, color: str = C.CYAN) -> str:
    """Render a small inline tag like  [ Python ] """
    return c("[", C.GREY) + c(text, color, C.BOLD) + c("]", C.GREY)

def pill_list(items: list, color: str = C.CYAN, per_row: int = 4) -> None:
    """Print a list of items as colored pills, wrapping at per_row."""
    if not items:
        print(c("  │     ", C.GREY) + c("none", C.DIM))
        return
    chunks = [items[i:i+per_row] for i in range(0, len(items), per_row)]
    for chunk in chunks:
        line = "  │     "
        for item in chunk:
            line += tag(item, color) + "  "
        print(line)

def end_section(color: str = C.CYAN) -> None:
    print(c("  └" + "─" * 52, color))


# ── Spinner ──────────────────────────────────────────────────────
class Spinner:
    """
    Animated terminal spinner for long-running operations.

    Usage:
        with Spinner("Running agent..."):
            time.sleep(3)
    """
    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, message: str = "Working", color: str = C.CYAN):
        self.message = message
        self.color   = color
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {self.color}{frame}{C.RESET}  {C.GREY}{self.message}...{C.RESET}   ")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
