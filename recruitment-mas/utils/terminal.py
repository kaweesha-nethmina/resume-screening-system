"""
utils/terminal.py — Modern Terminal UI for AI Resume Screening MAS
Zero external dependencies (pure Python ANSI + Unicode).
"""

import sys
import time
import re
import threading
import shutil

# ── Terminal Dimensions ─────────────────────────────────────────
WIDTH = 72

def term_width():
    return min(shutil.get_terminal_size().columns, 80)

# ── ANSI Color Codes ────────────────────────────────────────────
class C:
    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    DIM         = "\033[2m"
    ITALIC      = "\033[3m"
    UNDERLINE   = "\033[4m"

    CYAN        = "\033[96m"
    BLUE        = "\033[94m"
    GREEN       = "\033[92m"
    YELLOW      = "\033[93m"
    RED         = "\033[91m"
    MAGENTA     = "\033[95m"
    WHITE       = "\033[97m"
    GREY        = "\033[90m"
    ORANGE      = "\033[38;5;208m"

    BG_CYAN     = "\033[46m"
    BG_BLUE     = "\033[44m"
    BG_GREEN    = "\033[42m"
    BG_DARK     = "\033[40m"
    BG_DARKER   = "\033[48;5;235m"
    BG_CARD_GREY = "\033[48;5;235m"

def c(text: str, *codes) -> str:
    return "".join(codes) + str(text) + C.RESET

def _visual_len(text: str) -> int:
    return len(re.sub(r'\033\[[0-9;]*m', '', text))

# ── Header Banner ───────────────────────────────────────────────
def banner():
    w = WIDTH
    print()
    print(c("╭" + "─" * (w-2) + "╮", C.CYAN, C.BOLD))
    title = "AI RESUME SCREENING"
    subtitle = "LangGraph · Ollama llama3:8b · 4-Agent Pipeline"
    pad1 = (w - 2 - _visual_len(title)) // 2
    pad2 = (w - 2 - _visual_len(subtitle)) // 2
    print(c("│", C.CYAN, C.BOLD) + " " * pad1 + c(title, C.WHITE, C.BOLD) + " " * (w - 2 - pad1 - _visual_len(title)) + c("│", C.CYAN, C.BOLD))
    print(c("│", C.CYAN, C.BOLD) + " " * pad2 + c(subtitle, C.GREY, C.DIM) + " " * (w - 2 - pad2 - _visual_len(subtitle)) + c("│", C.CYAN, C.BOLD))
    print(c("╰" + "─" * (w-2) + "╯", C.CYAN, C.BOLD))
    print()

# ── Pipeline Progress ───────────────────────────────────────────
class PipelineTracker:
    STEPS = [
        ("JD Analyst",    "Extracting job requirements",     C.BLUE),
        ("CV Screener",   "Parsing candidate resumes",       C.GREEN),
        ("Scorer",        "Computing match scores",          C.YELLOW),
        ("Report Writer", "Generating recruitment report",   C.MAGENTA),
    ]

    def __init__(self):
        self.current = 0
        self.timings = []
        self.start_time = None

    def _print_pipeline_bar(self):
        w = WIDTH
        sep_w = len(self.STEPS) - 1
        seg_w = (w - 4 - sep_w) // len(self.STEPS)

        line = ""
        for i, (_, _, color) in enumerate(self.STEPS):
            if i < self.current:
                line += c("█" * seg_w, color)
            elif i == self.current:
                done = seg_w // 2
                line += c("█" * done, color) + c("░" * (seg_w - done), C.GREY)
            else:
                line += c("░" * seg_w, C.GREY)
            if i < len(self.STEPS) - 1:
                line += c("│", C.GREY, C.DIM)
        print(c("  ", C.GREY) + line)

    def _print_step_labels(self):
        w = WIDTH
        sep_w = len(self.STEPS) - 1
        seg_w = (w - 4 - sep_w) // len(self.STEPS)

        line = "  "
        for i, (name, _, color) in enumerate(self.STEPS):
            label = name if len(name) <= seg_w else name[:seg_w-1]
            padded = c(label, color if i <= self.current else C.GREY, C.BOLD if i == self.current else C.DIM) + " " * (seg_w - len(label))
            line += padded
            if i < len(self.STEPS) - 1:
                line += c("│", C.GREY, C.DIM)
        print(line)

    def start(self):
        self.start_time = time.time()
        print(c("╭─ Pipeline Execution " + "─" * (WIDTH - 22) + "╮", C.GREY))
        print()
        self._print_step_labels()
        self._print_pipeline_bar()
        print()

    def begin_step(self, step_index: int):
        self.current = step_index
        name, desc, color = self.STEPS[step_index]
        print()
        print(c(f"  ▸  STEP {step_index+1}/4: {name}", color, C.BOLD))
        print(c(f"     {desc}", C.GREY, C.DIM))
        print()

    def complete_step(self, step_index: int, duration: float, detail: str = ""):
        self.current = step_index + 1
        self.timings.append(duration)
        name, _, color = self.STEPS[step_index]
        elapsed = f"{duration:.1f}s"
        print(c(f"  ✓  {name} complete", C.GREEN, C.BOLD) + c(f"  ({elapsed})", C.GREY, C.DIM))
        if detail:
            print(c(f"     {detail}", C.GREY, C.DIM))
        print()

        self._print_step_labels()
        self._print_pipeline_bar()

    def finish(self):
        total = time.time() - self.start_time
        print()
        total_str = f"{total:.1f}s"
        avg = total / 4
        print(c("╰" + "─" * (WIDTH-2) + "╯", C.GREY))
        print()
        print(c(f"  Total Pipeline Time: {total_str}", C.WHITE, C.BOLD) + c(f"  (avg {avg:.1f}s per agent)", C.GREY, C.DIM))
        print()

# ── Agent Result Cards ──────────────────────────────────────────
def agent_card(title: str, icon: str, color: str, rows: list):
    """Display a result card with key-value rows. Values wrap if too long.
    
    rows = [(label, value, value_color), ...]
    """
    w = WIDTH
    content_w = w - 8
    label_w = 16

    print(c("  ╭" + "─" * content_w + "╮", color, C.DIM))

    title_prefix = f"  {icon}  {title}"
    title_padding = content_w - len(title_prefix)
    print(c("  │", color, C.DIM) + " " + c(title_prefix, color, C.BOLD) + " " * max(0, title_padding) + c("│", color, C.DIM))
    print(c("  ├" + "─" * content_w + "┤", color, C.DIM))

    for label, value, vcolor in rows:
        val_str = str(value)
        val_color = vcolor or C.WHITE
        available = content_w - label_w

        val_visual = _visual_len(c(val_str, val_color))
        if val_visual <= available:
            label_part = c(f"  {label:<{label_w}}", C.GREY, C.DIM)
            print(c("  │", color, C.DIM) + " " + label_part + c(val_str, val_color) + " " * max(0, available - val_visual) + c("│", color, C.DIM))
        else:
            label_part = c(f"  {label:<{label_w}}", C.GREY, C.DIM)
            print(c("  │", color, C.DIM) + " " + label_part + " " * max(0, available) + c("│", color, C.DIM))
            words = val_str.split()
            line = ""
            for word in words:
                test = line + (" " if line else "") + word
                if _visual_len(c(test, val_color)) <= available:
                    line = test
                else:
                    if line:
                        inner = c(" " * label_w + line, val_color) + " " * max(0, available - _visual_len(c(line, val_color)))
                        print(c("  │", color, C.DIM) + " " + inner + c("│", color, C.DIM))
                    line = word
            if line:
                inner = c(" " * label_w + line, val_color) + " " * max(0, available - _visual_len(c(line, val_color)))
                print(c("  │", color, C.DIM) + " " + inner + c("│", color, C.DIM))

    print(c("  ╰" + "─" * content_w + "╯", color, C.DIM))
    print()

# ── Skill Pills ─────────────────────────────────────────────────
def pills(items: list, color: str = C.CYAN, max_per_line: int = 5):
    if not items:
        print(c("     (none)", C.GREY, C.DIM))
        return

    chunks = [items[i:i+max_per_line] for i in range(0, len(items), max_per_line)]
    for chunk in chunks:
        line = "     "
        for item in chunk:
            line += c(f"▸ {item}", color, C.BOLD) + "  "
        print(line.rstrip())

# ── Score Display ───────────────────────────────────────────────
def score_bar(name: str, score: int, rank: int, detail: str = ""):
    bar_len = 20

    if score >= 80:
        color = C.GREEN
    elif score >= 60:
        color = C.YELLOW
    elif score >= 40:
        color = C.ORANGE
    else:
        color = C.RED

    filled = max(0, min(bar_len, int(score / 5)))
    bar = c("█" * filled, color) + c("░" * (bar_len - filled), C.GREY, C.DIM)

    rank_str = f"#{rank}"
    score_str = f"{score}/100"
    name_str = str(name)

    prefix = c(f"  {rank_str:<3}", C.GREY, C.DIM)
    name_part = c(f" {name_str:<25}", C.WHITE, C.BOLD)
    bar_part = bar
    score_part = c(f" {score_str:>6}", color, C.BOLD)

    print(prefix + name_part + bar_part + score_part)

    if detail:
        words = detail.split()
        line = "     "
        for word in words:
            test = line + word + " "
            if len(test) > WIDTH - 6:
                print(c(line, C.GREY, C.DIM).rstrip())
                line = "     " + word + " "
            else:
                line = test
        if line.strip():
            print(c(line, C.GREY, C.DIM).rstrip())

# ── Input Prompts ───────────────────────────────────────────────
def menu_choice() -> str:
    w = WIDTH
    print(c("╭─ Job Description Source " + "─" * (WIDTH - 28) + "╮", C.CYAN))
    print()
    print(c("     1", C.BG_BLUE + C.WHITE + C.BOLD) + c("  Type or paste in terminal", C.WHITE))
    print(c("     2", C.BG_DARKER + C.WHITE + C.BOLD) + c("  Load default file ", C.GREY) + c("(data/sample_jd.txt)", C.GREY, C.DIM))
    print(c("     3", C.BG_DARKER + C.WHITE + C.BOLD) + c("  Load from custom file path", C.GREY))
    print()
    choice = input(c("  Enter choice [1/2/3]: ", C.YELLOW, C.BOLD)).strip()
    print()
    print(c("╰" + "─" * (w-2) + "╯", C.CYAN))
    return choice

def collect_jd() -> str:
    w = WIDTH
    print()
    print(c("╭" + "─" * (w-2) + "╮", C.BLUE))
    print(c("│", C.BLUE) + c("  PASTE JOB DESCRIPTION", C.WHITE, C.BOLD) + " " * (w - 26) + c("│", C.BLUE))
    print(c("│", C.BLUE) + c("  Type below. Enter ", C.GREY, C.DIM) + c("END", C.YELLOW, C.BOLD) + c(" on a new line to finish.", C.GREY, C.DIM) + " " * (w - 53) + c("│", C.BLUE))
    print(c("╰" + "─" * (w-2) + "╯", C.BLUE))
    print()

    lines = []
    while True:
        try:
            line = input(c("  ❯ ", C.CYAN))
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    content = "\n".join(lines).strip()
    if not content:
        raise ValueError("No job description entered.")

    print()
    print(c(f"  ✓  Collected {len(content)} characters", C.GREEN, C.BOLD))
    return content

# ── Status Messages ─────────────────────────────────────────────
def success(msg: str):
    print(c(f"  ✓  {msg}", C.GREEN, C.BOLD))

def warn(msg: str):
    print(c(f"  ⚠  {msg}", C.YELLOW, C.BOLD))

def info(msg: str):
    print(c(f"  ℹ  {msg}", C.CYAN, C.DIM))

# ── Spinner ─────────────────────────────────────────────────────
class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Working", color: str = C.CYAN):
        self.message = message
        self.color = color
        self._stop = threading.Event()
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
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()

# ── Completion Banner ───────────────────────────────────────────
def complete():
    w = WIDTH
    print()
    print(c("╭" + "─" * (w-2) + "╮", C.GREEN, C.BOLD))
    pad = (w - 16) // 2
    print(c("│", C.GREEN, C.BOLD) + " " * pad + c("✓  COMPLETE", C.GREEN, C.BOLD) + " " * (w - 2 - pad - 10) + c("│", C.GREEN, C.BOLD))
    print(c("╰" + "─" * (w-2) + "╯", C.GREEN, C.BOLD))
    print()
