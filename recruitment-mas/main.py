#!/usr/bin/env python3
"""
main.py — AI Resume Screening MAS Entry Point
Run: python main.py
</s
"""
import logging
from pathlib import Path
from tools.jd_tools import read_jd_file, get_jd_from_user
from graph.pipeline import build_pipeline
from utils.logger import flush_logs
from utils.terminal import (
    C, c, header_block, section, row, end_section,
    success, warn, info, pill_list, divider, Spinner
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("logs/system.log")]
)

JD_PATH   = "data/sample_jd.txt"
CV_FOLDER = "data/cvs"


def select_jd_input_mode() -> str:
    """Interactive JD source selector."""
    section("Job Description Source", C.CYAN)
    print(c("  │", C.CYAN))
    print(c("  │  ", C.CYAN) + c("  1  ", C.BG_BLUE + C.WHITE + C.BOLD) + c("  Type / paste directly in terminal", C.WHITE))
    print(c("  │  ", C.CYAN) + c("  2  ", C.BG_DARK + C.GREY)           + c("  Load default file ", C.GREY) + c(f"({JD_PATH})", C.DIM + C.GREY))
    print(c("  │  ", C.CYAN) + c("  3  ", C.BG_DARK + C.GREY)           + c("  Load from custom file path", C.GREY))
    print(c("  │", C.CYAN))

    choice = input(c("  │  ", C.CYAN) + c("> Enter choice (1/2/3): ", C.YELLOW, C.BOLD)).strip()

    print(c("  │", C.CYAN))

    if choice == "1":
        end_section(C.CYAN)
        jd_text = _collect_jd_terminal()
        return jd_text
    elif choice == "3":
        path = input(c("  │  ", C.CYAN) + c("> File path: ", C.YELLOW)).strip()
        end_section(C.CYAN)
        jd_text = read_jd_file.run(path)
        success(f"Loaded from: {path}  ({len(jd_text)} chars)")
        return jd_text
    else:
        end_section(C.CYAN)
        jd_text = read_jd_file.run(JD_PATH)
        success(f"Loaded from: {JD_PATH}  ({len(jd_text)} chars)")
        return jd_text


def _collect_jd_terminal() -> str:
    """Styled multi-line terminal JD collector."""
    print()
    print(c("  " + "╔" + "═" * 52 + "╗", C.BLUE))
    print(c("  ║", C.BLUE) + c("   PASTE JOB DESCRIPTION", C.WHITE, C.BOLD) + " " * 29 + c("║", C.BLUE))
    print(c("  ║", C.BLUE) + c("   Type or paste below. Type ", C.GREY) + c("END", C.YELLOW, C.BOLD) + c(" alone to finish.", C.GREY) + " " * 6 + c("║", C.BLUE))
    print(c("  ╚" + "═" * 52 + "╝", C.BLUE))
    print()

    lines = []
    while True:
        try:
            line = input(c("  > ", C.CYAN))
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    content = "\n".join(lines).strip()

    if not content:
        raise ValueError("No job description entered.")

    print()
    success(f"JD collected - {len(content)} characters")
    return content


def _print_results(final_state: dict) -> None:
    """Pretty-print the full pipeline results."""

    reqs = final_state.get("job_requirements", {})
    section("Job Requirements Extracted", C.CYAN)
    row("Job Title",   reqs.get("job_title", "-"),        C.GREY, C.WHITE + C.BOLD)
    row("Domain",      reqs.get("domain", "-"),            C.GREY, C.MAGENTA)
    row("Experience",  f"{reqs.get('min_experience_years', '-')} years minimum", C.GREY, C.YELLOW)
    row("Education",   reqs.get("education_level", "-"),   C.GREY, C.CYAN)
    print(c("  │", C.CYAN))
    print(c("  │  ", C.CYAN) + c("Required Skills:", C.GREY))
    pill_list(reqs.get("required_skills", []), color=C.CYAN)
    print(c("  │", C.CYAN))
    print(c("  │  ", C.CYAN) + c("Nice to Have:", C.GREY))
    pill_list(reqs.get("nice_to_have", []), color=C.MAGENTA)
    end_section(C.CYAN)

    profiles   = final_state.get("candidate_profiles", [])
    scores     = final_state.get("scores", [])
    section("Candidates", C.GREEN)
    row("Processed",  str(len(profiles)),  C.GREY, C.WHITE)
    row("Scored",     str(len(scores)),    C.GREY, C.WHITE)
    if scores:
        print(c("  │", C.GREEN))
        for i, s in enumerate(scores[:3], 1):
            print(c("  │  ", C.GREEN) + f"#{i}  " + c(s.get('name', '?'), C.WHITE, C.BOLD) + "  " + c(f"{s.get('total_score', '?')}/100", C.YELLOW, C.BOLD))
    end_section(C.GREEN)

    section("Report", C.MAGENTA)
    report_path = final_state.get("report_path", "")
    if report_path:
        row("Saved to", report_path, C.GREY, C.GREEN)
    else:
        row("Status", "Not generated yet (Member 4 pending)", C.GREY, C.YELLOW)
    end_section(C.MAGENTA)

    logs = final_state.get("agent_logs", [])
    section("Observability", C.GREY)
    row("Log entries", str(len(logs)), C.GREY, C.WHITE)


def main():
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    header_block(
        "AI RESUME SCREENING MAS",
        "LangGraph . Ollama llama3:8b . 4 Agents"
    )

    jd_text = select_jd_input_mode()

    print()
    info("Building LangGraph pipeline...")
    app = build_pipeline()
    success("Pipeline ready - 4 agents loaded")
    print()

    initial_state = {
        "job_description":    jd_text,
        "cv_folder_path":     CV_FOLDER,
        "job_requirements":   {},
        "cv_file_paths":      [],
        "cv_raw_texts":       [],
        "candidate_profiles": [],
        "scores":             [],
        "interview_questions":[],
        "final_report":       "",
        "report_path":        "",
        "agent_logs":         [],
    }

    print(divider())
    with Spinner("Running agent pipeline", C.CYAN):
        final_state = app.invoke(initial_state)
    print(divider())

    _print_results(final_state)

    log_path = flush_logs(final_state["agent_logs"])
    row("Trace file",  log_path, C.GREY, C.GREY)
    end_section(C.GREY)

    print()
    print(c("  " + "╔" + "═" * 54 + "╗", C.GREEN, C.BOLD))
    print(c("  ║", C.GREEN, C.BOLD) + c("   Pipeline complete!", C.WHITE, C.BOLD) + " " * 34 + c("║", C.GREEN, C.BOLD))
    print(c("  ╚" + "═" * 54 + "╝", C.GREEN, C.BOLD))
    print()


if __name__ == "__main__":
    main()
