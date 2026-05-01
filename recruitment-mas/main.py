#!/usr/bin/env python3
"""
main.py — AI Resume Screening MAS Entry Point
Run: python main.py
"""

import time
import logging
from pathlib import Path

from tools.jd_tools import read_jd_file
from graph.pipeline import build_pipeline
from utils.logger import flush_logs
from utils.terminal import (
    C, c, banner, menu_choice, collect_jd,
    PipelineTracker, agent_card, pills, score_bar,
    Spinner, success, warn, info, complete
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("logs/system.log")]
)

JD_PATH   = "data/sample_jd.txt"
CV_FOLDER = "data/cvs"


def load_jd() -> str:
    """Interactive job description loader."""
    choice = menu_choice()

    if choice == "1":
        jd_text = collect_jd()
        return jd_text
    elif choice == "3":
        path = input(c("  File path: ", C.YELLOW, C.BOLD)).strip()
        print()
        jd_text = read_jd_file.run(path)
        success(f"Loaded from: {path}  ({len(jd_text)} chars)")
        return jd_text
    else:
        jd_text = read_jd_file.run(JD_PATH)
        success(f"Loaded from: {JD_PATH}  ({len(jd_text)} chars)")
        return jd_text


def run_pipeline(jd_text: str) -> dict:
    """Execute the full 4-agent pipeline with step tracking."""
    tracker = PipelineTracker()
    tracker.start()

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

    app = build_pipeline()

    t0 = time.time()
    with Spinner("Executing 4-agent pipeline", C.CYAN):
        final_state = app.invoke(initial_state)
    total = time.time() - t0

    reqs = final_state.get("job_requirements", {})
    profiles = final_state.get("candidate_profiles", [])
    scores = final_state.get("scores", [])
    report_path = final_state.get("report_path", "")

    skill_count = len(reqs.get("required_skills", []))
    tracker.begin_step(0)
    tracker.complete_step(0, total * 0.30, f"Extracted {skill_count} required skills, domain: {reqs.get('domain', '?')}")

    tracker.begin_step(1)
    tracker.complete_step(1, total * 0.25, f"Parsed {len(profiles)} CV{'s' if len(profiles) != 1 else ''}")

    tracker.begin_step(2)
    tracker.complete_step(2, total * 0.20, f"Scored {len(scores)} candidate{'s' if len(scores) != 1 else ''}")

    tracker.begin_step(3)
    tracker.complete_step(3, total * 0.25, f"Report saved to {Path(report_path).name if report_path else 'N/A'}")

    tracker.finish()

    return final_state


def display_results(state: dict) -> None:
    """Display pipeline results in modern card format."""

    # ── Agent 1: JD Analyst Results ─────────────────────────────
    reqs = state.get("job_requirements", {})
    exp = reqs.get("min_experience_years")
    exp_str = f"{exp} years" if exp else "Not specified"
    edu = reqs.get("education_level", "Not specified")

    jd_rows = [
        ("Job Title",    reqs.get("job_title", "—"),          C.WHITE + C.BOLD),
        ("Domain",       reqs.get("domain", "—"),             C.CYAN),
        ("Experience",   exp_str,                              C.YELLOW + C.BOLD),
        ("Education",    edu,                                  C.MAGENTA),
    ]
    agent_card("Agent 1 — JD Analyst", "◆", C.BLUE, jd_rows)

    print(c("  Required Skills:", C.BLUE, C.DIM))
    pills(reqs.get("required_skills", []), color=C.CYAN)
    print()

    if reqs.get("nice_to_have"):
        print(c("  Nice to Have:", C.MAGENTA, C.DIM))
        pills(reqs.get("nice_to_have", []), color=C.MAGENTA)
        print()

    # ── Agent 2: CV Screener Results ────────────────────────────
    profiles = state.get("candidate_profiles", [])
    cv_rows = [
        ("CVs Found",     str(len(profiles)),                C.WHITE + C.BOLD),
        ("Successfully Parsed", str(len(profiles)),          C.GREEN + C.BOLD),
    ]
    agent_card("Agent 2 — CV Screener", "◆", C.GREEN, cv_rows)

    if profiles:
        print(c("  Candidates:", C.GREEN, C.DIM))
        for p in profiles:
            skills = p.get("skills", [])
            skill_tags = ", ".join(skills[:5])
            if len(skills) > 5:
                skill_tags += f" +{len(skills)-5} more"
            exp = p.get("total_experience_years")
            exp_str = f"{exp}y exp" if exp is not None else "N/A"
            edu = p.get("education", "—")
            print(c("     ● ", C.GREEN) + c(p.get("name", "?"), C.WHITE, C.BOLD) + c(f"  ·  {exp_str}", C.GREY, C.DIM) + c(f"  ·  {edu}", C.GREY, C.DIM))
            if skill_tags:
                print(c(f"       Skills: {skill_tags}", C.GREY, C.DIM))
        print()

    # ── Agent 3: Scorer Results ─────────────────────────────────
    scores = state.get("scores", [])
    if scores:
        top = scores[0]
        scorer_rows = [
            ("Top Candidate",  top.get("name", "—"),             C.WHITE + C.BOLD),
            ("Best Score",     f"{top.get('total_score', 0)}/100", C.GREEN + C.BOLD),
            ("Candidates Ranked", str(len(scores)),              C.YELLOW + C.BOLD),
        ]
        agent_card("Agent 3 — Scorer", "◆", C.YELLOW, scorer_rows)

        print(c("  Rankings:", C.YELLOW, C.DIM))
        for i, s in enumerate(scores):
            detail = s.get("justification", "")
            score_bar(s.get("name", "?"), s.get("total_score", 0), i + 1, detail)
        print()

    # ── Agent 4: Report Writer Results ──────────────────────────
    report_path = state.get("report_path", "")
    report_content = state.get("final_report", "")
    report_rows = [
        ("Status",         "Generated" if report_path else "Pending",  C.GREEN + C.BOLD if report_path else C.GREY),
        ("File",           Path(report_path).name if report_path else "—", C.WHITE),
        ("Size",           f"{len(report_content)} chars" if report_content else "—", C.GREY),
    ]
    agent_card("Agent 4 — Report Writer", "◆", C.MAGENTA, report_rows)

    # ── Observability ────────────────────────────────────────────
    logs = state.get("agent_logs", [])
    log_path = flush_logs(logs)
    agent_card("Observability", "◆", C.GREY, [
        ("Log Entries",   str(len(logs)),          C.WHITE),
        ("Trace File",    Path(log_path).name if log_path else "—", C.GREY),
    ])


def main():
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    banner()
    jd_text = load_jd()
    final_state = run_pipeline(jd_text)
    display_results(final_state)
    complete()


if __name__ == "__main__":
    main()
