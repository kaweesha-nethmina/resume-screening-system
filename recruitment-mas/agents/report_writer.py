"""
agents/report_writer.py
Member 4 — Report Writer Agent.

Reads scores, candidate_profiles, and job_requirements from state.
Generates a structured Markdown recruitment report with:
  - Executive summary
  - Ranked candidate shortlist table
  - Top-3 candidate deep-dive analysis
  - 5 tailored interview questions per top candidate
  - Final hiring recommendation

Saves the report to disk using write_report_file() and flushes all
observability logs to a JSONL trace file.
"""
import logging
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from state.schema import RecruitmentState
from tools.report_tools import write_report_file
from utils.logger import log_event, flush_logs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — strict template enforced at the prompt level
# ---------------------------------------------------------------------------
REPORT_SYSTEM = """You are a senior technical recruiter writing a report for a hiring manager.

STRICT RULES:
1. Write ONLY valid Markdown. Use proper headings (#, ##, ###).
2. Include ALL sections listed in the template below — do not skip any section.
3. Be professional, objective, and concise.
4. Do NOT invent information — use only the data provided in the user message.
5. Interview questions must be specific, technical, and directly relevant to the role.
6. The ranked table must include every candidate — not just the top 3.
7. Do NOT add preamble or commentary outside the template structure.

TEMPLATE:
# Recruitment Report: {job_title}
_Generated: {date}_

## Executive Summary
(2–3 sentences summarising the overall candidate pool quality and top recommendation)

## Ranked Shortlist
| Rank | Name | Score | Key Strengths |
|------|------|-------|---------------|
(one row per candidate, ordered by rank)

## Top 3 Candidate Analysis

### 1. [Candidate Name] — Score: X/100
**Strengths:** ...
**Gaps:** ...
**Overall Assessment:** ...

### 2. [Candidate Name] — Score: X/100
**Strengths:** ...
**Gaps:** ...
**Overall Assessment:** ...

### 3. [Candidate Name] — Score: X/100
**Strengths:** ...
**Gaps:** ...
**Overall Assessment:** ...

## Interview Questions

### For [Candidate 1 Name]:
1. ...
2. ...
3. ...
4. ...
5. ...

### For [Candidate 2 Name]:
1. ...
2. ...
3. ...
4. ...
5. ...

### For [Candidate 3 Name]:
1. ...
2. ...
3. ...
4. ...
5. ...

## Recommendation
(A clear paragraph recommending who to hire and why, with any caveats or conditions)
"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def _build_context(state: RecruitmentState) -> str:
    """
    Assemble a compact, structured context string from state data.

    Pulls job requirements, ranked scores, and candidate profiles into a
    single string that fits comfortably within the LLM's context window.

    Args:
        state: Current RecruitmentState.

    Returns:
        str: A multi-line context string for the LLM prompt.
    """
    job = state["job_requirements"]
    scores = state["scores"]
    profiles = {p["name"]: p for p in state["candidate_profiles"]}

    lines = [
        f"JOB TITLE     : {job.get('job_title', 'Unknown Role')}",
        f"DOMAIN        : {job.get('domain', 'Not specified')}",
        f"REQUIRED SKILLS: {', '.join(job.get('required_skills', []))}",
        f"PREFERRED SKILLS: {', '.join(job.get('preferred_skills', []))}",
        f"MIN EXPERIENCE: {job.get('min_experience_years', 0)} years",
        "",
        "--- RANKED CANDIDATES ---",
    ]

    for rank, score in enumerate(scores, start=1):
        profile = profiles.get(score["name"], {})
        skills_preview = ", ".join(profile.get("skills", [])[:6])
        lines.append(
            f"#{rank}  {score['name']}"
            f" | Score: {score['total_score']}/100"
            f" | Skills: {skills_preview}"
            f" | Exp: {profile.get('total_experience_years', '?')} yrs"
            f" | Education: {profile.get('education', '?')}"
            f" | Justification: {score.get('justification', '')}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
def report_writer_node(state: RecruitmentState) -> dict:
    """
    LangGraph node: Report Writer Agent.

    Reads scores, candidate_profiles, and job_requirements from state.
    Invokes the LLM to generate a full Markdown recruitment report, saves
    it to disk, and flushes the observability log trace.

    Args:
        state: RecruitmentState — reads scores, candidate_profiles,
               job_requirements, agent_logs.

    Returns:
        dict: Partial state update containing:
            - final_report (str)  : Full Markdown report content.
            - report_path (str)   : Absolute path of the saved report file.
    """
    top_candidate = state["scores"][0]["name"] if state.get("scores") else "none"
    log_event(state, "ReportWriter", "start", {"top_candidate": top_candidate})

    # Build prompt components
    job_title = state["job_requirements"].get("job_title", "Open Role")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    context = _build_context(state)

    system_prompt = (
        REPORT_SYSTEM
        .replace("{job_title}", job_title)
        .replace("{date}", date_str)
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                "Write the complete recruitment report using ONLY the data below.\n"
                "Follow the template exactly — include every section.\n\n"
                f"{context}"
            )
        ),
    ]

    # Invoke LLM
    log_event(state, "ReportWriter", "tool_call", {"tool": "ChatOllama", "model": "llama3:8b"})
    llm = ChatOllama(model="llama3:8b", temperature=0.3)
    response = llm.invoke(messages)
    report_md = response.content.strip()

    # Save report to disk
    output_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    log_event(state, "ReportWriter", "tool_call", {
        "tool": "write_report_file",
        "path": output_path,
    })
    saved_path = write_report_file.invoke({
        "content": report_md,
        "output_path": output_path,
    })

    # Flush all observability logs to JSONL
    log_event(state, "ReportWriter", "complete", {
        "path": saved_path,
        "chars": len(report_md),
    })
    flush_logs(state.get("agent_logs", []))

    logger.info(f"[ReportWriter] Report saved to: {saved_path}")
    return {"final_report": report_md, "report_path": saved_path}
