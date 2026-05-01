"""Member 4 — Report Writer Agent

Generates a comprehensive screening report summarizing all candidates,
their scores, rankings, and recommended interview questions.
"""

import logging
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from state.schema import RecruitmentState
from tools.report_tools import write_report_file
from utils.logger import log_event, flush_logs

REPORT_SYSTEM = """You are a senior technical recruiter writing for a hiring manager.

STRICT RULES:
1. Write ONLY valid Markdown. Use proper headings (#, ##, ###).
2. Include ALL sections listed in the template — do not skip any.
3. Be professional, objective, and concise.
4. Do NOT invent information — use only data provided.
5. Interview questions must be technical and role-relevant.

TEMPLATE:
# Recruitment Report: {job_title}
_Generated: {date}_

## Executive Summary
(2-3 sentences summarising the candidate pool quality)

## Ranked Shortlist
| Rank | Name | Score | Key Strengths |
|------|------|-------|---------------|
(table rows for all candidates)

## Top 3 Candidate Analysis
### 1. [Name] — Score: X/100
**Strengths:** ...
**Gaps:** ...

### 2. [Name] — Score: X/100
...

### 3. [Name] — Score: X/100
...

## Interview Questions
### For [Candidate 1 Name]:
1. ... 2. ... 3. ... 4. ... 5. ...

### For [Candidate 2 Name]:
...

### For [Candidate 3 Name]:
...

## Recommendation
(Final hiring recommendation paragraph)"""

def _build_context(state: RecruitmentState) -> str:
    job = state["job_requirements"]
    scores = state["scores"]
    profiles = {p["name"]: p for p in state["candidate_profiles"]}
    
    lines = []
    lines.append(f"Job Title: {job.get('job_title', 'N/A')}")
    lines.append(f"Required Skills: {', '.join(job.get('required_skills', []))}")
    lines.append(f"Minimum Experience: {job.get('min_experience_years', 0)} years")
    lines.append(f"Domain: {job.get('domain', 'N/A')}")
    lines.append("---CANDIDATES (ranked)---")
    
    for i, score in enumerate(scores, 1):
        name = score["name"]
        profile = profiles.get(name, {})
        lines.append(
            f"{i}. {name} | Score:{score.get('total_score', 0)}/100 | "
            f"Skills:{', '.join(profile.get('skills', []))} | "
            f"Exp:{profile.get('total_experience_years', 0)}yr | "
            f"Edu:{profile.get('education', 'N/A')} | "
            f"Justification:{score.get('justification', 'N/A')}"
        )
        
    return "\\n".join(lines)

def report_writer_node(state: RecruitmentState) -> dict:
    """Generate and write the final screening report.

    Args:
        state: The current pipeline state containing scores, interview_questions, etc.

    Returns:
        dict: A dict with final_report and report_path to merge into the state.
    """
    top_candidate = state["scores"][0]["name"] if state.get("scores") else "none"
    log_event(state, "ReportWriter", "start", {"top_candidate": top_candidate})
    
    llm = ChatOllama(model="llama3:8b", temperature=0.3)
    
    job_title = state["job_requirements"].get("job_title", "Unknown Job")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    context = _build_context(state)
    system = REPORT_SYSTEM.replace("{job_title}", job_title).replace("{date}", date_str)
    
    response = llm.invoke([
        SystemMessage(content=system), 
        HumanMessage(content=f"Write the full recruitment report using this data:\\n\\n{context}")
    ])
    
    report_md = response.content.strip()
    
    output_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    saved_path = write_report_file.invoke({"content": report_md, "output_path": output_path})
    
    flush_logs(state.get("agent_logs", []))
    logging.info(f"Report Writer saved report to {saved_path}")
    
    log_event(state, "ReportWriter", "complete", {"path": saved_path, "chars": len(report_md)})
    
    return {"final_report": report_md, "report_path": saved_path}
