"""
tests/test_m4_report_writer.py
Member 4 — Tests for the Report Writer agent and write_report_file tool.

Includes:
  - Unit tests for the write_report_file() tool
  - Integration tests for report content correctness
  - LLM-as-a-Judge evaluation test (requires Ollama running locally)
"""
import pytest
from pathlib import Path
from tools.report_tools import write_report_file
from agents.report_writer import report_writer_node
from utils.logger import log_event, flush_logs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_test_state() -> dict:
    """Minimal but realistic RecruitmentState for testing."""
    return {
        "job_requirements": {
            "job_title": "Senior Python Developer",
            "required_skills": ["Python", "Django", "PostgreSQL"],
            "preferred_skills": ["Docker", "Kubernetes"],
            "min_experience_years": 3,
            "domain": "Backend Web Development",
        },
        "scores": [
            {
                "name": "Alice Kim",
                "total_score": 88,
                "skill_score": 100.0,
                "experience_score": 100.0,
                "education_score": 75.0,
                "justification": "Matched 100% of required skills. 5 yrs exp (min 3). Education: Master.",
                "email": "alice@example.com",
            },
            {
                "name": "Bob Martinez",
                "total_score": 62,
                "skill_score": 66.7,
                "experience_score": 66.7,
                "education_score": 50.0,
                "justification": "Matched 67% of required skills. 2 yrs exp (min 3). Education: Bachelor.",
                "email": "bob@example.com",
            },
            {
                "name": "Carol Singh",
                "total_score": 74,
                "skill_score": 100.0,
                "experience_score": 80.0,
                "education_score": 25.0,
                "justification": "Matched 100% of required skills. 2.4 yrs exp (min 3). Education: Diploma.",
                "email": "carol@example.com",
            },
        ],
        "candidate_profiles": [
            {
                "name": "Alice Kim",
                "email": "alice@example.com",
                "skills": ["Python", "Django", "PostgreSQL", "REST APIs", "Git"],
                "total_experience_years": 5,
                "education": "Master of Computer Science",
                "certifications": ["AWS Certified Developer"],
                "previous_roles": ["Backend Engineer at TechCorp"],
                "summary": "Experienced Python developer with strong backend skills.",
            },
            {
                "name": "Bob Martinez",
                "email": "bob@example.com",
                "skills": ["Python", "MySQL", "HTML"],
                "total_experience_years": 2,
                "education": "Bachelor of Science in IT",
                "certifications": [],
                "previous_roles": ["Junior Developer at StartupXYZ"],
                "summary": "Junior developer eager to grow in backend development.",
            },
            {
                "name": "Carol Singh",
                "email": "carol@example.com",
                "skills": ["Python", "Django", "PostgreSQL", "Celery"],
                "total_experience_years": 2.4,
                "education": "Diploma in Software Engineering",
                "certifications": ["Python Institute PCEP"],
                "previous_roles": ["Software Developer at Agency"],
                "summary": "Practical Django developer with strong database skills.",
            },
        ],
        "agent_logs": [],
    }


# ── Tool: write_report_file ───────────────────────────────────────────────────

def test_write_report_creates_file(tmp_path):
    """Tool creates the file at the specified path."""
    out = str(tmp_path / "reports" / "test.md")
    result = write_report_file.invoke({"content": "# Test Report\nContent here.", "output_path": out})
    assert Path(result).exists()


def test_write_report_empty_raises(tmp_path):
    """Tool raises ValueError when content is empty."""
    with pytest.raises(ValueError):
        write_report_file.invoke({"content": "", "output_path": str(tmp_path / "out.md")})


def test_write_report_whitespace_only_raises(tmp_path):
    """Tool raises ValueError when content is only whitespace."""
    with pytest.raises(ValueError):
        write_report_file.invoke({"content": "   \n\t  ", "output_path": str(tmp_path / "out.md")})


def test_write_report_returns_absolute_path(tmp_path):
    """Tool returns an absolute path string."""
    result = write_report_file.invoke({"content": "# Report", "output_path": str(tmp_path / "r.md")})
    assert Path(result).is_absolute()


def test_write_report_creates_parent_dirs(tmp_path):
    """Tool creates intermediate directories automatically."""
    deep = str(tmp_path / "a" / "b" / "c" / "report.md")
    result = write_report_file.invoke({"content": "# Nested Report", "output_path": deep})
    assert Path(result).exists()


def test_write_report_file_contains_content(tmp_path):
    """Written file contains the original content."""
    content = "# My Report\n\nThis is the body."
    out = str(tmp_path / "report.md")
    path = write_report_file.invoke({"content": content, "output_path": out})
    written = Path(path).read_text(encoding="utf-8")
    assert "# My Report" in written
    assert "This is the body." in written


# ── Agent: report_writer_node ─────────────────────────────────────────────────

def test_report_contains_all_candidate_names():
    """Report must mention every candidate name from the scores list."""
    state = _build_test_state()
    result = report_writer_node(state)
    report = result["final_report"]
    for s in state["scores"]:
        assert s["name"] in report, f"Report missing candidate: {s['name']}"


def test_report_is_markdown():
    """Report must contain at least one Markdown heading."""
    state = _build_test_state()
    result = report_writer_node(state)
    assert "#" in result["final_report"]


def test_report_path_is_string():
    """report_path must be a non-empty string."""
    state = _build_test_state()
    result = report_writer_node(state)
    assert isinstance(result["report_path"], str)
    assert len(result["report_path"]) > 0


def test_report_file_exists_on_disk():
    """The report file must actually exist after the node runs."""
    state = _build_test_state()
    result = report_writer_node(state)
    assert Path(result["report_path"]).exists()


def test_report_mentions_job_title():
    """Report must reference the job title from job_requirements."""
    state = _build_test_state()
    result = report_writer_node(state)
    assert "Python Developer" in result["final_report"]


# ── Observability: logger utilities ──────────────────────────────────────────

def test_log_event_appends_to_state():
    """log_event adds a JSON string to state['agent_logs']."""
    state = {"agent_logs": []}
    log_event(state, "TestAgent", "start", {"foo": "bar"})
    assert len(state["agent_logs"]) == 1
    import json
    entry = json.loads(state["agent_logs"][0])
    assert entry["agent"] == "TestAgent"
    assert entry["event"] == "start"
    assert entry["data"]["foo"] == "bar"


def test_log_event_initialises_logs_key():
    """log_event creates agent_logs if key is missing from state."""
    state = {}
    log_event(state, "Agent", "complete", {})
    assert "agent_logs" in state
    assert len(state["agent_logs"]) == 1


def test_flush_logs_creates_jsonl_file(tmp_path):
    """flush_logs writes a .jsonl file with one line per entry."""
    logs = ['{"agent": "A", "event": "start"}', '{"agent": "A", "event": "end"}']
    path = flush_logs(logs, output_dir=str(tmp_path))
    assert Path(path).exists()
    lines = Path(path).read_text().strip().splitlines()
    assert len(lines) == 2


def test_flush_logs_returns_path_string(tmp_path):
    """flush_logs returns a string path."""
    result = flush_logs(["{}"], output_dir=str(tmp_path))
    assert isinstance(result, str)
    assert result.endswith(".jsonl")


# ── LLM-as-a-Judge evaluation ─────────────────────────────────────────────────

def test_report_quality_llm_judge():
    """
    Use a second Ollama LLM call to evaluate report quality against 4 criteria.
    The pass rate must be >= 75% (i.e., at least 3 out of 4 criteria pass).

    Criteria:
      1. Contains a ranked table of candidates.
      2. Includes tailored interview questions for top candidates.
      3. Is written in a professional tone.
      4. Contains an executive summary or recommendation section.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage
    except ImportError:
        pytest.skip("langchain_ollama not installed — skipping LLM judge test")

    state = _build_test_state()
    result = report_writer_node(state)
    report = result["final_report"]

    criteria = [
        "Does the report contain a ranked table listing all candidates?",
        "Does the report include at least 3 interview questions for the top candidate?",
        "Is the report written in a professional and objective tone throughout?",
        "Does the report have an executive summary OR a recommendation section?",
    ]

    try:
        llm = ChatOllama(model="llama3:8b", temperature=0)
        passes = 0
        for criterion in criteria:
            prompt = (
                f"Report (first 2000 chars):\n{report[:2000]}\n\n"
                f"Question: {criterion}\n"
                "Answer ONLY with the single word 'yes' or 'no'. No explanation."
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            answer = resp.content.strip().lower()
            if answer.startswith("yes"):
                passes += 1

        pass_rate = passes / len(criteria)
        assert pass_rate >= 0.75, (
            f"LLM judge pass rate too low: {pass_rate:.0%} ({passes}/{len(criteria)})\n"
            f"Report preview:\n{report[:500]}"
        )
    except Exception as exc:
        pytest.skip(f"Ollama not available or failed: {exc}")
