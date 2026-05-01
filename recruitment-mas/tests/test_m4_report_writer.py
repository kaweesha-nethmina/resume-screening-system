"""Tests for Member 4 — Report Writer Agent."""

import pytest
import json
from pathlib import Path
from tools.report_tools import write_report_file
from agents.report_writer import report_writer_node
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage


def test_write_report_creates_file(tmp_path):
    output_path = str(tmp_path / "reports" / "test.md")
    result = write_report_file.invoke({"content": "# Test Report\\nContent here.", "output_path": output_path})
    assert Path(result).exists()

def test_write_report_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="Report content is empty. Cannot write an empty report file."):
        write_report_file.invoke({"content": "", "output_path": str(tmp_path / "test.md")})

def test_write_report_returns_absolute_path(tmp_path):
    output_path = str(tmp_path / "test.md")
    result = write_report_file.invoke({"content": "content", "output_path": output_path})
    assert Path(result).is_absolute()

def test_write_report_creates_parent_dirs(tmp_path):
    output_path = str(tmp_path / "a" / "b" / "c" / "r.md")
    result = write_report_file.invoke({"content": "content", "output_path": output_path})
    assert Path(result).exists()

def test_report_contains_candidate_names():
    state = _build_test_state()
    result = report_writer_node(state)
    assert "Alice K" in result["final_report"]
    assert "Bob M" in result["final_report"]

def test_report_is_markdown():
    state = _build_test_state()
    result = report_writer_node(state)
    assert "#" in result["final_report"]

def test_report_path_is_set():
    state = _build_test_state()
    result = report_writer_node(state)
    assert isinstance(result["report_path"], str)
    assert result["report_path"].endswith(".md")
    assert len(result["report_path"]) > 0

def test_report_quality_llm_judge():
    """
    Use a second Ollama call to evaluate report quality.
    The judge checks 4 criteria — pass rate must be >= 75%.
    """
    state = _build_test_state()
    result = report_writer_node(state)
    report = result["final_report"]

    criteria = [
        "Does the report contain a ranked table of candidates?",
        "Does the report include interview questions for top candidates?",
        "Is the report written in a professional tone?",
        "Does the report have an executive summary or recommendation section?",
    ]
    llm = ChatOllama(model="llama3:8b", temperature=0)
    passes = 0
    for criterion in criteria:
        msg = f"Report:\\n{report[:2000]}\\n\\nQuestion: {criterion}\\nAnswer ONLY yes or no."
        resp = llm.invoke([HumanMessage(content=msg)])
        if "yes" in resp.content.lower():
            passes += 1

    pass_rate = passes / len(criteria)
    assert pass_rate >= 0.75, f"LLM judge pass rate too low: {pass_rate:.0%} ({passes}/{len(criteria)})"

def _build_test_state() -> dict:
    return {
        "job_requirements": {"job_title": "Python Dev", "required_skills": ["Python"],
                             "min_experience_years": 3, "domain": "Software"},
        "scores": [
            {"name": "Alice K", "total_score": 85, "justification": "Strong Python skills.", "email": "a@x.com"},
            {"name": "Bob M",   "total_score": 60, "justification": "Limited experience.",   "email": "b@x.com"},
        ],
        "candidate_profiles": [
            {"name": "Alice K", "skills": ["Python","SQL"], "total_experience_years": 5, "education": "Master"},
            {"name": "Bob M",   "skills": ["Java"],         "total_experience_years": 2, "education": "Bachelor"},
        ],
        "agent_logs": []
    }
