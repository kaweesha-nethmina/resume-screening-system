"""Tests for Member 1 — JD Analyst Agent."""

import pytest

from agents.jd_analyst import jd_analyst_node
from tools.jd_tools import read_jd_file

SAMPLE_JD = (
    "Senior Python Developer needed. Required: Python, SQL, Docker. "
    "Preferred: Kubernetes, AWS. 3+ years experience. Bachelor's degree required."
)

# --- Tool tests ---


def test_read_jd_file_success(tmp_path):
    """Test that read_jd_file returns content from a valid .txt file."""
    file = tmp_path / "sample.txt"
    file.write_text("Hello World")
    result = read_jd_file.invoke({"file_path": str(file)})
    assert result == "Hello World"


def test_read_jd_file_not_found():
    """Test that read_jd_file raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        read_jd_file.invoke({"file_path": "/nonexistent/path/to/jd.txt"})


def test_read_jd_file_empty(tmp_path):
    """Test that read_jd_file raises ValueError for an empty file."""
    file = tmp_path / "empty.txt"
    file.write_text("   ")
    with pytest.raises(ValueError, match="empty"):
        read_jd_file.invoke({"file_path": str(file)})


# --- Agent tests (require Ollama running with llama3:8b) ---


def test_jd_analyst_returns_required_keys():
    """Test that the JD analyst node returns all 7 required keys."""
    state = {"job_description": SAMPLE_JD, "agent_logs": []}
    result = jd_analyst_node(state)
    req = result["job_requirements"]
    expected_keys = {
        "job_title",
        "required_skills",
        "nice_to_have",
        "min_experience_years",
        "education_level",
        "responsibilities",
        "domain",
    }
    assert expected_keys.issubset(set(req.keys()))


def test_jd_analyst_required_skills_is_list():
    """Test that required_skills is a list."""
    state = {"job_description": SAMPLE_JD, "agent_logs": []}
    result = jd_analyst_node(state)
    assert isinstance(result["job_requirements"]["required_skills"], list)


def test_jd_analyst_no_hallucination():
    """Test that the agent does not hallucinate skills not in the JD."""
    minimal_jd = "Looking for a React developer. Must know React and JavaScript."
    state = {"job_description": minimal_jd, "agent_logs": []}
    result = jd_analyst_node(state)
    skills = [s.lower() for s in result["job_requirements"]["required_skills"]]
    assert "python" not in skills
    assert "java" not in skills
