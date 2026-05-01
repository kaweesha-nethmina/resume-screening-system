"""Tests for Member 1 — JD Analyst Agent."""

import pytest

from agents.jd_analyst import FALLBACK_REQUIREMENTS, jd_analyst_node
from tools.jd_tools import read_jd_file

from hypothesis import given, settings, assume
from hypothesis import strategies as st

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


# --- get_jd_from_user tool tests ---


def test_get_jd_from_user_valid_input(monkeypatch):
    """Tool returns content when user types valid text followed by END."""
    responses = iter(["We need a Python developer.", "Required: Python, SQL.", "END"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    from tools.jd_tools import get_jd_from_user
    result = get_jd_from_user.run("")
    assert "Python" in result
    assert len(result) > 0


def test_get_jd_from_user_empty_raises(monkeypatch):
    """Tool raises ValueError when user types END immediately with no content."""
    monkeypatch.setattr("builtins.input", lambda _: "END")
    from tools.jd_tools import get_jd_from_user
    with pytest.raises(ValueError):
        get_jd_from_user.run("")


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


# --- Property-based tests (Hypothesis) ---


@given(st.text(min_size=20, max_size=500))
@settings(max_examples=15)
def test_jd_analyst_never_crashes(random_text):
    """Agent must return a valid dict for ANY text input — the fallback must always activate on garbage."""
    import unittest.mock as mock

    with mock.patch("agents.jd_analyst.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value.content = random_text
        state = {"job_description": "dummy jd text here", "agent_logs": []}
        result = jd_analyst_node(state)
    assert isinstance(result, dict)
    assert "job_requirements" in result
    assert isinstance(result["job_requirements"], dict)
    assert "required_skills" in result["job_requirements"]


@given(st.text(min_size=20, max_size=300))
@settings(max_examples=10)
def test_required_skills_always_list(random_text):
    """required_skills must be a list type regardless of LLM output — fallback enforces this."""
    import unittest.mock as mock

    with mock.patch("agents.jd_analyst.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value.content = random_text
        state = {"job_description": "dummy jd text here", "agent_logs": []}
        result = jd_analyst_node(state)
    assert isinstance(result["job_requirements"]["required_skills"], list)


@given(st.text(min_size=30, max_size=400))
@settings(max_examples=10)
def test_output_always_has_all_keys(random_text):
    """Output schema must be complete — all 7 keys must exist even when LLM fails and fallback fires."""
    import unittest.mock as mock

    with mock.patch("agents.jd_analyst.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value.content = random_text
        state = {"job_description": "dummy jd text here", "agent_logs": []}
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


# --- Mock-based unit tests for improvements ---


def test_type_enforcement_skills_string_becomes_list():
    """If LLM somehow returns required_skills as a plain string, agent must coerce it to a list."""
    import unittest.mock as mock

    broken_output = '{"job_title":"Dev","required_skills":"Python","nice_to_have":[],"min_experience_years":3,"education_level":"Bachelor","responsibilities":[],"domain":"Tech"}'
    with mock.patch("agents.jd_analyst.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value.content = broken_output
        state = {"job_description": "dummy jd text here for testing", "agent_logs": []}
        result = jd_analyst_node(state)
    assert isinstance(result["job_requirements"]["required_skills"], list)


def test_fallback_on_invalid_json():
    """When LLM returns unparseable output, fallback must return all 7 keys."""
    import unittest.mock as mock

    with mock.patch("agents.jd_analyst.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value.content = "Sorry, I cannot help with that."
        state = {"job_description": "dummy jd text here for testing", "agent_logs": []}
        result = jd_analyst_node(state)
    reqs = result["job_requirements"]
    for key in [
        "job_title",
        "required_skills",
        "nice_to_have",
        "min_experience_years",
        "education_level",
        "responsibilities",
        "domain",
    ]:
        assert key in reqs, f"Fallback missing key: {key}"
    assert isinstance(reqs["required_skills"], list)
