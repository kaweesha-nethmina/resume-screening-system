"""Tests for Member 2 — CV Screener Agent."""

import pytest
import os
from pathlib import Path
from hypothesis import given, settings
from hypothesis import strategies as st

from tools.cv_tools import extract_cv_text
from agents.cv_screener import cv_screener_node, _parse_single_cv, _get_cv_files
from langchain_ollama import ChatOllama
from state.schema import RecruitmentState

REQUIRED_PROFILE_KEYS = ["name", "skills", "total_experience_years", "education", "previous_roles"]

def test_extract_txt_file(tmp_path):
    cv_file = tmp_path / "test.txt"
    cv_file.write_text("John Doe\nPython Developer\n5 years experience", encoding="utf-8")
    
    result = extract_cv_text.invoke({"file_path": str(cv_file)})
    assert "John Doe" in result

def test_extract_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_cv_text.invoke({"file_path": "no/such/file.pdf"})

def test_extract_returns_string(tmp_path):
    cv_file = tmp_path / "test2.txt"
    cv_file.write_text("Any text here", encoding="utf-8")
    result = extract_cv_text.invoke({"file_path": str(cv_file)})
    assert isinstance(result, str)


def test_cv_screener_profiles_have_required_keys(tmp_path):
    cv_file = tmp_path / "test_cv.txt"
    cv_file.write_text("John Doe\nSoftware Engineer with 4 years experience in Python and SQL.\nBSc Computer Science\nPrevious roles: Software Engineer", encoding="utf-8")
    
    state = RecruitmentState(
        job_description="",
        jd_requirements={},
        cv_folder_path=str(tmp_path),
        cv_file_paths=[],
        cv_raw_texts=[],
        candidate_profiles=[],
        evaluated_candidates=[],
        rankings=[],
        events=[],
        metadata={},
        agent_logs=[]
    )
    
    result = cv_screener_node(state)
    profiles = result["candidate_profiles"]
    
    assert len(profiles) > 0
    for p in profiles:
        for key in REQUIRED_PROFILE_KEYS:
            assert key in p

def test_cv_screener_skills_is_list(tmp_path):
    cv_file = tmp_path / "test_cv.txt"
    cv_file.write_text("John Doe\nSoftware Engineer with 4 years experience in Python and SQL.\nBSc Computer Science\nPrevious roles: Software Engineer", encoding="utf-8")
    
    state = RecruitmentState(
        job_description="",
        jd_requirements={},
        cv_folder_path=str(tmp_path),
        cv_file_paths=[],
        cv_raw_texts=[],
        candidate_profiles=[],
        evaluated_candidates=[],
        rankings=[],
        events=[],
        metadata={},
        agent_logs=[]
    )
    
    result = cv_screener_node(state)
    profiles = result["candidate_profiles"]
    
    assert len(profiles) > 0
    for p in profiles:
        assert isinstance(p["skills"], list)

def test_empty_folder_raises(tmp_path):
    state = RecruitmentState(
        job_description="",
        jd_requirements={},
        cv_folder_path=str(tmp_path),
        cv_file_paths=[],
        cv_raw_texts=[],
        candidate_profiles=[],
        evaluated_candidates=[],
        rankings=[],
        events=[],
        metadata={},
        agent_logs=[]
    )
    
    with pytest.raises(ValueError):
        cv_screener_node(state)

@given(st.text(min_size=10, max_size=500))
@settings(max_examples=10)
def test_parse_single_cv_never_crashes(text):
    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    result = _parse_single_cv(text, llm)
    assert isinstance(result, dict)
    assert "skills" in result

@given(st.text(min_size=20, max_size=200))
def test_cv_screener_skills_always_list_property(broken_json_text):
    class MockResponse:
        def __init__(self, content):
            self.content = content
            
    class MockLLM:
        def invoke(self, messages):
            # Return broken json where skills is a string
            return MockResponse('{"name": "Test", "skills": "Python, SQL", "total_experience_years": 4, "education": "BSc", "previous_roles": []}')
            
    llm = MockLLM()
    result = _parse_single_cv(broken_json_text, llm)
    assert isinstance(result["skills"], list)
    assert "Python" in result["skills"]
    assert "SQL" in result["skills"]
