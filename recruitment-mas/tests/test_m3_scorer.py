"""Tests for Member 3 — Scorer Agent."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings
from tools.scoring_tools import compute_match_score, save_scores_to_db
from agents.scorer import scorer_node

SKILLS = ["Python", "SQL", "Docker", "React", "Java", "AWS"]
EDU_LEVELS = ["High School", "Bachelor", "Master", "PhD", None]

@given(
    cand_skills=st.lists(st.sampled_from(SKILLS), min_size=0, max_size=6), 
    req_skills=st.lists(st.sampled_from(SKILLS), min_size=1, max_size=6), 
    cand_exp=st.one_of(st.none(), st.integers(0,20)), 
    req_exp=st.one_of(st.none(), st.integers(1,10))
)
def test_score_always_0_to_100(cand_skills, req_skills, cand_exp, req_exp):
    """Test that total score is always clamped between 0 and 100."""
    candidate = {"skills": cand_skills, "total_experience_years": cand_exp}
    requirements = {"required_skills": req_skills, "min_experience_years": req_exp}
    
    # Check if tool is invoked directly or by `.invoke`
    # LangChain tools can be called directly or via invoke.
    try:
        result = compute_match_score.invoke({"candidate": candidate, "requirements": requirements})
    except AttributeError:
        result = compute_match_score(candidate, requirements)
        
    assert 0 <= result["total_score"] <= 100


@given(req_skills=st.lists(st.sampled_from(SKILLS), min_size=1, max_size=6))
def test_skill_score_monotonic(req_skills):
    """Test that candidate with more matching skills gets higher skill score."""
    candidate_a = {"skills": req_skills}
    candidate_b = {"skills": [s for s in SKILLS if s not in req_skills]}
    
    requirements = {"required_skills": req_skills}
    
    try:
        res_a = compute_match_score.invoke({"candidate": candidate_a, "requirements": requirements})
        res_b = compute_match_score.invoke({"candidate": candidate_b, "requirements": requirements})
    except AttributeError:
        res_a = compute_match_score(candidate_a, requirements)
        res_b = compute_match_score(candidate_b, requirements)
        
    assert res_a["skill_score"] >= res_b["skill_score"]


@given(req_exp=st.integers(1,5), cand_exp=st.integers(10,30))
def test_experience_bonus_capped(req_exp, cand_exp):
    """Test that experience score is capped at 100."""
    candidate = {"total_experience_years": cand_exp}
    requirements = {"min_experience_years": req_exp}
    
    try:
        result = compute_match_score.invoke({"candidate": candidate, "requirements": requirements})
    except AttributeError:
        result = compute_match_score(candidate, requirements)
        
    assert result["exp_score"] == 100


def test_perfect_match_scores_high():
    """Test a candidate with perfect match gets a high score."""
    candidate = {"skills": ["Python", "AWS"], "total_experience_years": 5, "education": "Master"}
    requirements = {"required_skills": ["Python", "AWS"], "min_experience_years": 5, "education_level": "Master"}
    
    try:
        result = compute_match_score.invoke({"candidate": candidate, "requirements": requirements})
    except AttributeError:
        result = compute_match_score(candidate, requirements)
        
    assert result["total_score"] >= 95


def test_zero_skills_zero_exp_scores_low():
    """Test candidate with no match gets low score."""
    candidate = {"skills": [], "total_experience_years": 0, "education": None}
    requirements = {"required_skills": ["Python", "AWS"], "min_experience_years": 5, "education_level": "Master"}
    
    try:
        result = compute_match_score.invoke({"candidate": candidate, "requirements": requirements})
    except AttributeError:
        result = compute_match_score(candidate, requirements)
        
    assert result["total_score"] <= 10


@patch("agents.scorer.ChatOllama")
def test_ranking_is_sorted(mock_llm):
    """Test that scorer_node returns sorted candidates."""
    mock_llm.return_value.invoke.return_value = MagicMock(content="Mock justification.")
    
    state = {
        "candidate_profiles": [
            {"name": "Bad Cand", "skills": [], "total_experience_years": 0, "education": None},
            {"name": "Good Cand", "skills": ["Python", "AWS"], "total_experience_years": 10, "education": "Master"}
        ],
        "job_requirements": {
            "required_skills": ["Python", "AWS"],
            "min_experience_years": 3,
            "education_level": "Bachelor"
        },
        "agent_logs": []
    }
    
    result = scorer_node(state)
    scores = result["scores"]
    
    assert len(scores) == 2
    assert scores[0]["total_score"] >= scores[1]["total_score"]
    assert scores[0]["name"] == "Good Cand"


def test_save_scores_to_db_creates_file():
    """Test database creation and insertion."""
    score_dict = [{
        "name": "Test User",
        "email": "test@example.com",
        "total_score": 90,
        "skill_score": 100,
        "exp_score": 80,
        "edu_score": 100,
        "justification": "Good candidate.",
        "source_file": "test.pdf"
    }]
    
    db_path = Path("logs/scores.db")
    if db_path.exists():
        db_path.unlink()
        
    try:
        save_scores_to_db.invoke({"scores": score_dict})
    except AttributeError:
        save_scores_to_db(score_dict)
        
    assert db_path.exists()


def test_output_has_all_keys():
    """Test output dictionary format."""
    candidate = {"skills": ["Python"], "total_experience_years": 3, "education": "Bachelor"}
    requirements = {"required_skills": ["Python"], "min_experience_years": 2, "education_level": "Bachelor"}
    
    try:
        result = compute_match_score.invoke({"candidate": candidate, "requirements": requirements})
    except AttributeError:
        result = compute_match_score(candidate, requirements)
        
    expected_keys = {"total_score", "skill_score", "exp_score", "edu_score", "skill_matches"}
    assert expected_keys.issubset(result.keys())

