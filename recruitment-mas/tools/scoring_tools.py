"""Tools for computing match scores between candidates and job requirements."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DB_PATH = "logs/scores.db"
EDUCATION_WEIGHTS = {
    "High School": 25, 
    "Bachelor": 50,
    "Master": 75, 
    "PhD": 100, 
    None: 0
}

@tool
def compute_match_score(candidate: dict, requirements: dict) -> dict:
    """Compute a match score between a candidate profile and job requirements.

    The scoring breakdown is as follows:
    - Skills Score (50% weight)
    - Experience Score (30% weight)
    - Education Score (20% weight)

    Args:
        candidate (dict): Dict containing the candidate's extracted profile data.
            Expected keys: 'skills', 'total_experience_years', 'education'.
        requirements (dict): Dict containing the parsed job requirements.
            Expected keys: 'required_skills', 'min_experience_years', 'education_level'.

    Returns:
        dict: A dict containing the computed score and breakdown details with keys:
            'total_score' (int), 'skill_score' (int), 'exp_score' (int),
            'edu_score' (int), 'skill_matches' (list).

    Example:
        >>> compute_match_score({"skills": ["Python"], "total_experience_years": 5, "education": "BSc"}, 
        ...                     {"required_skills": ["Python"], "min_experience_years": 3, "education_level": "Bachelor"})
        {'total_score': 100, 'skill_score': 100, 'exp_score': 100, 'edu_score': 100, 'skill_matches': ['python']}
    """
    # Skills Score (50% weight)
    req_skills_raw = requirements.get("required_skills", [])
    cand_skills_raw = candidate.get("skills", [])
    
    req_skills = [s.lower().strip() for s in req_skills_raw] if isinstance(req_skills_raw, list) else []
    cand_skills = [s.lower().strip() for s in cand_skills_raw] if isinstance(cand_skills_raw, list) else []
    
    matches = [s for s in req_skills if s in cand_skills]
    
    if req_skills:
        skill_score = int(len(matches) / len(req_skills) * 100)
    else:
        skill_score = 100

    # Experience Score (30% weight)
    min_exp = requirements.get("min_experience_years")
    cand_exp = candidate.get("total_experience_years")
    
    if min_exp is None:
        exp_score = 100
    elif cand_exp is None:
        exp_score = 0
    elif cand_exp >= min_exp:
        exp_score = min(100, 100 + (cand_exp - min_exp) * 5)
    else:
        # Prevent division by zero if min_exp is 0 (though >= condition catches cand_exp >= 0)
        exp_score = max(0, int(cand_exp / min_exp * 100))

    # Education Score (20% weight)
    edu_required = requirements.get("education_level")
    edu_candidate_str = candidate.get("education", "") or ""
    
    # Map candidate string to level
    edu_candidate_str_lower = edu_candidate_str.lower()
    cand_edu_level = None
    if "phd" in edu_candidate_str_lower:
        cand_edu_level = "PhD"
    elif "master" in edu_candidate_str_lower or "msc" in edu_candidate_str_lower:
        cand_edu_level = "Master"
    elif "bachelor" in edu_candidate_str_lower or "bsc" in edu_candidate_str_lower:
        cand_edu_level = "Bachelor"
    elif "high school" in edu_candidate_str_lower:
        cand_edu_level = "High School"
        
    req_weight = EDUCATION_WEIGHTS.get(edu_required, 50)
    cand_weight = EDUCATION_WEIGHTS.get(cand_edu_level, 0)
    
    if req_weight == 0:
        edu_score = 100
    else:
        edu_score = min(100, int(cand_weight / req_weight * 100))

    # Weighted total
    total = int(skill_score * 0.50 + exp_score * 0.30 + edu_score * 0.20)
    total = max(0, min(100, total))
    
    return {
        "total_score": total, 
        "skill_score": skill_score,
        "exp_score": exp_score, 
        "edu_score": edu_score, 
        "skill_matches": matches
    }

@tool
def save_scores_to_db(scores: list) -> str:
    """Save candidate scores to a SQLite database.
    
    Args:
        scores (list): A list of dictionaries containing the scored candidates' data.
        
    Returns:
        str: The path to the SQLite database where scores were saved.
        
    Raises:
        sqlite3.Error: If there is an error interacting with the database.
    """
    Path("logs").mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS candidate_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        run_timestamp TEXT, 
        rank INTEGER, 
        name TEXT, 
        email TEXT, 
        total_score INTEGER, 
        skill_score INTEGER, 
        exp_score INTEGER, 
        edu_score INTEGER, 
        justification TEXT, 
        source_file TEXT
    )
    ''')
    
    run_timestamp = datetime.now().isoformat()
    
    for rank, entry in enumerate(scores, 1):
        cursor.execute('''
        INSERT INTO candidate_scores 
        (run_timestamp, rank, name, email, total_score, skill_score, exp_score, edu_score, justification, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_timestamp,
            rank,
            entry.get("name"),
            entry.get("email"),
            entry.get("total_score"),
            entry.get("skill_score"),
            entry.get("exp_score"),
            entry.get("edu_score"),
            entry.get("justification"),
            entry.get("source_file")
        ))
        
    conn.commit()
    conn.close()
    
    logger.info(f"Saved {len(scores)} candidate scores to database at {DB_PATH}")
    return DB_PATH
