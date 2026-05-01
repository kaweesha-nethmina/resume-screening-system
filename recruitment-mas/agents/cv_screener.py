"""Member 2 — CV Screener Agent

Extracts and structures candidate information from uploaded CVs including
contact details, work experience, education, skills, and certifications.
"""

import json
import logging
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from state.schema import RecruitmentState
from tools.cv_tools import extract_cv_text
from utils.logger import log_event

logger = logging.getLogger(__name__)

CV_PARSE_PROMPT = """You are an expert CV parser for a recruitment system.

STRICT RULES:
1. Output ONLY valid JSON — no markdown fences, no explanation.
2. NEVER invent information. If a field is absent, use null or [].
3. skills: list all technical skills, languages, frameworks mentioned.
4. total_experience_years: best estimate as integer; null if unclear.
5. education: highest qualification only (e.g. "BSc Computer Science").
6. previous_roles: list of job titles (most recent first), max 5.
7. certifications: any mentioned certificates; [] if none.

OUTPUT FORMAT:
{
  "name": "Full Name",
  "email": "email or null",
  "phone": "phone or null",
  "skills": ["Python", "SQL", ...],
  "total_experience_years": 4,
  "education": "BSc Computer Science",
  "previous_roles": ["Software Engineer", "Junior Dev"],
  "certifications": ["AWS Certified", ...]
}"""


def _get_cv_files(folder_path: str) -> list[str]:
    """Get sorted list of CV file paths from a folder."""
    path = Path(folder_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder_path}")
    
    files = []
    # Collect files case-insensitively
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in ['.pdf', '.txt']:
            files.append(str(p.absolute()))
            
    if not files:
        raise ValueError(f"No CV files found in folder: {folder_path}")
        
    return sorted(files)


def _parse_single_cv(cv_text: str, llm: ChatOllama) -> dict:
    """Parse a single CV text using LLM to extract structured data."""
    fallback = {
        "name": "Parse Error", 
        "email": None, 
        "phone": None,
        "skills": [], 
        "total_experience_years": None,
        "education": None, 
        "previous_roles": [], 
        "certifications": []
    }
    
    messages = [
        SystemMessage(content=CV_PARSE_PROMPT),
        HumanMessage(content=f"Parse this resume:\n\n{cv_text[:3000]}")
    ]
    
    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip()
        
        if "```" in raw_output:
            raw_output = raw_output.split("```")[1].strip()
            if raw_output.startswith("json"):
                raw_output = raw_output[4:].strip()
                
        profile = json.loads(raw_output)
        
        # Coerce skills to list if necessary
        if not isinstance(profile.get("skills"), list):
            if isinstance(profile.get("skills"), str):
                profile["skills"] = [s.strip() for s in profile["skills"].split(",") if s.strip()]
            elif profile.get("skills") is None:
                profile["skills"] = []
            else:
                profile["skills"] = [str(profile.get("skills"))]
                
        return profile
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError parsing CV: {e}")
        return fallback
    except Exception as e:
        logger.error(f"Error parsing CV: {e}")
        return fallback


def cv_screener_node(state: RecruitmentState) -> dict:
    """Screen CVs and extract candidate profiles.

    Args:
        state: The current pipeline state containing cv_folder_path.

    Returns:
        A dict with cv_file_paths, cv_raw_texts, and candidate_profiles to merge into the state.
    """
    log_event(state, "CVScreener", "start", {"folder": state["cv_folder_path"]})
    
    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    
    try:
        cv_paths = _get_cv_files(state["cv_folder_path"])
    except Exception as e:
        logger.error(f"Failed to get CV files: {e}")
        raise
        
    raw_texts = []
    profiles = []
    
    for path in cv_paths:
        try:
            text = extract_cv_text.invoke({"file_path": path})
            raw_texts.append(text)
            
            profile = _parse_single_cv(text, llm)
            profile["source_file"] = path
            profiles.append(profile)
            
            log_event(state, "CVScreener", "parsed_cv", {"file": path, "name": profile.get("name")})
        except Exception as e:
            logger.error(f"Error processing CV {path}: {e}")
            continue
            
    return {
        "cv_file_paths": cv_paths, 
        "cv_raw_texts": raw_texts, 
        "candidate_profiles": profiles
    }
