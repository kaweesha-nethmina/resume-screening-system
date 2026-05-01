"""Member 3 — Scorer Agent

Computes match scores between candidate profiles and job requirements,
and generates tailored interview questions for each candidate.
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from state.schema import RecruitmentState
from tools.scoring_tools import compute_match_score, save_scores_to_db
from utils.logger import log_event

logger = logging.getLogger(__name__)

JUSTIFY_PROMPT = """You are a recruitment evaluator writing for a hiring manager.

Given a candidate's profile and their computed match score, write EXACTLY ONE concise
sentence (max 25 words) justifying why this score is appropriate.

Be objective. Mention specific skills or experience gaps.
Do NOT restate the score. Output only the sentence — no preamble."""

def _get_justification(profile: dict, requirements: dict, score: dict, llm: ChatOllama) -> str:
    """Helper to get a justification sentence from the LLM.
    
    Args:
        profile: The candidate's profile dictionary.
        requirements: The job requirements dictionary.
        score: The computed match score dictionary.
        llm: The ChatOllama instance.
        
    Returns:
        A concise string justifying the score.
    """
    prompt = (
        f"Candidate Name: {profile.get('name', 'Unknown')}\n"
        f"Skills: {', '.join(profile.get('skills', []))}\n"
        f"Experience: {profile.get('total_experience_years', 0)} years\n"
        f"Required Skills: {', '.join(requirements.get('required_skills', []))}\n"
        f"Computed Score: {score.get('total_score', 0)}"
    )
    
    messages = [
        SystemMessage(content=JUSTIFY_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()

def scorer_node(state: RecruitmentState) -> dict:
    """Score candidates against job requirements and generate justifications.

    Args:
        state: The current pipeline state containing job_requirements and candidate_profiles.

    Returns:
        A dict with scores to merge into the state.
    """
    candidates = state.get("candidate_profiles", [])
    requirements = state.get("job_requirements", {})
    
    log_event(state, "Scorer", "start", {"candidates": len(candidates)})
    
    llm = ChatOllama(model="llama3:8b", temperature=0.1)
    
    scored = []
    
    for profile in candidates:
        score_dict = compute_match_score.invoke({"candidate": profile, "requirements": requirements})
        
        try:
            justification = _get_justification(profile, requirements, score_dict, llm)
        except Exception as e:
            logger.error(f"Error getting justification for {profile.get('name')}: {e}")
            justification = "Score reflects skill and experience alignment."
            
        entry = {
            "name": profile.get("name"),
            "email": profile.get("email"),
            "total_score": score_dict.get("total_score"),
            "skill_score": score_dict.get("skill_score"),
            "exp_score": score_dict.get("exp_score"),
            "edu_score": score_dict.get("edu_score"),
            "justification": justification,
            "source_file": profile.get("source_file")
        }
        
        scored.append(entry)
        log_event(state, "Scorer", "scored", {"name": entry["name"], "score": entry["total_score"]})
        
    # Sort descending by total_score
    ranked = sorted(scored, key=lambda x: x["total_score"], reverse=True)
    
    save_scores_to_db.invoke({"scores": ranked})
    
    if ranked:
        top_cand = ranked[0]
        logger.info(f"Top candidate: {top_cand['name']} with score {top_cand['total_score']}")
        
    return {"scores": ranked}
