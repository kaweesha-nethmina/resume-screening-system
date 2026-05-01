"""Member 1 — JD Analyst Agent

Parses job descriptions and extracts key requirements including skills,
experience level, education, and nice-to-have qualifications.
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from state.schema import RecruitmentState
from tools.jd_tools import read_jd_file
from utils.logger import log_event

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise senior recruitment analyst.

Your ONLY task: extract structured requirements from a job description.

STRICT RULES:
1. Respond with ONLY valid JSON — no markdown, no explanation, no preamble.
2. NEVER invent or hallucinate skills not explicitly mentioned.
3. If a field is absent in the JD, use null or an empty list [].
4. required_skills = explicitly stated as mandatory/must-have.
5. nice_to_have = stated as "preferred", "bonus", "advantage".
6. min_experience_years = integer only; null if unspecified.
7. education_level = one of: "High School", "Bachelor", "Master", "PhD", null.

OUTPUT FORMAT (strict):
{
  "job_title": "string",
  "required_skills": ["skill1", "skill2"],
  "nice_to_have": ["skill1"],
  "min_experience_years": 3,
  "education_level": "Bachelor",
  "responsibilities": ["responsibility1", "responsibility2"],
  "domain": "string (e.g. Software Engineering, Finance)"
}"""

FALLBACK_REQUIREMENTS: dict = {
    "job_title": "Unknown",
    "required_skills": [],
    "nice_to_have": [],
    "min_experience_years": None,
    "education_level": None,
    "responsibilities": [],
    "domain": "Unknown",
}


def _extract_json(raw: str) -> str:
    """Robustly extract a JSON string from LLM output.

    Handles: plain JSON, ```json fences, ``` fences, leading/trailing text.

    Args:
        raw: Raw string output from the LLM.

    Returns:
        Cleaned string ready for json.loads().
    """
    # Strip markdown fences first
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
    # If there's a JSON object anywhere in the string, extract it
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def jd_analyst_node(state: RecruitmentState) -> dict:
    """Analyze the job description and extract structured requirements.

    Args:
        state: The current pipeline state containing job_description.

    Returns:
        A dict with job_requirements to merge into the state.
    """
    log_event(state, "JDAnalyst", "start", {"jd_chars": len(state["job_description"])})

    jd_text = state["job_description"]

    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract requirements from this job description:\n\n{jd_text}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    raw = _extract_json(raw)

    try:
        requirements = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("JSONDecodeError parsing LLM response: %s", raw[:200])
        log_event(state, "JDAnalyst", "error", {"reason": "json_parse_failed", "raw_preview": raw[:200]})
        requirements = FALLBACK_REQUIREMENTS.copy()

    # Enforce list types — small LLMs sometimes return a string instead of a list
    for list_field in ("required_skills", "nice_to_have", "responsibilities"):
        val = requirements.get(list_field)
        if not isinstance(val, list):
            requirements[list_field] = [val] if isinstance(val, str) and val else []

    # Enforce int or None for experience
    exp = requirements.get("min_experience_years")
    if exp is not None:
        try:
            requirements["min_experience_years"] = int(exp)
        except (ValueError, TypeError):
            requirements["min_experience_years"] = None

    # Fill any keys the LLM forgot to include
    for key, default in FALLBACK_REQUIREMENTS.items():
        requirements.setdefault(key, default)

    logger.info(
        "JD Analyst extracted %d required skills",
        len(requirements.get("required_skills", [])),
    )

    log_event(
        state,
        "JDAnalyst",
        "complete",
        {"skills_found": requirements.get("required_skills", [])},
    )

    return {"job_requirements": requirements}
