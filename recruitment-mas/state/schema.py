"""Shared state schema for the recruitment multi-agent system."""

from typing import List, TypedDict


class RecruitmentState(TypedDict):
    """State shared across all agents in the screening pipeline."""

    job_description: str
    cv_folder_path: str
    job_requirements: dict
    cv_file_paths: List[str]
    cv_raw_texts: List[str]
    candidate_profiles: List[dict]
    scores: List[dict]
    interview_questions: List[dict]
    final_report: str
    report_path: str
    agent_logs: List[str]
