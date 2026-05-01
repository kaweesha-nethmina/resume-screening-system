"""Member 3 — Scorer Agent

Computes match scores between candidate profiles and job requirements,
and generates tailored interview questions for each candidate.
"""

from state.schema import RecruitmentState


def scorer_node(state: RecruitmentState) -> dict:
    """Score candidates against job requirements and generate interview questions.

    Args:
        state: The current pipeline state containing job_requirements and candidate_profiles.

    Returns:
        A dict with scores and interview_questions to merge into the state.
    """
    # TODO: Member 3
    return {}
