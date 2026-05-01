"""Member 1 — JD Analyst Agent

Parses job descriptions and extracts key requirements including skills,
experience level, education, and nice-to-have qualifications.
"""

from state.schema import RecruitmentState


def jd_analyst_node(state: RecruitmentState) -> dict:
    """Analyze the job description and extract structured requirements.

    Args:
        state: The current pipeline state containing job_description.

    Returns:
        A dict with job_requirements to merge into the state.
    """
    # TODO: Member 1
    return {}
