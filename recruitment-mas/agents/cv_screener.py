"""Member 2 — CV Screener Agent

Extracts and structures candidate information from uploaded CVs including
contact details, work experience, education, skills, and certifications.
"""

from state.schema import RecruitmentState


def cv_screener_node(state: RecruitmentState) -> dict:
    """Screen CVs and extract candidate profiles.

    Args:
        state: The current pipeline state containing cv_file_paths and cv_raw_texts.

    Returns:
        A dict with candidate_profiles to merge into the state.
    """
    # TODO: Member 2
    return {}
