"""Member 4 — Report Writer Agent

Generates a comprehensive screening report summarizing all candidates,
their scores, rankings, and recommended interview questions.
"""

from state.schema import RecruitmentState


def report_writer_node(state: RecruitmentState) -> dict:
    """Generate and write the final screening report.

    Args:
        state: The current pipeline state containing scores, interview_questions, etc.

    Returns:
        A dict with final_report and report_path to merge into the state.
    """
    # TODO: Member 4
    return {}
