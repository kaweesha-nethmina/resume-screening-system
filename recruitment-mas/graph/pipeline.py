"""LangGraph pipeline definition for the recruitment screening system."""

from langgraph.graph import StateGraph, START, END

from state.schema import RecruitmentState
from agents.jd_analyst import jd_analyst_node
from agents.cv_screener import cv_screener_node
from agents.scorer import scorer_node
from agents.report_writer import report_writer_node


def build_pipeline() -> StateGraph:
    """Build and compile the LangGraph StateGraph for the screening pipeline.

    Creates a sequential pipeline: START -> jd_analyst -> cv_screener -> scorer -> report_writer -> END

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    graph = StateGraph(RecruitmentState)

    graph.add_node("jd_analyst", jd_analyst_node)
    graph.add_node("cv_screener", cv_screener_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("report_writer", report_writer_node)

    graph.add_edge(START, "jd_analyst")
    graph.add_edge("jd_analyst", "cv_screener")
    graph.add_edge("cv_screener", "scorer")
    graph.add_edge("scorer", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()
