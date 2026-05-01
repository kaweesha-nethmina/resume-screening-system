"""
utils/logger.py
Member 4 — Shared observability/tracing utilities.
ALL agents import log_event() from here.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def log_event(state: dict, agent: str, event: str, data: dict) -> None:
    """
    Append a structured JSON trace entry to state["agent_logs"].

    Called by every agent at the start and end of each step,
    and on every tool call, to provide full observability.

    Args:
        state:  The current RecruitmentState dict (mutated in place).
        agent:  Agent name string (e.g. "JDAnalyst", "CVScreener").
        event:  Event type (e.g. "start", "complete", "tool_call", "error").
        data:   Arbitrary dict of additional context for this event.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "event": event,
        "data": data,
    }
    state.setdefault("agent_logs", []).append(json.dumps(entry))
    logger.debug(f"[{agent}] {event} — {data}")


def flush_logs(logs: list, output_dir: str = "logs") -> str:
    """
    Write all accumulated log entries to a JSONL trace file.

    Args:
        logs:       List of JSON strings from state["agent_logs"].
        output_dir: Directory to write the trace file into.

    Returns:
        str: Path to the written .jsonl file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"trace_{ts}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for entry in logs:
            f.write(entry + "\n")

    logger.info(f"[Logger] Flushed {len(logs)} log entries to {path}")
    return str(path)
