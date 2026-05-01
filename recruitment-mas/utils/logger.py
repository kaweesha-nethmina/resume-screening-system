"""
utils/logger.py
Member 4 — Shared observability/tracing utilities.

ALL agents in the pipeline import log_event() from here.
This module provides structured JSON event logging and JSONL file flushing,
giving full visibility into each agent's inputs, tool calls, and outputs.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def log_event(state: dict, agent: str, event: str, data: dict) -> None:
    """
    Append a structured JSON trace entry to state["agent_logs"].

    Called by every agent at the start and end of each pipeline step,
    and on every tool call, to provide full observability of the system.
    Entries are accumulated in memory and written to disk by flush_logs()
    at the end of the pipeline.

    Args:
        state (dict): The current RecruitmentState dict. The entry is
                      appended to state["agent_logs"] (created if absent).
        agent (str):  Agent name string (e.g. "JDAnalyst", "CVScreener",
                      "Scorer", "ReportWriter").
        event (str):  Event type string. Standard values:
                        "start"     — node begins execution
                        "complete"  — node finishes successfully
                        "tool_call" — agent invokes a custom tool
                        "error"     — an exception or failure occurred
        data (dict):  Arbitrary key-value pairs providing context for this
                      event (e.g. candidate name, file path, score, chars).

    Returns:
        None. Mutates state["agent_logs"] in place.

    Example:
        >>> state = {"agent_logs": []}
        >>> log_event(state, "ReportWriter", "start", {"top_candidate": "Alice"})
        >>> import json; entry = json.loads(state["agent_logs"][0])
        >>> assert entry["agent"] == "ReportWriter"
        >>> assert entry["event"] == "start"
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
    Write all accumulated log entries to a timestamped JSONL trace file.

    Each line in the output file is a valid JSON object representing one
    log event. The file can be parsed line-by-line for analysis or ingested
    into any observability platform that supports JSONL format.

    Args:
        logs (list):       List of JSON strings from state["agent_logs"].
                           Each element must be a valid JSON-serialised string.
        output_dir (str):  Directory path to write the trace file into.
                           Created automatically if it does not exist.
                           Defaults to "logs".

    Returns:
        str: The path to the written .jsonl file as a string.

    Example:
        >>> path = flush_logs(['{"agent":"A","event":"start","data":{}}'], "logs")
        >>> assert path.endswith(".jsonl")
        >>> assert Path(path).exists()
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"trace_{ts}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for entry in logs:
            f.write(entry + "\n")

    logger.info(f"[Logger] Flushed {len(logs)} log entries to {path}")
    return str(path)
