"""Logging utilities for the recruitment multi-agent system."""

import json
from datetime import datetime
from pathlib import Path


def log_event(state, agent, event, data):
    """Append a JSON-formatted log entry to state['agent_logs'].

    Args:
        state: The current RecruitmentState dict.
        agent: Name of the agent generating the log entry.
        event: Type/name of the event being logged.
        data: Arbitrary data associated with the event (must be JSON serializable).
    """
    entry = json.dumps(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent,
            "event": event,
            "data": data,
        }
    )
    state["agent_logs"].append(entry)


def flush_logs(logs, output_dir="logs"):
    """Write all log entries to a timestamped .jsonl file.

    Args:
        logs: List of JSON-formatted log strings.
        output_dir: Directory to write the log file to. Created if it doesn't exist.

    Returns:
        The path to the written log file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_logs_{timestamp}.jsonl"
    file_path = output_path / filename

    with open(file_path, "w") as f:
        for entry in logs:
            f.write(entry + "\n")

    return str(file_path)
