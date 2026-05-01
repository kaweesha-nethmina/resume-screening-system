"""Tools for parsing and analyzing job descriptions."""

import logging
from pathlib import Path

from langchain.tools import tool

logger = logging.getLogger(__name__)


@tool
def read_jd_file(file_path: str) -> str:
    """Read and return the text content of a job description file.

    Reads a job description from a .txt or .md file with utf-8 encoding,
    falling back to latin-1 if a UnicodeDecodeError occurs.

    Args:
        file_path: Path to the job description file (.txt or .md).

    Returns:
        The stripped text content of the job description.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is empty or contains only whitespace.

    Example:
        >>> from tools.jd_tools import read_jd_file
        >>> content = read_jd_file.invoke({"file_path": "data/sample_jd.txt"})
        >>> print(content[:100])
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Job description file not found: {path}")

    ext = path.suffix.lower()
    if ext not in (".txt", ".md"):
        logger.warning(
            "File extension '%s' is not .txt or .md. Reading anyway: %s",
            ext,
            path,
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decode failed for %s, falling back to latin-1", path
        )
        text = path.read_text(encoding="latin-1")

    stripped = text.strip()

    if not stripped:
        raise ValueError(f"Job description file is empty: {path}")

    return stripped


@tool
def get_jd_from_user() -> str:
    """Interactively collect a job description from the user via terminal input.

    Prompts the user to paste or type a job description directly into the
    terminal. Accepts multi-line input — the user types END on a new line
    alone to finish. Validates the input is non-empty before returning.

    Args:
        None

    Returns:
        str: The full job description text entered by the user, stripped
             of leading/trailing whitespace.

    Raises:
        ValueError: If the user submits empty or whitespace-only input.

    Example:
        >>> jd = get_jd_from_user()
        Paste your job description below. Type END on a new line when done:
        > We are hiring a Python developer...
        > Required: Python, FastAPI
        > END
    """
    print("\n" + "=" * 50)
    print("PASTE YOUR JOB DESCRIPTION BELOW")
    print("   Type or paste the JD, then type END on a")
    print("   new line alone and press Enter to finish.")
    print("=" * 50)

    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    content = "\n".join(lines).strip()

    if not content:
        raise ValueError(
            "No job description entered. Please paste a valid JD and type END to finish."
        )

    logger.info("[Tool:get_jd_from_user] Collected %d chars from terminal input", len(content))
    return content
