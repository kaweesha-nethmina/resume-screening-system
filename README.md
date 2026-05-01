# AI Resume Screening Multi-Agent System

> SE4010 — CTSE Assignment 2 | Built with LangGraph + Ollama (llama3:8b)

A production-quality multi-agent system for automated resume screening. The system accepts a job description and a folder of candidate CVs as input, and autonomously produces a ranked shortlist with detailed analysis, scores, and tailored interview questions — all running locally at zero cloud cost.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Component Description](#component-description)
- [Testing](#testing)
- [Observability](#observability)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

---

## Overview

The system orchestrates four specialised agents in a sequential LangGraph pipeline:

| Step | Agent | Responsibility |
|------|-------|----------------|
| 1 | **JD Analyst** (Member 1 - Jayarathna S K N) | Extracts structured requirements from job descriptions (skills, experience, education, domain) | 

| 2 | **CV Screener** (Member 2 - Weerasinghe D I) | Parses PDF/TXT CVs into structured candidate profiles (name, skills, experience, education) |

| 3 | **Scorer** (Member 3 - Wahalathanthri H C) | Computes weighted match scores (50% skills, 30% experience, 20% education) with LLM justifications |

| 4 | **Report Writer** (Member 4 - Wijekoon K S M) | Generates a professional Markdown recruitment report with ranked shortlist and interview questions |

All agents share a single `RecruitmentState` TypedDict that flows through the pipeline. Each agent reads only the fields it needs and returns a partial dict that LangGraph merges automatically — ensuring zero context loss between handoffs.

---

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐     ┌───────────────┐
│  JD Analyst │ ──▶ │ CV Screener  │ ──▶ │ Scorer   │ ──▶ │ Report Writer │
│  (M1)       │     │ (M2)         │     │ (M3)     │     │ (M4)          │
└─────────────┘     └──────────────┘     └──────────┘     └───────────────┘
       │                   │                  │                  │
   read_jd_file       extract_cv_text    compute_match_score  write_report_file
   get_jd_from_user                      save_scores_to_db
```

**Shared State**: `RecruitmentState` (11 fields) — defined in `state/schema.py`  
**LLM Engine**: Ollama `llama3:8b` (local, zero cloud cost)  
**Orchestrator**: LangGraph `StateGraph` with sequential edge routing  
**Observability**: JSONL structured logging + SQLite score history

---

## Project Structure

```
recruitment-mas/
│
├── main.py                    # Entry point & interactive terminal UI
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── state/
│   └── schema.py              # RecruitmentState TypedDict (shared state)
│
├── graph/
│   └── pipeline.py            # LangGraph StateGraph builder (4 nodes + edges)
│
├── agents/
│   ├── jd_analyst.py          # Member 1 — JD extraction with anti-hallucination
│   ├── cv_screener.py         # Member 2 — CV parsing with per-file resilience
│   ├── scorer.py              # Member 3 — Deterministic scoring + LLM justification
│   └── report_writer.py       # Member 4 — Markdown report generation
│
├── tools/
│   ├── jd_tools.py            # Member 1 — read_jd_file, get_jd_from_user
│   ├── cv_tools.py            # Member 2 — extract_cv_text (pdfplumber)
│   ├── scoring_tools.py       # Member 3 — compute_match_score, save_scores_to_db
│   └── report_tools.py        # Member 4 — write_report_file
│
├── utils/
│   ├── logger.py              # JSONL structured logging (log_event, flush_logs)
│   └── terminal.py            # ANSI styled terminal UI helpers
│
├── tests/
│   ├── test_m1_jd_analyst.py  # Member 1 — 13 tests (unit + Hypothesis + mock)
│   ├── test_m2_cv_screener.py # Member 2 — CV parser tests
│   ├── test_m3_scorer.py      # Member 3 — Scoring property-based tests
│   └── test_m4_report_writer.py # Member 4 — Report tests + LLM-as-a-Judge
│
├── data/
│   ├── sample_jd.txt          # Default job description
│   └── cvs/                   # Candidate CV folder (PDF + TXT)
│
├── logs/                      # Auto-generated: JSONL traces + SQLite DB
└── outputs/                   # Auto-generated: Markdown recruitment reports
```

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| Ollama | Latest | Local LLM inference engine |
| pip | Latest | Python package manager |
| venv | stdlib | Isolated Python environment |

**Ollama Setup** (required before running):

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Start the Ollama service
3. Pull the model: `ollama pull llama3:8b`

---

## Installation

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd recruitment-mas
```

### Step 2: Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain>=0.2.0` — Agent framework, @tool decorator, message types
- `langchain-ollama>=0.1.0` — Ollama LLM integration
- `langchain-community>=0.2.0` — Community tools and utilities
- `langgraph>=0.1.0` — StateGraph orchestration
- `pdfplumber>=0.10.0` — PDF text extraction
- `hypothesis>=6.100.0` — Property-based testing
- `pytest>=8.0.0` — Test runner
- `python-dotenv>=1.0.0` — Environment variable management

### Step 4: Verify Ollama model

```bash
ollama list
```

Ensure `llama3:8b` appears in the list. If not:

```bash
ollama pull llama3:8b
```

### Step 5: Prepare data

Place your job description in `data/sample_jd.txt` and candidate CVs (`.pdf` or `.txt`) in `data/cvs/`.

---

## Running the System

### Full Pipeline

```bash
python main.py
```

At startup, you will see an interactive mode selector:

```
┌── Job Description Source ──┐
│  [1]  Type / paste directly in terminal
│  [2]  Load default file (data/sample_jd.txt)
│  [3]  Load from custom file path
│
│  > Enter choice (1/2/3):
```

- **Option 1**: Paste a job description directly. Type `END` on a new line to finish.
- **Option 2**: Load from `data/sample_jd.txt`.
- **Option 3**: Provide a custom file path (`.txt` or `.md`).

The pipeline then runs all four agents sequentially with a styled animated spinner. Upon completion, it displays:
- Extracted job requirements (title, domain, skills, experience)
- Candidate count and top-3 scores
- Report save path
- Observability log entry count

### Quick Run (no interaction)

To run without the interactive prompt, modify `main.py` to skip `select_jd_input_mode()` and pass a fixed JD string directly to the initial state.

---

## Component Description

### `main.py` — Entry Point

The main entry point handles:
- Interactive JD input mode selection (paste / default file / custom file)
- LangGraph pipeline construction via `build_pipeline()`
- Initial state preparation with all 11 fields
- Pipeline invocation with styled spinner animation
- Results display with color-coded sections
- Log flushing to disk after completion

### `state/schema.py` — Shared State

Defines `RecruitmentState` as a `TypedDict` with 11 fields. All agents read from and write to this single state object. LangGraph handles partial updates automatically.

### `graph/pipeline.py` — Pipeline Orchestrator

Builds a LangGraph `StateGraph` with four nodes and sequential edges:
```
START → jd_analyst → cv_screener → scorer → report_writer → END
```

### `agents/jd_analyst.py` — Member 1

- **Model**: `llama3:8b` at temperature `0.0` (deterministic)
- **System Prompt**: Enforces JSON-only output, anti-hallucination rules, strict field types
- **Resilience**: `_extract_json()` regex strips markdown fences; `FALLBACK_REQUIREMENTS` constant on parse failure; post-parse type coercion for lists and integers; `setdefault` merge guarantees all 7 keys present
- **Output**: 7-field dict (job_title, required_skills, nice_to_have, min_experience_years, education_level, responsibilities, domain)

### `agents/cv_screener.py` — Member 2

- **Model**: `llama3:8b` at temperature `0.0`
- **CV Scanning**: `_get_cv_files()` collects `.pdf` and `.txt` files, raises `ValueError` if empty
- **Parsing**: `_parse_single_cv()` truncates CV to 3000 chars, calls LLM, strips fences, coerces skills to list
- **Resilience**: Per-CV try/except — one failed CV does not abort the pipeline; fallback profile with `name='Parse Error'` on JSONDecodeError
- **Output**: List of candidate profiles with source_file tagging

### `agents/scorer.py` — Member 3

- **Scoring**: Deterministic Python algorithm via `compute_match_score` tool (not LLM)
- **Weights**: Skills 50%, Experience 30%, Education 20% — clamped to [0, 100]
- **Justification**: LLM generates one concise sentence per candidate (temperature `0.1`)
- **Persistence**: `save_scores_to_db` writes to `logs/scores.db` with run history
- **Output**: Ranked list sorted descending by total_score

### `agents/report_writer.py` — Member 4

- **Model**: `llama3:8b` at temperature `0.3` (natural prose)
- **Context Builder**: `_build_context()` assembles compact LLM-friendly summary under 500 tokens
- **Template**: Strict Markdown structure with 5 mandatory sections (Executive Summary, Ranked Shortlist, Top 3 Analysis, Interview Questions, Recommendation)
- **Output**: Markdown report saved to `outputs/report_YYYYMMDD_HHMMSS.md`
- **Cleanup**: Calls `flush_logs()` to persist full JSONL trace to disk

### `tools/jd_tools.py` — Member 1 Tools

- **`read_jd_file`**: Reads `.txt`/`.md` files with utf-8 primary and latin-1 fallback; raises `FileNotFoundError` or `ValueError`
- **`get_jd_from_user`**: Interactive multi-line terminal input with styled border; terminates on `END`

### `tools/cv_tools.py` — Member 2 Tool

- **`extract_cv_text`**: PDF via pdfplumber (page-by-page text block collection); TXT via multi-encoding fallback (utf-8 → latin-1 → cp1252); never crashes — returns empty string on total failure

### `tools/scoring_tools.py` — Member 3 Tools

- **`compute_match_score`**: Weighted scoring with case-insensitive skill matching, experience ratio with bonus for overqualification, fuzzy education mapping (`'bsc'` → Bachelor, `'msc'` → Master)
- **`save_scores_to_db`**: SQLite `CREATE TABLE IF NOT EXISTS` + `INSERT` with run_timestamp; accumulates across runs

### `tools/report_tools.py` — Member 4 Tool

- **`write_report_file`**: Auto-creates parent directories; prepends HTML comment metadata header with ISO timestamp and character count; returns absolute path

### `utils/logger.py` — Observability

- **`log_event(state, agent, event, data)`**: Appends JSON-serialised trace entry to `state['agent_logs']`
- **`flush_logs(logs, output_dir)`**: Writes all entries to `logs/agent_logs_YYYYMMDD_HHMMSS.jsonl`

### `utils/terminal.py` — Terminal UI

ANSI-styled helper functions for colored output, bordered sections, pill-style lists, spinners, and dividers. Pure Python — no external UI libraries.

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Individual Member Tests

```bash
# Member 1 — JD Analyst (13 tests)
pytest tests/test_m1_jd_analyst.py -v

# Member 2 — CV Screener
pytest tests/test_m2_cv_screener.py -v

# Member 3 — Scorer
pytest tests/test_m3_scorer.py -v

# Member 4 — Report Writer
pytest tests/test_m4_report_writer.py -v
```

### Test Levels

| Level | Type | Tool | Coverage |
|-------|------|------|----------|
| L1 | Unit tests | pytest | All members — tool & agent function tests |
| L2 | Property-based | pytest + Hypothesis | Members 1, 2, 3 — invariants under random input |
| L3 | LLM-as-a-Judge | pytest + ChatOllama | Member 4 — second LLM evaluates report quality (4 criteria, ≥75% pass) |

### Member 1 Test Highlights

- **Unit**: `read_jd_file` reads correctly, handles missing files, rejects empty files
- **Agent**: Output contains all 7 required keys; `required_skills` is always a list
- **Property-based (Hypothesis)**: Agent returns valid dict for ANY random text input; all keys always present; skills always list type
- **Mock**: Type enforcement coerces string skills to list; fallback returns complete schema on garbage LLM output

---

## Observability

Every agent logs structured events to the shared `agent_logs` list:

```json
{
  "timestamp": "2026-05-01T09:53:44.123456",
  "agent": "JDAnalyst",
  "event": "start",
  "data": {"jd_chars": 1858}
}
```

**Log events per agent**:

| Agent | Events |
|-------|--------|
| JD Analyst | `start` (jd_chars), `complete` (skills_found), `error` (raw preview) |
| CV Screener | `start` (folder path), `parsed_cv` per file (file path + candidate name) |
| Scorer | `start` (candidate count), `scored` per candidate (name + total_score) |
| Report Writer | `start` (top candidate name), `complete` (report path + char count) |

After pipeline completion, Member 4 calls `flush_logs()` which persists all entries to `logs/agent_logs_YYYYMMDD_HHMMSS.jsonl`.

Additionally, the Scorer persists scores to `logs/scores.db` for historical comparison across runs.

---

## Output Files

| Output | Location | Format |
|--------|----------|--------|
| Recruitment report | `outputs/report_YYYYMMDD_HHMMSS.md` | Markdown with metadata header |
| Agent trace log | `logs/agent_logs_YYYYMMDD_HHMMSS.jsonl` | JSON Lines (one entry per event) |
| Score history | `logs/scores.db` | SQLite database (accumulates across runs) |
| System log | `logs/system.log` | Standard Python logging |

---

## Troubleshooting

### `ollama: command not found`
Install Ollama from [https://ollama.ai](https://ollama.ai) and ensure it is in your PATH.

### `Error: model "llama3:8b" not found`
```bash
ollama pull llama3:8b
```

### `ModuleNotFoundError: No module named 'langchain'`
Ensure your virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `No CV files found in folder`
Place at least one `.pdf` or `.txt` file in the `data/cvs/` directory.

### `ConnectionError` when calling Ollama
Ensure the Ollama service is running:
```bash
ollama list
```
If the service is not running, start it according to your OS instructions.

### Slow pipeline execution
The pipeline calls the LLM once per CV plus once per candidate for justification. With many CVs, this may take several minutes. The `llama3:8b` model runs locally — performance depends on your hardware.

---

## License

This project is submitted as part of SE4010 — CTSE Assignment 2 at the Sri Lanka Institute of Information Technology (SLIIT).
