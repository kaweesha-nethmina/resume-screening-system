# AI Resume Screening Multi-Agent System

A multi-agent system built with LangGraph and Ollama for automated resume screening against job descriptions.

## Overview

This project implements a pipeline of four cooperating agents:
1. **JD Analyst** — Parses and extracts key requirements from job descriptions
2. **CV Screener** — Extracts and structures candidate information from CVs
3. **Scorer** — Computes match scores between candidates and job requirements
4. **Report Writer** — Generates a final screening report with interview questions

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running

### Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull the Ollama model
ollama pull llama3:8b
```

## Running

```bash
# Run the full pipeline
python main.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
recruitment-mas/
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── state/                   # Shared state schema
├── graph/                   # LangGraph pipeline definition
├── agents/                  # Individual agent node implementations
├── tools/                   # Utility tools for each agent
├── utils/                   # Logging and shared utilities
├── tests/                   # Unit and integration tests
├── data/                    # Sample job descriptions and CVs
├── logs/                    # Auto-generated agent logs
└── outputs/                 # Auto-generated reports
```
