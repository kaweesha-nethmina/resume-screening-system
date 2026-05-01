"""Entry point for the AI Resume Screening Multi-Agent System."""

from pathlib import Path

from tools.jd_tools import read_jd_file
from graph.pipeline import build_pipeline
from utils.logger import flush_logs


def main():
    """Run the full resume screening pipeline."""
    logs_dir = Path("logs")
    outputs_dir = Path("outputs")
    logs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    jd_path = Path("data/sample_jd.txt")
    job_description = read_jd_file.invoke({"file_path": str(jd_path)})

    pipeline = build_pipeline()

    initial_state = {
        "job_description": job_description,
        "cv_folder_path": str(Path("data/cvs")),
        "job_requirements": {},
        "cv_file_paths": [],
        "cv_raw_texts": [],
        "candidate_profiles": [],
        "scores": [],
        "interview_questions": [],
        "final_report": "",
        "report_path": "",
        "agent_logs": [],
    }

    print("=== AI Resume Screening Pipeline ===\n")
    print(f"Job Description loaded from: {jd_path}\n")

    result = pipeline.invoke(initial_state)

    print("--- Pipeline Results ---")
    print(f"Job Requirements: {result.get('job_requirements', {})}")
    print(f"Candidates Processed: {len(result.get('candidate_profiles', []))}")
    print(f"Scores: {result.get('scores', [])}")
    print(f"Interview Questions: {result.get('interview_questions', [])}")
    print(f"Final Report: {result.get('final_report', '')}")
    print(f"Report Path: {result.get('report_path', '')}")

    logs = result.get("agent_logs", [])
    if logs:
        log_file = flush_logs(logs, output_dir=str(logs_dir))
        print(f"\nLogs written to: {log_file}")

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
