"""Entry point for the AI Resume Screening Multi-Agent System."""

from pathlib import Path

from tools.jd_tools import read_jd_file
from graph.pipeline import build_pipeline
from utils.logger import flush_logs

JD_PATH = "data/sample_jd.txt"


def select_jd_input_mode() -> str:
    """Ask the user how they want to provide the job description.

    Returns:
        str: The full job description text, from whichever source the user chose.
    """
    print("\n" + "=" * 50)
    print("AI RESUME SCREENING MAS")
    print("=" * 50)
    print("\nHow would you like to provide the Job Description?\n")
    print("  [1] Type or paste it now in the terminal")
    print("  [2] Load from file  (default: data/sample_jd.txt)")
    print("  [3] Load from a custom file path")
    print()

    choice = input("Enter choice (1 / 2 / 3): ").strip()

    if choice == "1":
        from tools.jd_tools import get_jd_from_user
        jd_text = get_jd_from_user.run("")
        print(f"\nJD collected - {len(jd_text)} characters")
        return jd_text

    elif choice == "3":
        custom_path = input("Enter file path: ").strip()
        from tools.jd_tools import read_jd_file
        jd_text = read_jd_file.run(custom_path)
        print(f"\nJD loaded from: {custom_path} - {len(jd_text)} characters")
        return jd_text

    else:
        from tools.jd_tools import read_jd_file
        jd_text = read_jd_file.run(JD_PATH)
        print(f"\nJD loaded from: {JD_PATH} - {len(jd_text)} characters")
        return jd_text


def main():
    """Run the full resume screening pipeline."""
    logs_dir = Path("logs")
    outputs_dir = Path("outputs")
    logs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    jd_text = select_jd_input_mode()

    pipeline = build_pipeline()

    initial_state = {
        "job_description": jd_text,
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
    print(f"Job Description length: {len(jd_text)} characters\n")

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
