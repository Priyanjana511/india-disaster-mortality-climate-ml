
# ==========================================================
# PROJECT: Disaster Death Analysis in India
# FILE: main.py
# PURPOSE: Run the complete ML project pipeline
# ==========================================================

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SCRIPTS = [
    ("1/4 Preprocessing mortality data",
     "01_data_preprocessing.py"),

    ("2/4 Clustering and graph generation",
     "02_clustering.py"),

    ("3/4 State-wise temperature analysis",
     "temperature_analysis.py"),

    ("4/4 Heat-stroke regression analysis",
     "03_regression.py"),
]


def run_script(description, filename):

    script_path = SRC / filename

    print("\n" + "=" * 65)
    print(description)
    print("=" * 65)

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{filename} failed. "
            "Check the error above before continuing."
        )

    print(f"\nCompleted: {filename}")


def main():

    print("\nDISASTER DEATH ANALYSIS IN INDIA")
    print("Starting complete project pipeline...")

    for description, filename in SCRIPTS:
        run_script(description, filename)

    print("\n" + "=" * 65)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check the outputs folder.")
    print("=" * 65)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nPROJECT STOPPED:", error)
        sys.exit(1)
