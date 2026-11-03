
# ==========================================================
# PROJECT: Disaster Death Analysis in India
# FILE: main.py
# PURPOSE: Run the complete ML project pipeline
# ==========================================================

from pathlib import Path
import subprocess
import sys


# ----------------------------------------------------------
# 1. PROJECT PATHS
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

# Ensure output folders exist
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 2. PIPELINE SCRIPTS
# ----------------------------------------------------------

SCRIPTS = [
    (
        "1/4 Preprocessing mortality data",
        "01_data_preprocessing.py"
    ),

    (
        "2/4 Clustering and graph generation",
        "02_clustering.py"
    ),

    (
        "3/4 State-wise temperature analysis",
        "temperature_analysis.py"
    ),

    (
        "4/4 Heat-stroke regression analysis",
        "03_regression.py"
    ),
]


# ----------------------------------------------------------
# 3. RUN INDIVIDUAL SCRIPT
# ----------------------------------------------------------

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
            f"{filename} failed.\n"
            "Check the error above before continuing."
        )

    print(f"\nCompleted successfully: {filename}")


# ----------------------------------------------------------
# 4. MAIN PIPELINE
# ----------------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("DISASTER DEATH ANALYSIS IN INDIA")
    print("COMPLETE MACHINE LEARNING PIPELINE")
    print("=" * 65)

    print("\nStarting project pipeline...")

    for description, filename in SCRIPTS:

        run_script(
            description,
            filename
        )

    # ------------------------------------------------------
    # FINAL COMPLETION MESSAGE
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    print("\nGenerated files:")

    print("\nCSV tables:")
    print(TABLES)

    print("\nGraphs and visualizations:")
    print(FIGURES)

    print("\nAll four analysis scripts executed successfully.")


# ----------------------------------------------------------
# 5. EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n" + "=" * 65)
        print("PROJECT PIPELINE STOPPED!")
        print("=" * 65)

        print("\nError:", error)

        sys.exit(1)
