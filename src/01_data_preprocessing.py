# ============================================================
# PROJECT: Deaths from Lightning, Heat Stroke and Floods
# FILE: 01_data_preprocessing.py
# PURPOSE:
#   1. Read and standardise NCRB/ADSI death CSV files
#   2. Read State/UT population from Census 2011 Excel workbook
#   3. Calculate hazard-specific death rates per lakh population
#   4. Save all CSV files inside outputs/tables/
# ============================================================

from pathlib import Path
import re
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. STANDARDISE STATE NAMES
# ------------------------------------------------------------

def standardise_state(value):
    """Clean State/UT names while preserving historically distinct units."""

    name = re.sub(r"\s+", " ", str(value).strip().upper())
    name = name.replace("&", "AND")

    aliases = {
        "ORISSA": "ODISHA",
        "UTTARANCHAL": "UTTARAKHAND",
        "NCT OF DELHI": "DELHI",
        "DELHI (UT)": "DELHI",
        "DELHI UT": "DELHI",
        "A AND N ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        "D AND N HAVELI": "DADRA AND NAGAR HAVELI",
    }

    # Telangana is not merged into Andhra Pradesh.
    # The 2011 Census population for undivided Andhra Pradesh
    # cannot be used as Telangana's separate denominator.

    return aliases.get(name, name)


# ------------------------------------------------------------
# 3. STANDARDISE CAUSE NAMES
# ------------------------------------------------------------

def standardise_cause(value):
    """Map ADSI cause labels to common categories."""

    cause = re.sub(r"\s+", " ", str(value).strip().lower())

    if "lightning" in cause:
        return "lightning"

    if "heat" in cause and ("stroke" in cause or "sun" in cause):
        return "heat_stroke"

    if "flood" in cause:
        return "flood"

    if "other natural" in cause:
        return "other_natural"

    return None


# ------------------------------------------------------------
# 4. IDENTIFY DEATH CSV FILES AUTOMATICALLY
# ------------------------------------------------------------

def read_csv_with_fallback(path):
    """Try common encodings used by downloaded government CSV files."""

    for encoding in ["utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read CSV encoding: {path.name}")


def identify_death_csvs():
    """
    Identify CSVs by their column structure instead of relying
    on exact filenames. Non-death CSV files are ignored.
    """

    if not RAW.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW}")

    identified = []

    for path in RAW.glob("*.csv"):
        try:
            sample = read_csv_with_fallback(path)
        except Exception as error:
            print(f"Skipping unreadable CSV {path.name}: {error}")
            continue

        columns = {str(col).strip().lower() for col in sample.columns}

        has_state = (
            "state/ut" in columns
            or "states/uts" in columns
        )

        has_year = "year" in columns
        has_total = "grand total" in columns
        has_cause = "cause" in columns or "crimehead" in columns

        if has_state and has_year and has_total and has_cause:
            identified.append(path)

    if not identified:
        raise FileNotFoundError(
            "No ADSI death CSV files were identified. "
            "Check that the original death CSV files are inside data/raw/."
        )

    print("\nIdentified death CSV files:")
    for path in identified:
        print(" -", path.name)

    return identified


# ------------------------------------------------------------
# 5. CLEAN EACH DEATH CSV
# ------------------------------------------------------------

def clean_death_file(path):

    df = read_csv_with_fallback(path)

    lower_to_original = {
        str(col).strip().lower(): col for col in df.columns
    }

    if "state/ut" in lower_to_original:
        state_col = lower_to_original["state/ut"]
    elif "states/uts" in lower_to_original:
        state_col = lower_to_original["states/uts"]
    else:
        raise ValueError(f"State column not found in {path.name}")

    if "cause" in lower_to_original:
        cause_col = lower_to_original["cause"]
    elif "crimehead" in lower_to_original:
        cause_col = lower_to_original["crimehead"]
    else:
        raise ValueError(f"Cause column not found in {path.name}")

    year_col = lower_to_original["year"]
    deaths_col = lower_to_original["grand total"]

    cleaned = df[
        [state_col, year_col, cause_col, deaths_col]
    ].copy()

    cleaned.columns = [
        "state_original",
        "year",
        "cause_original",
        "deaths"
    ]

    cleaned["state"] = cleaned["state_original"].map(standardise_state)
    cleaned["cause"] = cleaned["cause_original"].map(standardise_cause)

    cleaned["year"] = pd.to_numeric(
        cleaned["year"], errors="coerce"
    )

    cleaned["deaths"] = pd.to_numeric(
        cleaned["deaths"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    aggregate_names = {
        "INDIA",
        "TOTAL",
        "TOTAL (ALL INDIA)",
        "TOTAL ALL INDIA",
        "TOTAL (STATES)",
        "TOTAL (UTS)",
        "TOTAL STATES",
        "TOTAL UTS",
    }

    cleaned = cleaned[
        ~cleaned["state_original"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(aggregate_names)
    ]

    cleaned = cleaned.dropna(
        subset=["state", "year", "cause", "deaths"]
    )

    cleaned["year"] = cleaned["year"].astype(int)
    cleaned = cleaned[cleaned["deaths"] >= 0]

    cleaned = cleaned[
        cleaned["cause"].isin(
            ["lightning", "heat_stroke", "flood", "other_natural"]
        )
    ]

    return cleaned[
        ["state", "year", "cause", "deaths"]
    ]


# ------------------------------------------------------------
# 6. COMBINE ALL DEATH CSV FILES
# ------------------------------------------------------------

def combine_death_files():

    files = identify_death_csvs()
    frames = []

    for path in files:
        print(f"\nCleaning {path.name} ...")

        frame = clean_death_file(path)

        print("Rows retained:", len(frame))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    combined = (
        combined.groupby(
            ["state", "year", "cause"],
            as_index=False
        )["deaths"]
        .sum()
    )

    combined = combined.sort_values(
        ["state", "year", "cause"]
    )

    # Save CSV inside outputs/tables/
    output_file = TABLES / "adsi_cleaned.csv"
    combined.to_csv(output_file, index=False)

    print("\nCleaned ADSI data saved:", output_file)
    print("Total cleaned records:", len(combined))
    print("Years:", sorted(combined["year"].unique()))

    return combined


# ------------------------------------------------------------
# 7. READ STATE/UT POPULATION FROM CENSUS 2011 EXCEL
# ------------------------------------------------------------

def identify_census_workbook():

    files = list(RAW.glob("*.xlsx")) + list(RAW.glob("*.xls"))

    if len(files) == 0:
        raise FileNotFoundError(
            "Census Excel workbook not found inside data/raw/."
        )

    if len(files) > 1:

        census_named = [
            f for f in files
            if "census" in f.name.lower()
            or "population" in f.name.lower()
        ]

        if len(census_named) == 1:
            return census_named[0]

        raise ValueError(
            "More than one Excel workbook found. Keep only the intended "
            "Census population workbook in data/raw/ or adjust this function."
        )

    return files[0]


def load_census_population():

    workbook = identify_census_workbook()

    print("\nReading Census workbook:", workbook.name)

    excel = pd.ExcelFile(workbook)

    sheet = (
        "Data"
        if "Data" in excel.sheet_names
        else excel.sheet_names[0]
    )

    census = pd.read_excel(workbook, sheet_name=sheet)

    required = {"Level", "Name", "TRU", "TOT_P"}

    if not required.issubset(census.columns):
        raise ValueError(
            f"Expected Census columns {required}. "
            f"Found: {census.columns.tolist()}"
        )

    population = census[
        census["Level"].astype(str).str.strip().str.upper().eq("STATE")
        & census["TRU"].astype(str).str.strip().str.lower().eq("total")
    ].copy()

    population["state"] = population["Name"].map(standardise_state)

    population["population_2011"] = pd.to_numeric(
        population["TOT_P"], errors="coerce"
    )

    population = population[
        ["state", "population_2011"]
    ].dropna()

    population = population[
        population["population_2011"] > 0
    ]

    population = population.drop_duplicates(
        subset=["state"],
        keep="first"
    )

    # Save CSV inside outputs/tables/
    output_file = TABLES / "state_population_2011.csv"
    population.to_csv(output_file, index=False)

    print("Census population saved:", output_file)
    print("State/UT population rows:", len(population))

    return population


# ------------------------------------------------------------
# 8. CALCULATE DEATH RATES PER LAKH POPULATION
# ------------------------------------------------------------

def calculate_death_rates(deaths, population):
    """
    Calculate rate = deaths / population * 100000.

    IMPORTANT:
    This initial version uses the 2011 Census population as a fixed
    denominator for every year. It is not an annual population estimate.
    """

    wide = deaths.pivot_table(
        index=["state", "year"],
        columns="cause",
        values="deaths",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    wide.columns.name = None

    hazard_columns = [
        "lightning",
        "heat_stroke",
        "flood",
        "other_natural"
    ]

    for column in hazard_columns:
        if column not in wide.columns:
            wide[column] = 0

    wide = wide.merge(
        population,
        on="state",
        how="left",
        validate="many_to_one"
    )

    unmatched = sorted(
        wide.loc[
            wide["population_2011"].isna(), "state"
        ].unique()
    )

    if unmatched:
        print("\nWARNING: No matching 2011 Census population for:")

        for state in unmatched:
            print(" -", state)

        print(
            "These State-year rows will have missing rates until "
            "their population denominators are resolved."
        )

    for cause in hazard_columns:
        rate_column = cause + "_rate_per_lakh"

        wide[rate_column] = (
            wide[cause] / wide["population_2011"] * 100000
        )

    wide["population_year_used"] = 2011

    wide["population_method"] = (
        "Fixed 2011 Census population used for all years; "
        "preliminary approximation, not annual population estimates."
    )

    # Save CSV inside outputs/tables/
    output_file = TABLES / "state_year_death_rates_2001_2014.csv"
    wide.to_csv(output_file, index=False)

    print("\nDeath-rate dataset saved:", output_file)
    print("State-year rows:", len(wide))

    return wide


# ------------------------------------------------------------
# 9. MAIN: PREPROCESSING ONLY
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("DISASTER MORTALITY DATA PREPROCESSING")
    print("=" * 60)

    deaths = combine_death_files()
    population = load_census_population()

    rates = calculate_death_rates(
        deaths,
        population
    )

    print("\nPREPROCESSING COMPLETED.")

    print("\nFiles created inside outputs/tables/:")
    print("1. adsi_cleaned.csv")
    print("2. state_population_2011.csv")
    print("3. state_year_death_rates_2001_2014.csv")

    print(
        "\nNext step: inspect unmatched State names and population "
        "coverage before running 02_clustering.py."
    )


if __name__ == "__main__":
    main()