
# ==========================================================
# PROJECT: Deaths from lightning, heat stroke and floods
# FILE: 03_regression.py
# PURPOSE: Temperature and Heat-Stroke Mortality Regression
#
# Predictor: Summer maximum temperature
# Target: Heat-stroke deaths per lakh population
#
# Study period: 2001-2014
# Training: 2001-2011
# Testing: 2012-2014
# ==========================================================

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ----------------------------------------------------------
# 1. PATHS AND SETTINGS
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

MORTALITY_FILE = (
    TABLES / "state_year_death_rates_2001_2014.csv"
)

TEMPERATURE_FILE = (
    TABLES / "state_temperature_summer_2001_2025.csv"
)

TARGET = "heat_stroke_rate_per_lakh"
PREDICTOR = "summer_tmax_c"

FIRST_YEAR = 2001
LAST_YEAR = 2014

TRAIN_LAST_YEAR = 2011
TEST_FIRST_YEAR = 2012

OUTPUTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 2. STANDARDISE STATE NAMES
# ----------------------------------------------------------

def standardise_state(name):

    name = str(name).strip().upper()
    name = re.sub(r"\s+", " ", name)

    aliases = {
        "ORISSA": "ODISHA",
        "UTTARANCHAL": "UTTARAKHAND",
        "NCT OF DELHI": "DELHI",
        "DELHI UT": "DELHI",
        "PONDICHERRY": "PUDUCHERRY",
        "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
        "JAMMU-KASHMIR": "JAMMU AND KASHMIR"
    }

    return aliases.get(name, name)


# ----------------------------------------------------------
# 3. LOAD DATA
# ----------------------------------------------------------

def read_required_csv(path, required_columns, description):

    if not path.exists():
        raise FileNotFoundError(
            f"{description} file not found: {path}"
        )

    df = pd.read_csv(path)

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{description} CSV is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def load_and_prepare_data():

    print("\nSTEP 1: Loading datasets")

    deaths = read_required_csv(
        MORTALITY_FILE,
        ["state", "year", TARGET],
        "Mortality"
    )

    temperature = read_required_csv(
        TEMPERATURE_FILE,
        ["state", "year", PREDICTOR],
        "Temperature"
    )

    for df in [deaths, temperature]:

        df["state"] = df["state"].apply(
            standardise_state
        )

        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce"
        )

    deaths[TARGET] = pd.to_numeric(
        deaths[TARGET],
        errors="coerce"
    )

    temperature[PREDICTOR] = pd.to_numeric(
        temperature[PREDICTOR],
        errors="coerce"
    )

    deaths = deaths.dropna(
        subset=["state", "year", TARGET]
    )

    temperature = temperature.dropna(
        subset=["state", "year", PREDICTOR]
    )

    deaths["year"] = deaths["year"].astype(int)
    temperature["year"] = temperature["year"].astype(int)

    # Check duplicate State-year records
    for df, label in [
        (deaths, "mortality"),
        (temperature, "temperature")
    ]:

        duplicates = df.duplicated(
            ["state", "year"],
            keep=False
        )

        if duplicates.any():

            examples = df.loc[
                duplicates,
                ["state", "year"]
            ].head(10)

            raise ValueError(
                f"Duplicate State-year records in {label} data:\n"
                f"{examples.to_string(index=False)}"
            )

    # Keep overlapping years only
    deaths = deaths[
        deaths["year"].between(
            FIRST_YEAR,
            LAST_YEAR
        )
    ]

    temperature = temperature[
        temperature["year"].between(
            FIRST_YEAR,
            LAST_YEAR
        )
    ]

    # Merge both datasets
    data = pd.merge(
        deaths,
        temperature,
        on=["state", "year"],
        how="inner",
        validate="one_to_one"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=[TARGET, PREDICTOR]
    )

    data = data[
        (data[TARGET] >= 0) &
        (data[PREDICTOR] < 99.9)
    ]

    if data.empty:
        raise ValueError(
            "No valid matched State-year records. "
            "Check State names, years and input columns."
        )

    data = data.sort_values(
        ["year", "state"]
    ).reset_index(drop=True)

    # Save merged regression dataset
    data.to_csv(
        TABLES / "regression_dataset_2001_2014.csv",
        index=False
    )

    print("\nMatched observations:", len(data))
    print("Number of States/UTs:", data["state"].nunique())

    print(
        "Years:",
        data["year"].min(),
        "-",
        data["year"].max()
    )

    print("\nMerged dataset preview:")
    print(data.head())

    return data


# ----------------------------------------------------------
# 4. TRAIN AND TEST LINEAR REGRESSION
# ----------------------------------------------------------

def train_and_evaluate(data):

    print("\nSTEP 2: Splitting data")

    train = data[
        data["year"] <= TRAIN_LAST_YEAR
    ].copy()

    test = data[
        data["year"] >= TEST_FIRST_YEAR
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or testing dataset is empty. "
            "Check the available years."
        )

    X_train = train[[PREDICTOR]]
    y_train = train[TARGET]

    X_test = test[[PREDICTOR]]
    y_test = test[TARGET]

    print("Training observations:", len(train))
    print("Testing observations:", len(test))

    print("\nSTEP 3: Training model")

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predicted = model.predict(X_test)

    # Training mean baseline
    baseline = np.full(
        len(y_test),
        y_train.mean()
    )

    def get_metrics(actual, prediction):

        return {
            "MAE": mean_absolute_error(
                actual,
                prediction
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    actual,
                    prediction
                )
            ),

            "R2": (
                r2_score(actual, prediction)
                if len(actual) >= 2
                else np.nan
            )
        }

    model_scores = get_metrics(
        y_test,
        predicted
    )

    baseline_scores = get_metrics(
        y_test,
        baseline
    )

    results = pd.DataFrame([
        {
            "Model": "Linear Regression",
            **model_scores
        },
        {
            "Model": "Training Mean Baseline",
            **baseline_scores
        }
    ])

    # Save regression metrics
    results.to_csv(
        TABLES / "regression_metrics.csv",
        index=False
    )

    # Save predictions
    predictions = test[
        ["state", "year"]
    ].copy()

    predictions["actual_heat_stroke_rate"] = (
        y_test.to_numpy()
    )

    predictions["predicted_heat_stroke_rate"] = (
        predicted
    )

    predictions["residual"] = (
        predictions["actual_heat_stroke_rate"]
        - predictions["predicted_heat_stroke_rate"]
    )

    predictions.to_csv(
        TABLES / "regression_test_predictions.csv",
        index=False
    )

    print("\nRegression equation:")

    print(
        f"Y = {model.intercept_:.5f} "
        f"+ ({model.coef_[0]:.5f} * X)"
    )

    print("\nEvaluation results:")
    print(results.to_string(index=False))

    return model, test, predicted


# ----------------------------------------------------------
# 5. CREATE GRAPHS
# ----------------------------------------------------------

def create_graphs(data, model, test, predicted):

    print("\nSTEP 4: Creating graphs")

    # ------------------------------------------------------
    # GRAPH 1: Temperature vs Heat-Stroke Mortality
    # ------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.scatter(
        data[PREDICTOR],
        data[TARGET],
        alpha=0.65,
        label="State-year observations"
    )

    x_values = np.linspace(
        data[PREDICTOR].min(),
        data[PREDICTOR].max(),
        200
    )

    y_values = model.predict(
        pd.DataFrame({PREDICTOR: x_values})
    )

    plt.plot(
        x_values,
        y_values,
        label="Fitted regression line"
    )

    plt.xlabel(
        "Summer mean maximum temperature (°C)"
    )

    plt.ylabel(
        "Heat-stroke deaths per lakh population"
    )

    plt.title(
        "Temperature vs Heat-stroke Mortality (2001-2014)"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / "07_temperature_vs_heatstroke.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------------------
    # GRAPH 2: Actual vs Predicted
    # ------------------------------------------------------

    actual = test[TARGET].to_numpy()

    plt.figure(figsize=(7, 6))

    plt.scatter(
        actual,
        predicted,
        alpha=0.7
    )

    lower = min(
        float(actual.min()),
        float(predicted.min())
    )

    upper = max(
        float(actual.max()),
        float(predicted.max())
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    plt.xlabel(
        "Actual heat-stroke deaths per lakh"
    )

    plt.ylabel(
        "Predicted heat-stroke deaths per lakh"
    )

    plt.title(
        "Actual vs Predicted Mortality (Test: 2012-2014)"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / "08_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------------------
    # GRAPH 3: Residual Plot
    # ------------------------------------------------------

    residuals = actual - predicted

    plt.figure(figsize=(8, 5))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted heat-stroke deaths per lakh"
    )

    plt.ylabel(
        "Residual (actual - predicted)"
    )

    plt.title(
        "Regression Residual Plot (Test: 2012-2014)"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / "09_regression_residuals.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("All three graphs saved successfully.")


# ----------------------------------------------------------
# 6. MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("TEMPERATURE AND HEAT-STROKE REGRESSION")
    print("=" * 60)

    # Step 1: Load and prepare data
    data = load_and_prepare_data()

    # Step 2: Train and evaluate regression
    model, test, predicted = train_and_evaluate(data)

    # Step 3: Create graphs
    create_graphs(
        data,
        model,
        test,
        predicted
    )

    # Step 4: Display research limitations
    print("\nRESEARCH LIMITATIONS:")

    print(
        "1. Association does not establish causation."
    )

    print(
        "2. Fixed 2011 Census population denominators are approximate."
    )

    print(
        "3. Other factors may affect heat-stroke mortality."
    )

    print(
        "4. Repeated State-year observations are not independent."
    )

    print(
        "5. Compare results with the training-mean baseline."
    )

    print("\nREGRESSION TASK COMPLETED!")

    print("\nCSV files saved inside outputs/tables/")
    print("Graphs saved inside outputs/figures/")


# ----------------------------------------------------------
# 7. EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print("\nERROR:", exc)
        sys.exit(1)
