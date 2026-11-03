
"""
temperature_analysis.py

PROJECT:
Deaths from lightning, heat stroke and floods:
A State typology for early-warning priorities.

PURPOSE:
Converts IMD 1-degree daily Tmax GRD files
into State-wise summer temperatures.

IMD GRID ASSUMPTIONS:
- 31 latitudes: 7.5 to 37.5 N
- 31 longitudes: 67.5 to 97.5 E
- 1-degree interval
- Daily float32 binary values
- 99.9 is the missing-value sentinel
- 365/366 daily grids per year

SEASON:
March-May (pre-monsoon/summer).

RUN:
python src/temperature_analysis.py
"""

from pathlib import Path
import calendar
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
BOUNDARIES = ROOT / "data" / "boundaries"

OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

START_YEAR = 2001
END_YEAR = 2025

MISSING_VALUE = 99.9

N_LAT = 31
N_LON = 31

LATS = np.arange(7.5, 38.0, 1.0)
LONS = np.arange(67.5, 98.0, 1.0)

SUMMER_MONTHS = [3, 4, 5]

# Create output folders
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 2. FIND YEARLY IMD GRD FILE
# ----------------------------------------------------------

def find_year_file(year):

    candidates = list(
        RAW.glob(f"Maxtemp_MaxT_{year}*.GRD")
    )

    if not candidates:
        candidates = list(
            RAW.glob(f"Maxtemp_MaxT_{year}*.grd")
        )

    if not candidates:
        raise FileNotFoundError(
            f"No Maxtemp_MaxT file found for {year} "
            f"in {RAW}"
        )

    # Sort possible duplicate files
    candidates = sorted(
        candidates,
        key=lambda p: (len(p.name), p.name)
    )

    # Check file sizes
    sizes = {
        p.stat().st_size
        for p in candidates
    }

    if len(sizes) > 1:
        raise ValueError(
            f"Conflicting GRD file sizes for {year}: "
            f"{candidates}"
        )

    return candidates[0]


# ----------------------------------------------------------
# 3. LOAD DAILY TEMPERATURE GRID
# ----------------------------------------------------------

def load_year(year, path):

    expected_days = (
        366 if calendar.isleap(year) else 365
    )

    expected_values = (
        expected_days * N_LAT * N_LON
    )

    expected_bytes = expected_values * 4

    if path.stat().st_size != expected_bytes:

        raise ValueError(
            f"{path.name}: file size "
            f"{path.stat().st_size:,} bytes; "
            f"expected {expected_bytes:,} bytes "
            f"for {expected_days} x 31 x 31 float32 values. "
            "Check the IMD GRD file format."
        )

    # Read little-endian float32 binary data
    arr = np.fromfile(
        path,
        dtype="<f4"
    )

    # Reshape into daily grids
    arr = arr.reshape(
        expected_days,
        N_LAT,
        N_LON
    )

    # Replace missing-value sentinel
    arr[arr == MISSING_VALUE] = np.nan

    # Generate daily dates
    dates = pd.date_range(
        f"{year}-01-01",
        periods=expected_days,
        freq="D"
    )

    return arr, dates


# ----------------------------------------------------------
# 4. LOAD STATE BOUNDARIES
# ----------------------------------------------------------

def load_boundaries():

    shp_files = list(
        BOUNDARIES.glob("STATE_BOUNDARY.shp")
    )

    if not shp_files:

        raise FileNotFoundError(
            f"STATE_BOUNDARY.shp not found in {BOUNDARIES}.\n"
            "Copy the .shp, .shx, .dbf, .prj and .cpg "
            "files into the boundaries folder."
        )

    states = gpd.read_file(
        shp_files[0]
    )

    if "STATE" not in states.columns:

        raise ValueError(
            "Expected STATE field in shapefile.\n"
            f"Available fields: {list(states.columns)}"
        )

    states = states[
        states["STATE"].notna()
    ].copy()

    states["STATE"] = (
        states["STATE"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Remove disputed annotation polygons
    states = states[
        ~states["STATE"].str.startswith("DISPUTED")
    ]

    states = states[
        ~states["STATE"].str.contains(
            "INDIA",
            na=False
        )
    ]

    if states.crs is None:

        raise ValueError(
            "State shapefile has no CRS. "
            "Define its correct CRS before continuing."
        )

    # Convert boundaries to latitude-longitude CRS
    states = states.to_crs("EPSG:4326")

    return states


# ----------------------------------------------------------
# 5. MAIN TEMPERATURE ANALYSIS
# ----------------------------------------------------------

def main():

    print("=" * 65)
    print("IMD TEMPERATURE DATA ANALYSIS")
    print("=" * 65)

    print("\nReading State/UT boundaries...")

    states = load_boundaries()

    print("State boundaries loaded successfully.")

    # Create latitude-longitude grid
    lon_grid, lat_grid = np.meshgrid(
        LONS,
        LATS
    )

    # Create grid centre points
    points = gpd.GeoDataFrame(
        {
            "grid_id": np.arange(
                N_LAT * N_LON
            ),
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel()
        },

        geometry=gpd.points_from_xy(
            lon_grid.ravel(),
            lat_grid.ravel()
        ),

        crs="EPSG:4326"
    )

    # ------------------------------------------------------
    # 5.1 ASSIGN GRID POINTS TO STATES
    # ------------------------------------------------------

    print("\nAssigning IMD grid points to States...")

    joined = gpd.sjoin(
        points,
        states[["STATE", "geometry"]],
        how="left",
        predicate="within"
    )

    unmatched = int(
        joined["STATE"].isna().sum()
    )

    if unmatched > 0:

        print(
            "Warning:",
            unmatched,
            "grid centres are outside the supplied State polygons."
        )

    grid_to_state = joined.dropna(
        subset=["STATE"]
    )[["grid_id", "STATE"]]

    # Check whether grid points were assigned
    if grid_to_state.empty:

        raise ValueError(
            "No IMD grid points matched the State boundary shapefile."
        )

    # Check for grid points assigned to multiple States
    duplicate_grid_ids = grid_to_state[
        "grid_id"
    ].duplicated()

    if duplicate_grid_ids.any():

        raise ValueError(
            "Some grid centres matched multiple State polygons. "
            "Check the State boundary shapefile for overlapping polygons."
        )

    print(
        "Matched grid centres:",
        len(grid_to_state)
    )

    # ------------------------------------------------------
    # 5.2 CALCULATE YEARLY STATE TEMPERATURES
    # ------------------------------------------------------

    print(
        "\nCalculating March-May mean daily Tmax "
        "for every year..."
    )

    yearly_rows = []
    grid_rows = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        file_path = find_year_file(year)

        print(
            f"\nProcessing {year}: {file_path.name}"
        )

        # Load yearly daily temperature data
        arr, dates = load_year(
            year,
            file_path
        )

        # Select March, April and May
        seasonal_mask = dates.month.isin(
            SUMMER_MONTHS
        )

        seasonal = arr[
            seasonal_mask,
            :,
            :
        ]

        # Calculate mean seasonal Tmax for each grid cell
        with np.errstate(
            invalid="ignore"
        ):

            cell_means = np.nanmean(
                seasonal,
                axis=0
            ).ravel()

        # Create grid-level dataframe
        cell_df = pd.DataFrame(
            {
                "grid_id": np.arange(
                    N_LAT * N_LON
                ),
                "summer_tmax_c": cell_means
            }
        )

        # Assign grid cells to States
        cell_df = cell_df.merge(
            grid_to_state,
            on="grid_id",
            how="inner"
        )

        # Remove grid cells with no valid temperature
        cell_df = cell_df.dropna(
            subset=["summer_tmax_c"]
        )

        # Calculate State-level mean temperature
        state_means = (
            cell_df.groupby(
                "STATE",
                as_index=False
            )["summer_tmax_c"]
            .mean()
        )

        state_means["year"] = year

        state_means = state_means.rename(
            columns={
                "STATE": "state"
            }
        )

        yearly_rows.append(
            state_means
        )

        # Save individual grid-cell results
        valid = np.isfinite(
            cell_means
        )

        for gid in np.where(valid)[0]:

            grid_rows.append(
                {
                    "year": year,
                    "grid_id": int(gid),
                    "latitude": float(
                        lat_grid.ravel()[gid]
                    ),
                    "longitude": float(
                        lon_grid.ravel()[gid]
                    ),
                    "summer_tmax_c": float(
                        cell_means[gid]
                    )
                }
            )

        print(
            "States with valid temperature data:",
            len(state_means)
        )

    # ------------------------------------------------------
    # 5.3 COMBINE YEARLY RESULTS
    # ------------------------------------------------------

    print("\nCombining yearly temperature results...")

    if not yearly_rows:

        raise ValueError(
            "No yearly temperature results were generated."
        )

    result = pd.concat(
        yearly_rows,
        ignore_index=True
    )

    # Standardise common State names
    result["state"] = result["state"].replace(
        {
            "ORISSA": "ODISHA",
            "UTTARANCHAL": "UTTARAKHAND",
            "PONDICHERRY": "PUDUCHERRY",
            "NCT OF DELHI": "DELHI"
        }
    )

    result = result.sort_values(
        ["year", "state"]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # 5.4 SAVE STATE-LEVEL TEMPERATURE CSV
    # ------------------------------------------------------

    state_output = (
        TABLES /
        "state_temperature_summer_2001_2025.csv"
    )

    result.to_csv(
        state_output,
        index=False
    )

    # ------------------------------------------------------
    # 5.5 SAVE GRID-LEVEL TEMPERATURE CSV
    # ------------------------------------------------------

    grid_output = (
        TABLES /
        "grid_summer_temperature_2001_2025.csv"
    )

    grid_df = pd.DataFrame(
        grid_rows,
        columns=[
            "year",
            "grid_id",
            "latitude",
            "longitude",
            "summer_tmax_c"
        ]
    )

    grid_df.to_csv(
        grid_output,
        index=False
    )

    # ------------------------------------------------------
    # 6. NATIONAL TEMPERATURE TREND
    # ------------------------------------------------------

    print("\nCreating national temperature trend graph...")

    # Descriptive average across available State-level values
    national = (
        result.groupby(
            "year",
            as_index=False
        )["summer_tmax_c"]
        .mean()
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        national["year"],
        national["summer_tmax_c"],
        marker="o"
    )

    plt.xlabel("Year")

    plt.ylabel(
        "Mean March-May maximum temperature (°C)"
    )

    plt.title(
        "India: State-averaged Summer Maximum Temperature (2001-2025)"
    )

    plt.grid(
        alpha=0.3
    )

    plt.xticks(
        range(START_YEAR, END_YEAR + 1, 2),
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "temperature_trend_2001_2025.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------------------
    # 7. DISPLAY FINAL RESULTS
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("TEMPERATURE ANALYSIS COMPLETED!")
    print("=" * 65)

    print("\nState-level temperature CSV:")
    print(state_output)

    print("\nGrid-level temperature CSV:")
    print(grid_output)

    print("\nTemperature trend graph:")
    print(
        FIGURES /
        "temperature_trend_2001_2025.png"
    )

    print("\nTotal State-year records:", len(result))

    print(
        "Number of States/UTs:",
        result["state"].nunique()
    )

    print(
        "Year range:",
        result["year"].min(),
        "-",
        result["year"].max()
    )

    print("\nMETHOD NOTE:")
    print(
        "Grid-cell centres are assigned to State polygons."
    )

    print(
        "Grid cells are equally weighted within each State."
    )

    print(
        "This is not area-weighted or population-weighted exposure."
    )

    print(
        "\nIMPORTANT: Verify the IMD binary format, "
        "byte order, grid orientation and missing-value "
        "sentinel against the official metadata."
    )


# ----------------------------------------------------------
# 8. EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"\nERROR: {exc}"
        )

        sys.exit(1)
