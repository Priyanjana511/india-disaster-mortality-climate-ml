"""
temperature_analysis.py
Converts IMD 1-degree daily Tmax GRD files into State-wise summer temperatures.

IMD specifications:
- 31 latitudes: 7.5 to 37.5 N, 1-degree interval
- 31 longitudes: 67.5 to 97.5 E, 1-degree interval
- daily float32 binary values, latitude then longitude
- 99.9 is the missing-value sentinel
- 365/366 daily grids per year

Season used here: March-May (pre-monsoon/summer).
Run from project root: python src/temperature_analysis.py
"""

from pathlib import Path
import calendar
import re
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BOUNDARIES = ROOT / "data" / "boundaries"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

START_YEAR, END_YEAR = 2001, 2025
MISSING_VALUE = 99.9
N_LAT = N_LON = 31
LATS = np.arange(7.5, 38.0, 1.0)
LONS = np.arange(67.5, 98.0, 1.0)
SUMMER_MONTHS = [3, 4, 5]


def find_year_file(year):
    candidates = list(RAW.glob(f"Maxtemp_MaxT_{year}*.GRD"))
    if not candidates:
        candidates = list(RAW.glob(f"Maxtemp_MaxT_{year}*.grd"))
    if not candidates:
        raise FileNotFoundError(f"No Maxtemp_MaxT file found for {year} in {RAW}")
    # Multiple copies may have been uploaded; require same binary size/content.
    candidates = sorted(candidates, key=lambda p: (len(p.name), p.name))
    sizes = {p.stat().st_size for p in candidates}
    if len(sizes) > 1:
        raise ValueError(f"Conflicting GRD file sizes for {year}: {candidates}")
    return candidates[0]


def load_year(year, path):
    expected_days = 366 if calendar.isleap(year) else 365
    expected_values = expected_days * N_LAT * N_LON
    expected_bytes = expected_values * 4

    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path.name}: file size {path.stat().st_size:,} bytes; "
            f"expected {expected_bytes:,} for {expected_days} x 31 x 31 "
            "float32 values. Check that this is the IMD 1-degree annual Tmax file."
        )

    # Official IMD GRD is raw float32; values in supplied files show 99.9 sentinel.
    arr = np.fromfile(path, dtype="<f4")
    arr = arr.reshape(expected_days, N_LAT, N_LON)
    arr[arr == MISSING_VALUE] = np.nan

    # The IMD grid control definition has X=longitude and Y=latitude,
    # both increasing from the listed starting coordinate.
    dates = pd.date_range(f"{year}-01-01", periods=expected_days, freq="D")
    return arr, dates


def load_boundaries():
    shp_files = list(BOUNDARIES.glob("STATE_BOUNDARY.shp"))
    if not shp_files:
        raise FileNotFoundError(
            f"STATE_BOUNDARY.shp not found in {BOUNDARIES}. "
            "Copy the .shp, .shx, .dbf, .prj and .cpg files there."
        )

    states = gpd.read_file(shp_files[0])
    if "STATE" not in states.columns:
        raise ValueError(f"Expected STATE field in shapefile; found {list(states.columns)}")

    states = states[states["STATE"].notna()].copy()
    states["STATE"] = states["STATE"].astype(str).str.strip().str.upper()
    # These are map annotation/disputed polygons, not normal State/UT units.
    states = states[~states["STATE"].str.startswith("DISPUTED")]
    states = states[~states["STATE"].str.contains("INDIA", na=False)]

    if states.crs is None:
        raise ValueError("State shapefile has no CRS; define its correct CRS before continuing.")
    return states.to_crs("EPSG:4326")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Reading State/UT boundaries...")
    states = load_boundaries()

    lon_grid, lat_grid = np.meshgrid(LONS, LATS)
    points = gpd.GeoDataFrame(
        {"grid_id": np.arange(N_LAT * N_LON),
         "latitude": lat_grid.ravel(),
         "longitude": lon_grid.ravel()},
        geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
        crs="EPSG:4326",
    )

    # Assign each grid-cell centre to the State polygon containing it.
    joined = gpd.sjoin(
        points, states[["STATE", "geometry"]],
        how="left", predicate="within"
    )
    if joined["STATE"].isna().any():
        print(
            "Warning:", int(joined["STATE"].isna().sum()),
            "grid centres are outside the supplied State polygons."
        )

    grid_to_state = joined.dropna(subset=["STATE"])[["grid_id", "STATE"]]
    if grid_to_state.empty:
        raise ValueError("No IMD grid points matched the State boundary shapefile.")

    print("Calculating March-May mean of daily Tmax for every year...")
    yearly_rows = []
    grid_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        file_path = find_year_file(year)
        print(f"  {year}: {file_path.name}")
        arr, dates = load_year(year, file_path)

        seasonal_mask = dates.month.isin(SUMMER_MONTHS)
        seasonal = arr[seasonal_mask, :, :]
        # Average daily values by grid cell over March-May.
        cell_means = np.nanmean(seasonal, axis=0).ravel()

        cell_df = pd.DataFrame({
            "grid_id": np.arange(N_LAT * N_LON),
            "summer_tmax_c": cell_means,
        }).merge(grid_to_state, on="grid_id", how="inner")

        # State-level mean of grid-cell seasonal means. Each grid cell is
        # equally weighted; this is not an area-weighted population exposure.
        state_means = (
            cell_df.groupby("STATE", as_index=False)["summer_tmax_c"]
            .mean()
        )
        state_means["year"] = year
        state_means = state_means.rename(columns={"STATE": "state"})
        yearly_rows.append(state_means)

        valid = np.isfinite(cell_means)
        for gid in np.where(valid)[0]:
            grid_rows.append({
                "year": year,
                "grid_id": int(gid),
                "latitude": float(lat_grid.ravel()[gid]),
                "longitude": float(lon_grid.ravel()[gid]),
                "summer_tmax_c": float(cell_means[gid]),
            })

    result = pd.concat(yearly_rows, ignore_index=True)
    result["state"] = result["state"].replace({
        "ORISSA": "ODISHA",
        "UTTARANCHAL": "UTTARAKHAND",
        "PONDICHERRY": "PUDUCHERRY",
        "NCT OF DELHI": "DELHI",
    })
    result.to_csv(OUT / "state_temperature_summer_2001_2025.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT / "grid_summer_temperature_2001_2025.csv", index=False)

    # National average across State-level values, descriptive only.
    national = result.groupby("year", as_index=False)["summer_tmax_c"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(national["year"], national["summer_tmax_c"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean March-May maximum temperature (°C)")
    plt.title("India: State-averaged summer maximum temperature, 2001-2025")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "temperature_trend_2001_2025.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print("outputs/state_temperature_summer_2001_2025.csv")
    print("outputs/grid_summer_temperature_2001_2025.csv")
    print("outputs/figures/temperature_trend_2001_2025.png")
    print("\nMethod note: grid-cell centres are assigned to States; grid cells")
    print("are equally weighted. This is not area- or population-weighted exposure.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
