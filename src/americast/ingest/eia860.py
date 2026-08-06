"""EIA-860 ingestion: the California utility-scale solar plant registry.

EIA-860 is the federal annual census of US generators (>= 1 MW): location,
capacity, status, and for solar, tracking type. Published as Excel
workbooks, refreshed once a year — so this ingest is "download, build,
done" rather than a daily feed. We use the 2025 Early Release
(published 2026-06-09; final expected September 2026).

The Solar schedule is generator-level (large plants file several phases);
build_registry aggregates to one row per plant and joins coordinates,
county, and balancing authority from the Plant schedule.
"""

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from americast.schemas import PLANTS_CA

EIA860_URL = "https://www.eia.gov/electricity/data/eia860/xls/eia8602025ER.zip"
RAW_DIR = Path("data/eia860")
REGISTRY_PATH = Path("data/registry/plants_ca.parquet")

_PLANT_XLSX = "2___Plant_Y2025_Early_Release.xlsx"
_SOLAR_XLSX = "3_3_Solar_Y2025_Early_Release.xlsx"


def download_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Download and extract the EIA-860 zip; skips work already done."""
    extracted = raw_dir / "extracted"
    if (extracted / _PLANT_XLSX).exists():
        return extracted
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / Path(EIA860_URL).name
    if not zip_path.exists():
        urllib.request.urlretrieve(EIA860_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted)
    return extracted


def load_sheets(extracted: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Plant and Solar schedules (headers live on sheet row 3)."""
    plant = pd.read_excel(extracted / _PLANT_XLSX, skiprows=2)
    solar = pd.read_excel(extracted / _SOLAR_XLSX, skiprows=2)
    return plant, solar


def build_registry(plant: pd.DataFrame, solar: pd.DataFrame) -> pd.DataFrame:
    """One row per operating CA utility-scale solar PV plant.

    Filters the generator-level Solar schedule (state CA, status OP,
    technology Solar Photovoltaic), sums capacity per plant, picks the
    capacity-dominant tracking type, then joins location facts from the
    Plant schedule.
    """
    gens = solar[
        (solar["State"] == "CA")
        & (solar["Status"] == "OP")
        & (solar["Technology"] == "Solar Photovoltaic")
    ].copy()
    gens["tracking"] = _tracking_type(gens)

    per_plant = (
        gens.groupby("Plant Code")
        .agg(capacity_mw_ac=("Nameplate Capacity (MW)", "sum"))
        .reset_index()
        .merge(_dominant_tracking(gens), on="Plant Code")
    )

    location = plant[
        [
            "Plant Code",
            "Plant Name",
            "Latitude",
            "Longitude",
            "County",
            "Balancing Authority Code",
        ]
    ]
    merged = per_plant.merge(location, on="Plant Code", how="left")

    out = pd.DataFrame(
        {
            "plant_id": merged["Plant Code"].astype("int64"),
            "plant_name": merged["Plant Name"].astype("str"),
            "latitude": pd.to_numeric(merged["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(merged["Longitude"], errors="coerce"),
            "capacity_mw_ac": merged["capacity_mw_ac"].astype("float64"),
            "tracking": merged["tracking"],
            "county": merged["County"].fillna("UNKNOWN").astype("str"),
            "balancing_authority": merged["Balancing Authority Code"]
            .fillna("UNKNOWN")
            .astype("str"),
        }
    )
    return out.sort_values("plant_id").reset_index(drop=True)


def _tracking_type(gens: pd.DataFrame) -> pd.Series:
    """Per-generator tracking from EIA's Y/N flag columns."""
    tracking = pd.Series("unknown", index=gens.index, dtype="str")
    tracking[gens["Fixed Tilt?"] == "Y"] = "fixed"
    tracking[gens["Dual-Axis Tracking?"] == "Y"] = "dual_axis"
    tracking[gens["Single-Axis Tracking?"] == "Y"] = "single_axis"
    return tracking


def _dominant_tracking(gens: pd.DataFrame) -> pd.DataFrame:
    """Capacity-dominant tracking type per plant (ties: first alphabetical)."""
    cap = (
        gens.groupby(["Plant Code", "tracking"])["Nameplate Capacity (MW)"]
        .sum()
        .reset_index()
        .sort_values(
            ["Plant Code", "Nameplate Capacity (MW)", "tracking"],
            ascending=[True, False, True],
        )
    )
    return cap.drop_duplicates("Plant Code")[["Plant Code", "tracking"]]


def write_registry(df: pd.DataFrame, path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=PLANTS_CA, preserve_index=False)
    pq.write_table(table, path)


if __name__ == "__main__":
    plant, solar = load_sheets(download_raw())
    registry = build_registry(plant, solar)
    dropped = registry[registry[["latitude", "longitude"]].isna().any(axis=1)]
    if not dropped.empty:
        print(f"dropping {len(dropped)} plants with missing coordinates:")
        print(dropped[["plant_id", "plant_name", "capacity_mw_ac"]].to_string())
        registry = registry.dropna(subset=["latitude", "longitude"])
    write_registry(registry.reset_index(drop=True))
    print(
        f"registry written: {len(registry)} plants, "
        f"{registry['capacity_mw_ac'].sum() / 1000:.2f} GW AC → {REGISTRY_PATH}"
    )
