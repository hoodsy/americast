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

import os
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

# Stand-ins for the few generators EIA left blank. Every one of them is
# measured from the California operating fleet in this same vintage, so
# a filled value sits where the reported ones already are rather than
# importing a national rule of thumb.
#
# ILR is the capacity-weighted mean of the 1,099 generators that do
# report a DC rating. Only 0.06% of state capacity needs it.
FALLBACK_ILR = 1.275
# Capacity-weighted mean tilt of California's fixed-tilt fleet. Higher
# than the plain median of 15 degrees because the larger plants tilt
# more steeply, and it is capacity that the aggregation weights.
FALLBACK_FIXED_TILT = 22.0
# A tracker's axis lies flat. Also used for dual-axis and unknown,
# where the column is not read anyway.
FALLBACK_AXIS_TILT = 0.0
# Due south, the reported value for most of the fleet.
FALLBACK_AZIMUTH = 180.0


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
    technology Solar Photovoltaic), sums AC and DC capacity per plant,
    picks the capacity-dominant tracking type and the array geometry
    that goes with it, dates the plant from its first phase, then joins
    location facts from the Plant schedule.
    """
    gens = solar[
        (solar["State"] == "CA")
        & (solar["Status"] == "OP")
        & (solar["Technology"] == "Solar Photovoltaic")
    ].copy()
    gens["tracking"] = _tracking_type(gens)
    gens["dc_mw"] = _dc_capacity(gens)

    grouped = gens.groupby("Plant Code")
    totals = grouped.agg(
        capacity_mw_ac=("Nameplate Capacity (MW)", "sum"),
        dc_capacity_mw=("dc_mw", "sum"),
    ).reset_index()
    layout = _layout(gens)
    online = _first_online(gens)
    per_plant = totals.merge(layout, on="Plant Code").merge(online, on="Plant Code")

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
            "dc_capacity_mw": merged["dc_capacity_mw"].astype("float64"),
            "tracking": merged["tracking"],
            "tilt": _array_tilt(merged),
            "azimuth": merged["azimuth"].fillna(FALLBACK_AZIMUTH).astype("float64"),
            "county": merged["County"].fillna("UNKNOWN").astype("str"),
            "balancing_authority": merged["Balancing Authority Code"]
            .fillna("UNKNOWN")
            .astype("str"),
            "operating_date": merged["operating_date"],
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


def _dc_capacity(gens: pd.DataFrame) -> pd.Series:
    """DC nameplate per generator, estimated where EIA left it blank.

    The DC side is what the panels produce and the AC side is what the
    inverters can pass, so the model needs both to know when a plant
    clips. Five of 1,104 California generators omit it, holding 0.06%
    of state capacity — small enough that the fallback is a rounding
    error rather than a modelling choice.
    """
    reported = pd.to_numeric(gens["DC Net Capacity (MW)"], errors="coerce")
    ac = pd.to_numeric(gens["Nameplate Capacity (MW)"], errors="coerce")
    return reported.fillna(ac * FALLBACK_ILR)


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


def _layout(gens: pd.DataFrame) -> pd.DataFrame:
    """Tracking, tilt and azimuth per plant, read off one real generator.

    The largest generator of the dominant tracking type supplies the
    geometry, rather than a mean across the plant's phases. Two reasons.
    Azimuth is a compass bearing, and a plant mixing generators recorded
    at 0 and 180 — the two ways to write one north-south tracker axis —
    would average to 90 and claim an east-west axis that exists nowhere
    on site. And tilt means different things per tracking type, so
    averaging a fixed phase with a tracker phase mixes a panel angle
    into an axis angle.

    Reading from a generator that actually matches the plant's dominant
    tracking keeps the three fields describing one real array.
    """
    tracking = _dominant_tracking(gens)
    matched = gens.merge(tracking, on=["Plant Code", "tracking"], how="inner")
    ordered = matched.sort_values(
        ["Plant Code", "Nameplate Capacity (MW)"], ascending=[True, False]
    )
    biggest = ordered.drop_duplicates("Plant Code").copy()
    biggest["tilt"] = pd.to_numeric(biggest["Tilt Angle"], errors="coerce")
    biggest["azimuth"] = pd.to_numeric(biggest["Azimuth Angle"], errors="coerce")
    return biggest[["Plant Code", "tracking", "tilt", "azimuth"]]


def _first_online(gens: pd.DataFrame) -> pd.DataFrame:
    """The month a plant's first phase started generating, in UTC.

    Only 79.7% of today's California solar capacity was running at the
    end of 2023, so weighting a 2023 forecast hour with today's fleet
    would invent a fifth of the state's panels. The date lets the
    aggregation drop plants that did not exist yet.

    One date per plant, not per phase. A plant whose second phase
    arrived later therefore counts at full size from its first — 27
    plants holding 3.74 GW are built in stages, but only 8 of them
    (0.04 GW) stagger by more than two years, so the residual is small.
    Generator-level dating is the fix if Gate 5 ever shows it matters.
    """
    parts = pd.DataFrame(
        {
            "year": pd.to_numeric(gens["Operating Year"], errors="coerce"),
            "month": pd.to_numeric(gens["Operating Month"], errors="coerce"),
            "day": 1,
        }
    )
    dates = pd.to_datetime(parts, errors="coerce")
    frame = pd.DataFrame(
        {"Plant Code": gens["Plant Code"], "operating_date": dates.dt.tz_localize("UTC")}
    )
    return frame.groupby("Plant Code", as_index=False)["operating_date"].min()


def _array_tilt(merged: pd.DataFrame) -> pd.Series:
    """The tilt to model with: reported for fixed mounts, flat for trackers.

    EIA form question 30b asks for "the tilt angle of the unit" from
    fixed mounts and from single-axis technologies with a fixed tilt
    angle, so on paper a tracker's number is its axis tilt. The
    California data says otherwise. Tracker tilts are bimodal: a modal
    0, and a second cluster at 45, 52 and 60 degrees — textbook tracker
    rotation limits, not axis tilts. Eland Solar reports 60 across
    618 MW, as do Scarlet and Sandrini at 200 MW each. No utility-scale
    tracker stands its axis at 60 degrees; those respondents answered a
    different question from the one asked.

    So the reported value is used only where it is unambiguous, on
    fixed mounts. Trackers get a horizontal axis, which is both the
    standard utility-scale design and the modal reported value. This
    discards a real number for 13% of fleet capacity, which is the
    point — a 60 degree axis tilt would bend those plants' output
    curves badly, and wrongly.
    """
    reported = pd.to_numeric(merged["tilt"], errors="coerce")
    is_fixed = merged["tracking"] == "fixed"
    tilt = pd.Series(FALLBACK_AXIS_TILT, index=merged.index, dtype="float64")
    tilt[is_fixed] = reported[is_fixed].fillna(FALLBACK_FIXED_TILT)
    return tilt.astype("float64")


def write_registry(df: pd.DataFrame, path: Path = REGISTRY_PATH) -> None:
    """Write the registry in one atomic step.

    Every backfill worker re-reads this file at the start of each run,
    so a rebuild can land while a dozen processes are reading. Writing
    to a neighbouring temp file and renaming means a reader sees either
    the whole old file or the whole new one, never a half-written one.
    os.replace is atomic within a filesystem, which the temp file is
    guaranteed to share by sitting in the same directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=PLANTS_CA, preserve_index=False)
    staged = path.with_suffix(".parquet.tmp")
    pq.write_table(table, staged)
    os.replace(staged, path)


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
