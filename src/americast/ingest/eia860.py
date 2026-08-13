"""EIA-860 ingestion: the CAISO utility-scale solar plant registry.

EIA-860 is the federal census of US generators (>= 1 MW): location,
capacity, status, and for solar, tracking type. Two vintages are read,
because neither one alone is both current and complete.

**EIA-860M**, the preliminary monthly inventory, is the spine. It says
which plants are operating right now, at what capacity, where, and since
when. It runs about two months behind rather than the annual file's
eight.

**The annual Solar schedule** supplies what the monthly file omits:
tracking type, tilt and azimuth. Array geometry does not change once a
plant is built, so reading it from an older vintage costs nothing. A
plant too new to appear in the annual file takes the fleet defaults
below.

## The filter is the balancing authority, not the state

This is the correction that Gate 5 forced, and it runs in both
directions. The label is CAISO's reported generation, and CAISO is a
balancing authority whose territory reaches into Arizona and Nevada. A
`state == CA` filter therefore did two wrong things at once: it admitted
140 Californian plants in LDWP, IID, BANC, PacifiCorp and WALC whose
output never reaches CAISO's number, and it excluded 2,478 MW of
Arizona and Nevada solar whose output does.

The second error was the expensive one. It made the modelled clear-sky
ceiling smaller than the fleet it was meant to bound, so CAISO's real
peak of 23,208 MW sat above the whole modelled fleet. See docs/model.md
for what that did to the confidence band.
"""

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa

from americast import storage
from americast.features.county import CISO_BA
from americast.schemas import PLANTS_CISO

EIA860_URL = "https://www.eia.gov/electricity/data/eia860/xls/eia8602025ER.zip"

# Pinned, like the annual URL. EIA publishes one of these a month and
# keeps the old ones, so naming the vintage makes a rebuild reproduce
# rather than quietly drift. Bump it, rebuild, and re-read the capacity
# line the run prints.
EIA860M_URL = (
    "https://www.eia.gov/electricity/data/eia860m/xls/june_generator2026.xlsx"
)

RAW_DIR = Path("data/eia860")
REGISTRY_PATH = storage.key("registry/plants_ciso.parquet")

_PLANT_XLSX = "2___Plant_Y2025_Early_Release.xlsx"
_SOLAR_XLSX = "3_3_Solar_Y2025_Early_Release.xlsx"
_MONTHLY_SHEET = "Operating"

# EIA spells the status "(OP) Operating" in the monthly file. The same
# sheet also carries "(OS)" and "(OA)" — out of service, one of them
# expected back. Neither generates today, and the label only counts what
# generated.
_OPERATING = "(OP)"
_SOLAR_PV = "Solar Photovoltaic"

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


def download_monthly(raw_dir: Path = RAW_DIR) -> Path:
    """Download the monthly inventory workbook; skips work already done."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / Path(EIA860M_URL).name
    if not path.exists():
        urllib.request.urlretrieve(EIA860M_URL, path)
    return path


def load_sheets(extracted: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Plant and Solar schedules (headers live on sheet row 3)."""
    plant = pd.read_excel(extracted / _PLANT_XLSX, skiprows=2)
    solar = pd.read_excel(extracted / _SOLAR_XLSX, skiprows=2)
    return plant, solar


def load_monthly(path: Path) -> pd.DataFrame:
    """Read the monthly inventory's Operating sheet.

    Same layout convention as the annual workbooks: the header is on
    sheet row 3. The last rows carry EIA's source note rather than
    generators, and they arrive with a null Plant ID, so they are cut
    here rather than surviving as a plant called NaN.
    """
    frame = pd.read_excel(path, sheet_name=_MONTHLY_SHEET, skiprows=2)
    return frame[frame["Plant ID"].notna()].copy()


def build_registry(solar: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per operating CAISO utility-scale solar PV plant.

    The monthly inventory decides who is in the fleet and supplies
    capacity, location and date. The annual Solar schedule supplies
    array geometry, joined on plant id, because the monthly file does
    not carry it.

    Geometry is a left join on purpose. A plant that appears in the
    monthly file and not in the annual one is a plant built since the
    annual vintage — real, generating, and with no reported tilt. It
    takes the fleet defaults, which is a better answer than dropping
    2,478 MW of desert because a spreadsheet is eight months old.
    """
    gens = monthly[
        (monthly["Balancing Authority Code"] == CISO_BA)
        & (monthly["Technology"] == _SOLAR_PV)
        & (monthly["Status"].astype("str").str.startswith(_OPERATING))
    ].copy()
    gens["dc_mw"] = _dc_capacity(gens)

    totals = (
        gens.groupby("Plant ID")
        .agg(
            capacity_mw_ac=("Nameplate Capacity (MW)", "sum"),
            dc_capacity_mw=("dc_mw", "sum"),
        )
        .reset_index()
    )
    online = _first_online(gens, key="Plant ID")
    location = gens.drop_duplicates("Plant ID")[
        [
            "Plant ID",
            "Plant Name",
            "Latitude",
            "Longitude",
            "County",
            "Balancing Authority Code",
        ]
    ]
    merged = totals.merge(online, on="Plant ID").merge(location, on="Plant ID")

    geometry = _geometry(solar, set(merged["Plant ID"]))
    merged = merged.merge(geometry, on="Plant ID", how="left")
    merged["tracking"] = merged["tracking"].fillna("unknown")

    out = pd.DataFrame(
        {
            "plant_id": merged["Plant ID"].astype("int64"),
            "plant_name": merged["Plant Name"].astype("str"),
            "latitude": pd.to_numeric(merged["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(merged["Longitude"], errors="coerce"),
            "capacity_mw_ac": merged["capacity_mw_ac"].astype("float64"),
            "dc_capacity_mw": merged["dc_capacity_mw"].astype("float64"),
            "tracking": merged["tracking"],
            "tilt": _reported_tilt(merged),
            "azimuth": merged["azimuth"].fillna(FALLBACK_AZIMUTH).astype("float64"),
            "county": merged["County"].fillna("UNKNOWN").astype("str"),
            "balancing_authority": merged["Balancing Authority Code"]
            .fillna("UNKNOWN")
            .astype("str"),
            "operating_date": merged["operating_date"],
        }
    )
    return out.sort_values("plant_id").reset_index(drop=True)


def _geometry(solar: pd.DataFrame, plant_ids: set) -> pd.DataFrame:
    """Tracking, tilt and azimuth per plant, from the annual schedule.

    Array geometry is fixed at construction, so an eight-month-old
    reading of it is as good as a fresh one. Keyed to `Plant ID` to
    match the monthly file's column name.
    """
    gens = solar[
        solar["Plant Code"].isin(plant_ids)
        & (solar["Status"] == "OP")
        & (solar["Technology"] == _SOLAR_PV)
    ].copy()
    if gens.empty:
        return pd.DataFrame(columns=["Plant ID", "tracking", "tilt", "azimuth"])
    gens["tracking"] = _tracking_type(gens)
    return _layout(gens).rename(columns={"Plant Code": "Plant ID"})


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


def _first_online(gens: pd.DataFrame, key: str = "Plant Code") -> pd.DataFrame:
    """The month a plant's first phase started generating, in UTC.

    Only 79.7% of today's California solar capacity was running at the
    end of 2023, so weighting a 2023 forecast hour with today's fleet
    would invent a fifth of the state's panels. The date lets the
    aggregation drop plants that did not exist yet.

    One date per plant, not per phase. A plant whose second phase
    arrived later therefore counts at full size from its first — 27
    plants holding 3.74 GW are built in stages, but only 8 of them
    (0.04 GW) stagger by more than two years, so the residual is small.
    Dating each generator separately is the fix if the residual ever
    shows up in evaluation.
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
        {key: gens[key], "operating_date": dates.dt.tz_localize("UTC")}
    )
    return frame.groupby(key, as_index=False)["operating_date"].min()


def _reported_tilt(merged: pd.DataFrame) -> pd.Series:
    """Tilt exactly as EIA reported it, with only the blanks filled.

    Stored raw rather than interpreted, because the number means
    different things on different mounts and only the power model has
    the context to tell them apart. On a fixed mount it is the panel
    angle. On a tracker, EIA form question 30b asks for the axis tilt,
    but a third of the tracked capacity that answers gives 45, 52 or
    60 degrees — rotation limits, not axis tilts. No utility-scale
    tracker stands its axis at 60 degrees.

    Both readings are useful and neither is guesswork: below about 30
    degrees the number can only be an axis tilt, above it can only be a
    rotation limit. `features/power.py` makes that split. Throwing the
    value away here would lose a real per-plant limit for 2.7 GW.

    A blank on a fixed mount takes the fleet's tilt, because a panel
    angle it must have. A blank on a tracker becomes zero, which reads
    as a flat axis and leaves the rotation limit to the fleet default —
    the same answer as the 4.9 GW that reports zero outright.
    """
    reported = pd.to_numeric(merged["tilt"], errors="coerce")
    is_fixed = merged["tracking"] == "fixed"
    blank = pd.Series(FALLBACK_AXIS_TILT, index=merged.index, dtype="float64")
    blank[is_fixed] = FALLBACK_FIXED_TILT
    return reported.fillna(blank).astype("float64")


def write_registry(df: pd.DataFrame, path: Path | str = REGISTRY_PATH) -> None:
    """Write the registry in one atomic step.

    Every backfill worker re-reads this file at the start of each run,
    so a rebuild can land while a dozen processes are reading. Writing
    to a neighbouring temp file and renaming means a reader sees either
    the whole old file or the whole new one, never a half-written one.
    os.replace is atomic within a filesystem, which the temp file is
    guaranteed to share by sitting in the same directory.
    """
    table = pa.Table.from_pandas(df, schema=PLANTS_CISO, preserve_index=False)
    storage.write_parquet(table, path)


def verify(registry: pd.DataFrame) -> dict:
    """What the schema cannot express: is this the fleet CAISO reports?

    - `non_ciso`: plants outside the balancing authority. Any is a bug;
      their output never enters the label.
    - `by_state`: capacity per state. Arizona and Nevada appearing here
      is the point of the balancing-authority filter, not a fault.
    - `default_geometry`: plants too new for the annual schedule, which
      therefore carry fleet-default tilt and azimuth.
    - `unknown_county`: counties `features.county` cannot map to a zone.
      `fleet()` raises on these, so this is the earlier, kinder warning.
    """
    from americast.features.county import COUNTY_ZONE

    counties = registry["county"].str.lower()
    return {
        "n_plants": len(registry),
        "capacity_gw": registry["capacity_mw_ac"].sum() / 1000.0,
        "non_ciso": int((registry["balancing_authority"] != CISO_BA).sum()),
        "default_geometry": int((registry["tracking"] == "unknown").sum()),
        "unknown_county": sorted(set(counties) - set(COUNTY_ZONE)),
        "newest_plant": registry["operating_date"].max(),
        "missing_coordinates": int(
            registry[["latitude", "longitude"]].isna().any(axis=1).sum()
        ),
    }


if __name__ == "__main__":
    _, solar = load_sheets(download_raw())
    monthly = load_monthly(download_monthly())
    registry = build_registry(solar, monthly)

    dropped = registry[registry[["latitude", "longitude"]].isna().any(axis=1)]
    if not dropped.empty:
        print(f"dropping {len(dropped)} plants with missing coordinates:")
        print(dropped[["plant_id", "plant_name", "capacity_mw_ac"]].to_string())
        registry = registry.dropna(subset=["latitude", "longitude"])

    registry = registry.reset_index(drop=True)
    write_registry(registry)
    audit = verify(registry)
    print(
        f"registry written: {audit['n_plants']} plants, "
        f"{audit['capacity_gw']:.2f} GW AC → {REGISTRY_PATH}"
    )
    print(f"  newest plant     {audit['newest_plant']:%Y-%m}")
    print(f"  default geometry {audit['default_geometry']} plants")
    if audit["unknown_county"]:
        print(f"  UNMAPPED COUNTIES: {audit['unknown_county']}")
