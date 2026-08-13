"""EIA-923 ingestion: monthly net generation, one row per plant.

The only per-plant truth in this project. Everything else is statewide:
CAISO publishes one number for the whole fleet, and a single number can
say that the fleet out-produces the physical model but never which
plants do. That distinction is the whole reason this module exists.

## The question it was built to answer

The physical model was almost exactly right in 2023 — median
`(generation + curtailment) / modelled` of 1.0021 — and under-predicts
by 8% in 2025. The drift is uniform across seasons and hours, which
rules out weather, and it survives adding curtailment back, which rules
out grid economics. What remains is the equipment: either plants built
recently out-produce the model, or every plant increasingly does.

A statewide sum cannot separate those two. Monthly per-plant generation
can, by grouping plants by the year they were commissioned and asking
whether the newer group's shortfall is larger.

## What is here and what is not

EIA collects monthly from larger generators and annually from the rest,
so this covers about two thirds of CISO capacity across roughly 114
plants — split near evenly between pre-2022 and 2022-onward capacity,
which is what makes the comparison possible.

Net generation is metered at the grid connection over the whole month.
It is energy, not power, and it is net of station service — the same
convention as the CAISO label, one level down. It is therefore directly
comparable to the physical model's megawatts summed over the month,
and to nothing else without care.

## Two vintages of URL

EIA moves a year's workbook into an archive directory once the next
year opens. The current year sits at the plain path and holds only the
months published so far. Both are pinned here rather than discovered,
so a rebuild reproduces.
"""

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from americast.schemas import EIA923_SOLAR_MONTHLY

RAW_DIR = Path("data/eia923")
STORE_PATH = Path("data/eia923/solar_monthly.parquet")

_ARCHIVE = "https://www.eia.gov/electricity/data/eia923/archive/xls/f923_{year}.zip"
_CURRENT = "https://www.eia.gov/electricity/data/eia923/xls/f923_{year}.zip"

# The year EIA has not yet archived. Everything before it is settled and
# will not change; this one grows a month at a time.
CURRENT_YEAR = 2026

YEARS = (2023, 2024, 2025, 2026)

# EIA's code for solar in the fuel-type column.
SOLAR_CODE = "SUN"

# The generation sheet, and how far down its real header sits.
_SHEET = "Page 1 Generation and Fuel Data"
_HEADER_ROWS = 5

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def url_for(year: int) -> str:
    """Where EIA keeps that year's workbook."""
    template = _CURRENT if year >= CURRENT_YEAR else _ARCHIVE
    return template.format(year=year)


def download_year(year: int, raw_dir: Path = RAW_DIR) -> Path:
    """Download and extract one year's zip; skips work already done.

    Returns the workbook path. EIA stamps the publication date into the
    filename, so the file is found by glob rather than by name — the
    same year re-published in March and in July has two different names
    and identical content.
    """
    extracted = raw_dir / f"y{year}"
    found = _workbook(extracted)
    if found is not None:
        return found

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"f923_{year}.zip"
    if not zip_path.exists():
        # EIA redirects the archive path, and urlretrieve follows it.
        urllib.request.urlretrieve(url_for(year), zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted)

    workbook = _workbook(extracted)
    if workbook is None:
        raise FileNotFoundError(f"no generation workbook inside {zip_path}")
    return workbook


def load_year(path: Path, year: int) -> pd.DataFrame:
    """One workbook's solar rows, as plant-months.

    A plant files one row per prime mover and fuel, so a site with two
    inverter technologies appears twice and its rows are summed. Months
    the plant did not report arrive as EIA's "." placeholder and become
    null, which is dropped rather than zeroed — a plant that filed
    nothing for March did not generate zero in March, it said nothing.
    """
    raw = pd.read_excel(path, sheet_name=_SHEET, skiprows=_HEADER_ROWS)
    columns = _columns(raw)
    solar = raw[raw[columns["fuel"]] == SOLAR_CODE]
    if solar.empty:
        return _empty()

    tidy = []
    for number, name in enumerate(_MONTHS, start=1):
        values = pd.to_numeric(solar[columns[name]], errors="coerce")
        tidy.append(
            pd.DataFrame(
                {
                    "plant_id": solar[columns["plant"]],
                    "month": pd.Timestamp(f"{year}-{number:02d}-01", tz="UTC"),
                    "net_generation_mwh": values,
                }
            )
        )
    stacked = pd.concat(tidy, ignore_index=True).dropna(
        subset=["plant_id", "net_generation_mwh"]
    )
    return (
        stacked.groupby(["plant_id", "month"], as_index=False)["net_generation_mwh"]
        .sum()
        .astype({"plant_id": "int64", "net_generation_mwh": "float64"})
        .sort_values(["plant_id", "month"], ignore_index=True)
    )


def build(years: tuple[int, ...] = YEARS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Every year's solar plant-months, stacked."""
    frames = [load_year(download_year(year, raw_dir), year) for year in years]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["plant_id", "month"], ignore_index=True
    )


def write(frame: pd.DataFrame, path: Path = STORE_PATH) -> None:
    """Write under the declared schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        frame[[field.name for field in EIA923_SOLAR_MONTHLY]],
        schema=EIA923_SOLAR_MONTHLY,
        preserve_index=False,
    )
    pq.write_table(table, path)


def load(path: Path = STORE_PATH) -> pd.DataFrame:
    """Read the stored plant-months."""
    return pd.read_parquet(path)


def verify(frame: pd.DataFrame, registry: pd.DataFrame | None = None) -> dict:
    """Value checks the schema cannot express.

    - `negative_months`: a month of net generation below zero. Real but
      rare — a plant offline all month still draws station service — so
      a large count means a parsing fault rather than a quiet winter.
    - `duplicated`: more than one row for a plant-month. The prime-mover
      rows must have been summed, not stacked.
    - `matched_plants` / `matched_capacity_share`: how much of the CISO
      fleet this sample actually covers, which decides whether any
      comparison drawn from it means anything.
    """
    out = {
        "n_rows": len(frame),
        "n_plants": frame["plant_id"].nunique(),
        "span": (frame["month"].min(), frame["month"].max()),
        "negative_months": int((frame["net_generation_mwh"] < 0.0).sum()),
        "duplicated": int(frame.duplicated(["plant_id", "month"]).sum()),
        "total_twh": frame["net_generation_mwh"].sum() / 1e6,
    }
    if registry is not None:
        matched = registry[registry["plant_id"].isin(set(frame["plant_id"]))]
        out["matched_plants"] = len(matched)
        out["matched_capacity_share"] = (
            matched["capacity_mw_ac"].sum() / registry["capacity_mw_ac"].sum()
        )
    return out


def _workbook(extracted: Path) -> Path | None:
    """The generation workbook inside an extracted year, if it is there."""
    if not extracted.exists():
        return None
    found = sorted(extracted.glob("EIA923_Schedules_2_3_4_5*.xlsx"))
    return found[0] if found else None


def _columns(raw: pd.DataFrame) -> dict[str, str]:
    """Map the columns we need onto EIA's spelling of them.

    The headers carry embedded newlines — the plant's fuel code is
    literally "Reported\\nFuel Type Code" — and the wrapping has moved
    between vintages. Matching on flattened text survives that; naming
    the columns literally does not.
    """
    flat = {" ".join(str(column).split()).lower(): column for column in raw.columns}

    def find(*words: str) -> str:
        for text, original in flat.items():
            if all(word in text for word in words):
                return original
        raise KeyError(f"no EIA-923 column matching {words}")

    mapping = {
        "plant": find("plant", "id"),
        "fuel": find("reported", "fuel", "type"),
    }
    for name in _MONTHS:
        mapping[name] = find("netgen", name.lower())
    return mapping


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": pd.Series([], dtype="int64"),
            "month": pd.Series([], dtype="datetime64[us, UTC]"),
            "net_generation_mwh": pd.Series([], dtype="float64"),
        }
    )


if __name__ == "__main__":
    from americast.region import CAISO_CA

    built = build()
    write(built)
    registry = pd.read_parquet(CAISO_CA.plant_registry_path)
    audit = verify(built, registry)

    print(f"{audit['n_rows']:,} plant-months from {audit['n_plants']} plants")
    print(f"  span {audit['span'][0]:%Y-%m} to {audit['span'][1]:%Y-%m}")
    print(f"  {audit['total_twh']:,.1f} TWh total -> {STORE_PATH}")
    print(
        f"  covers {audit['matched_plants']} CISO plants, "
        f"{audit['matched_capacity_share']:.1%} of fleet capacity"
    )
