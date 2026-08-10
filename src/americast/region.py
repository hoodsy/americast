"""Region configuration: the one seam for extending beyond California.

Every module that touches region-specific facts (which grid operator to
query, which timezone to use for feature engineering, where the plant
registry lives) takes a RegionConfig instead of hardcoding them.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegionConfig:
    name: str
    timezone: str  # IANA key; features/display only — storage stays UTC
    iso: str  # grid operator id as gridstatus spells it
    plant_registry_path: Path  # written by ingest/eia860, read everywhere


CAISO_CA = RegionConfig(
    name="CAISO_CA",
    timezone="America/Los_Angeles",
    iso="CAISO",
    plant_registry_path=Path("data/registry/plants_ca.parquet"),
)
