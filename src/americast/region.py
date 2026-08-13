"""Region configuration: the one seam for extending beyond California.

Every module that touches region-specific facts (which grid operator to
query, which timezone to use for feature engineering, where the plant
registry lives) takes a RegionConfig instead of hardcoding them.
"""

from dataclasses import dataclass
from pathlib import Path

from americast import storage


@dataclass(frozen=True)
class RegionConfig:
    name: str
    timezone: str  # IANA key; features/display only — storage stays UTC
    iso: str  # grid operator id as gridstatus spells it
    # Written by ingest/eia860, read everywhere. A Path locally, an
    # s3:// string when AMERICAST_DATA_ROOT points at a bucket.
    plant_registry_path: Path | str


CAISO_CA = RegionConfig(
    name="CAISO_CA",
    timezone="America/Los_Angeles",
    iso="CAISO",
    plant_registry_path=storage.key("registry/plants_ciso.parquet"),
)
