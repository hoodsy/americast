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
    # URL slug and the key every published object is filed under. Short,
    # lowercase and stable: it appears in public URLs, so renaming one
    # breaks every client that bookmarked it.
    id: str
    name: str  # display name, shown to a reader
    # What kind of thing this is. The balancing authority is the unit
    # that generalises: EIA-860 tags every plant with one, EIA-930
    # publishes hourly generation per one, and CAISO is simply the ISO
    # that operates a large one. An expansion to a utility or a
    # non-ISO balancing authority changes this field and nothing else.
    kind: str  # "iso" | "balancing_authority" | "utility"
    timezone: str  # IANA key; features/display only — storage stays UTC
    iso: str  # grid operator id as gridstatus spells it
    # Written by ingest/eia860, read everywhere. A Path locally, an
    # s3:// string when AMERICAST_DATA_ROOT points at a bucket.
    plant_registry_path: Path | str
    # Is there a public hourly actuals feed to score against? False is
    # a real state, not a defect: HRRR covers the whole country, so a
    # region can be forecast long before it can be graded. A consumer
    # must be able to tell the two apart.
    graded: bool = True


CAISO_CA = RegionConfig(
    id="caiso",
    name="California ISO",
    kind="iso",
    timezone="America/Los_Angeles",
    iso="CAISO",
    plant_registry_path=storage.key("registry/plants_ciso.parquet"),
)
