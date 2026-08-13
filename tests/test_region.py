import dataclasses
from zoneinfo import ZoneInfo

import pytest

from americast.region import CAISO_CA


def test_caiso_ca_fields_populated() -> None:
    assert CAISO_CA.id == "caiso"
    assert CAISO_CA.name == "California ISO"
    assert CAISO_CA.kind == "iso"
    assert CAISO_CA.iso == "CAISO"
    assert ZoneInfo(CAISO_CA.timezone).key == "America/Los_Angeles"
    assert CAISO_CA.plant_registry_path.suffix == ".parquet"


def test_region_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        CAISO_CA.name = "ERCOT"  # type: ignore[misc]
