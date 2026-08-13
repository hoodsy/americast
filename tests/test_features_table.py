import pandas as pd
import pyarrow.parquet as pq
import pytest
from test_features import registry, weather

from americast.features.table import build, load, verify, write
from americast.region import CAISO_CA
from americast.schemas import HRRR_WEATHER, PLANTS_CISO, TRAIN_TABLE

# One plant per zone, because the training table's schema declares
# every zone's weather non-nullable — a fleet missing a zone cannot be
# written at all, which `test_an_absent_zone_cannot_be_written` pins.
# Yuma is the sonoran zone, and it is Arizona: CISO reaches outside
# California, so the fixture fleet has to as well.
ALL_ZONES = {
    "plant_ids": (1, 2, 3, 4, 5, 6),
    "counties": ("Kern", "Imperial", "Fresno", "Riverside", "Monterey", "Yuma"),
    "capacities": (100.0, 200.0, 50.0, 300.0, 20.0, 150.0),
}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> tuple:
    """A tiny two-run weather store and a matching label file.

    Small on purpose: this checks the wiring of the fold, the schema
    write and the label join. Whether the numbers describe California
    is the golden tests' job, and they read the real table.
    """
    root = tmp_path_factory.mktemp("store")
    hrrr = root / "hrrr"
    hrrr.mkdir()

    plants = registry(**ALL_ZONES)
    for day in ("20240615", "20240616"):
        run = f"2024-06-{day[-2:]} 06:00"
        frame = weather(run, leads=range(1, 25), plant_ids=ALL_ZONES["plant_ids"])
        frame.to_parquet(hrrr / f"hrrr_{day}_06z.parquet")

    stamps = pd.date_range("2024-06-15", periods=72, freq="1h", tz="UTC")
    label = pd.DataFrame({"utc_time": stamps, "solar_mw": 5000.0})
    caiso = root / "caiso.parquet"
    label.to_parquet(caiso)

    registry_path = root / "plants.parquet"
    plants.to_parquet(registry_path)
    region = CAISO_CA.__class__(
        id="test",
        name="TEST",
        kind="iso",
        timezone=CAISO_CA.timezone,
        iso=CAISO_CA.iso,
        plant_registry_path=registry_path,
    )
    return region, hrrr, caiso, root


@pytest.fixture(scope="module")
def built(store) -> pd.DataFrame:
    region, hrrr, caiso, _ = store
    return build(region=region, hrrr_dir=hrrr, caiso_path=caiso)


def test_the_weather_fixture_matches_the_real_schema() -> None:
    """If this drifts, every test in this file is testing a fiction."""
    frame = weather("2024-06-15 06:00")
    for field in HRRR_WEATHER:
        assert field.name in frame.columns


def test_the_registry_fixture_matches_the_real_schema() -> None:
    """The gap this closes cost an afternoon: plant_name was missing."""
    frame = registry()
    for field in PLANTS_CISO:
        assert field.name in frame.columns, f"fixture is missing {field.name}"


def test_the_fold_keeps_one_row_per_run_and_hour(built: pd.DataFrame) -> None:
    assert not built.duplicated(["run_time", "valid_time"]).any()
    assert built["run_time"].nunique() == 2


def test_every_run_loses_exactly_its_last_hour(built: pd.DataFrame) -> None:
    per_run = built.groupby("run_time").size()
    assert (per_run == 23).all(), "24 forecast hours in, 23 hour means out"


def test_the_label_joined_on_valid_time(built: pd.DataFrame) -> None:
    assert (built["solar_mw"] == 5000.0).all()


def test_an_unlabelled_hour_survives_as_a_feature_row(store) -> None:
    """Gate 6 grades tomorrow; the row has to exist to be graded."""
    region, hrrr, _, root = store
    short = pd.DataFrame(
        {"utc_time": pd.date_range("2024-06-15", periods=3, freq="1h", tz="UTC")}
    )
    short["solar_mw"] = 100.0
    path = root / "short_label.parquet"
    short.to_parquet(path)

    out = build(region=region, hrrr_dir=hrrr, caiso_path=path)
    assert out["solar_mw"].isna().any()
    assert len(out) == 46, "no feature row was dropped for want of a label"


def test_an_empty_store_fails_loudly(store, tmp_path) -> None:
    region, _, caiso, _ = store
    with pytest.raises(FileNotFoundError, match="no HRRR runs"):
        build(region=region, hrrr_dir=tmp_path, caiso_path=caiso)


def test_the_written_table_matches_the_declared_schema(built, tmp_path) -> None:
    """The schema is the check; this proves it is actually applied."""
    path = tmp_path / "table.parquet"
    write(built, path)
    assert pq.read_schema(path).equals(TRAIN_TABLE)
    assert len(load(path)) == len(built)


def test_a_wrong_column_cannot_be_written(built, tmp_path) -> None:
    broken = built.drop(columns=["fleet_ac_mw"])
    with pytest.raises(KeyError):
        write(broken, tmp_path / "broken.parquet")


def test_an_absent_zone_cannot_be_written(store, tmp_path) -> None:
    """A fleet with an empty zone must stop the build, not fill a guess.

    An empty zone has no temperature. The column is carried through as
    null so the declared schema can refuse it here, which is the whole
    reason `_spread` reindexes instead of letting the pivot drop it.
    """
    region, _, caiso, root = store
    hrrr = root / "one_zone"
    hrrr.mkdir(exist_ok=True)
    weather("2024-06-15 06:00", leads=range(1, 25), plant_ids=(1,)).to_parquet(
        hrrr / "hrrr_20240615_06z.parquet"
    )
    only_kern = root / "kern.parquet"
    registry(plant_ids=(1,), counties=("Kern",), capacities=(100.0,)).to_parquet(
        only_kern
    )
    narrow = region.__class__(
        id="kern",
        name="KERN",
        kind="iso",
        timezone=region.timezone,
        iso=region.iso,
        plant_registry_path=only_kern,
    )
    partial = build(region=narrow, hrrr_dir=hrrr, caiso_path=caiso)
    assert partial["coastal_dswrf"].isna().all()
    assert (partial["coastal_ac_mw"] == 0.0).all(), "no plants is zero megawatts"
    with pytest.raises(ValueError, match="non-nullable"):
        write(partial, tmp_path / "narrow.parquet")


def test_verify_reports_a_clean_table(built: pd.DataFrame) -> None:
    audit = verify(built)
    assert audit["predicts_the_past"] == 0
    assert audit["n_runs"] == 2
    assert len(audit["missing_days"]) == 0


def test_verify_catches_a_row_that_predicts_the_past(built: pd.DataFrame) -> None:
    broken = built.copy()
    broken.loc[broken.index[0], "valid_time"] = broken.loc[
        broken.index[0], "run_time"
    ] - pd.Timedelta(hours=1)
    assert verify(broken)["predicts_the_past"] == 1


def test_two_runs_are_not_enough_for_a_baseline(built: pd.DataFrame) -> None:
    """Smart persistence needs a week; it must say so rather than guess."""
    assert built["baseline_smart_mw"].isna().all()


def test_a_table_with_no_labels_at_all_still_builds(store) -> None:
    """A fresh region, or Gate 6 before the first grading, hits this."""
    region, hrrr, _, root = store
    stamps = pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC")
    unrelated = pd.DataFrame({"utc_time": stamps, "solar_mw": 100.0})
    path = root / "no_overlap.parquet"
    unrelated.to_parquet(path)

    out = build(region=region, hrrr_dir=hrrr, caiso_path=path)
    assert out["solar_mw"].isna().all()
    assert out["baseline_clear_sky_mw"].isna().all()
    assert out["baseline_smart_mw"].isna().all()
