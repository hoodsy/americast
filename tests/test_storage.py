"""The storage seam: does a Path still behave, and does the root move?

No network. The S3 branch is exercised only as far as path resolution,
because everything past that is pyarrow's to get right.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from americast import storage


@pytest.fixture(autouse=True)
def local_root(tmp_path, monkeypatch):
    monkeypatch.setenv(storage.ENV_VAR, str(tmp_path))
    return tmp_path


def table() -> pa.Table:
    return pa.Table.from_pandas(
        pd.DataFrame({"a": [1, 2, 3]}), preserve_index=False
    )


# --- resolution -------------------------------------------------------


def test_a_local_root_resolves_to_a_path(local_root) -> None:
    assert storage.key("caiso/labels.parquet") == local_root / "caiso/labels.parquet"
    assert not storage.is_remote()


def test_an_s3_root_resolves_to_a_uri(monkeypatch) -> None:
    monkeypatch.setenv(storage.ENV_VAR, "s3://bucket/americast")
    assert storage.key("caiso/labels.parquet") == "s3://bucket/americast/caiso/labels.parquet"
    assert storage.is_remote()


def test_a_trailing_slash_does_not_double_up(monkeypatch) -> None:
    monkeypatch.setenv(storage.ENV_VAR, "s3://bucket/americast/")
    assert storage.key("a.parquet") == "s3://bucket/americast/a.parquet"


def test_the_default_root_is_the_local_data_directory(monkeypatch) -> None:
    monkeypatch.delenv(storage.ENV_VAR, raising=False)
    assert storage.root() == storage.DEFAULT_ROOT
    assert not storage.is_remote()


def test_the_root_is_read_fresh_every_time(monkeypatch) -> None:
    """Import-time capture would make the env var untestable and CI-fragile."""
    monkeypatch.setenv(storage.ENV_VAR, "s3://one/x")
    first = storage.key("a")
    monkeypatch.setenv(storage.ENV_VAR, "s3://two/y")
    assert storage.key("a") != first


def test_public_writes_land_under_the_browser_prefix(monkeypatch) -> None:
    monkeypatch.setenv(storage.ENV_VAR, "s3://bucket/americast")
    assert storage.public("forecast.json") == (
        f"s3://bucket/americast/{storage.PUBLIC_PREFIX}/forecast.json"
    )


# --- a plain Path still works ----------------------------------------


def test_a_bare_path_round_trips(tmp_path) -> None:
    """Every existing caller hands these functions a Path. That must keep working."""
    path = tmp_path / "nested" / "store.parquet"
    storage.write_parquet(table(), path)
    assert path.exists()
    assert list(storage.read_parquet(path)["a"]) == [1, 2, 3]


def test_writing_creates_the_parent_directory(tmp_path) -> None:
    storage.write_parquet(table(), tmp_path / "a" / "b" / "c.parquet")
    assert (tmp_path / "a" / "b" / "c.parquet").exists()


def test_exists_answers_for_a_missing_object(tmp_path) -> None:
    assert not storage.exists(tmp_path / "absent.parquet")
    storage.write_parquet(table(), tmp_path / "present.parquet")
    assert storage.exists(tmp_path / "present.parquet")


def test_columns_can_be_selected_without_reading_the_rest(tmp_path) -> None:
    wide = pa.Table.from_pandas(
        pd.DataFrame({"a": [1], "b": [2]}), preserve_index=False
    )
    storage.write_parquet(wide, tmp_path / "wide.parquet")
    assert list(storage.read_parquet(tmp_path / "wide.parquet", columns=["a"])) == ["a"]


def test_the_schema_reads_without_the_rows(tmp_path) -> None:
    storage.write_parquet(table(), tmp_path / "s.parquet")
    assert storage.read_schema(tmp_path / "s.parquet").names == ["a"]


def test_text_round_trips(tmp_path) -> None:
    storage.write_text(tmp_path / "out" / "forecast.json", '{"a": 1}')
    assert storage.read_text(tmp_path / "out" / "forecast.json") == '{"a": 1}'


# --- listing ----------------------------------------------------------


def test_listdir_is_sorted_and_filtered(tmp_path) -> None:
    """The weather store is read as a set and depends on filename order."""
    for name in ("hrrr_20240102_06z.parquet", "hrrr_20240101_06z.parquet", "notes.txt"):
        storage.write_text(tmp_path / name, "x")
    found = storage.listdir(tmp_path, suffix=".parquet")
    assert [Path(p).name for p in found] == [
        "hrrr_20240101_06z.parquet",
        "hrrr_20240102_06z.parquet",
    ]


def test_listdir_on_a_missing_prefix_is_empty_not_an_error(tmp_path) -> None:
    """A first run has no store yet, and that is not a failure."""
    assert storage.listdir(tmp_path / "never_written") == []


def test_listdir_results_can_be_read_back(tmp_path) -> None:
    """The round trip every fold depends on.

    `features.table.build` lists the run store and reads each path it
    gets back. pyarrow reports S3 objects as `bucket/key` with no
    scheme, and a scheme-less string resolves as a local path — so
    without restoring it, listing a remote store yields paths that
    cannot be opened.
    """
    storage.write_parquet(table(), tmp_path / "a.parquet")
    storage.write_parquet(table(), tmp_path / "b.parquet")
    for found in storage.listdir(tmp_path, ".parquet"):
        assert len(storage.read_parquet(found)) == 3
