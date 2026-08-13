"""Where the data lives: one seam between local disk and S3.

Every store in this project — labels, registry, weather, model, the
live forecasts — resolves its location through here. Locally that is
`data/`, unchanged. In CI it is a bucket, and nothing else in the
codebase knows the difference.

## No new dependency

pyarrow ships `S3FileSystem` compiled in, so reading and writing S3
costs nothing beyond what Gate 0 already locked. That is why this
module speaks `pyarrow.fs` rather than pandas' `s3://` support, which
would have pulled in s3fs, fsspec, aiobotocore and botocore to do the
same job.

## A Path still works

`_resolve` treats anything that does not start with `s3://` as a local
path. Every function here therefore accepts a plain `Path` and behaves
exactly as it did before this module existed — which is what lets the
test suite go on handing `tmp_path` to writers without knowing about
any of this.

## Configuration

    AMERICAST_DATA_ROOT=data                      # default, local disk
    AMERICAST_DATA_ROOT=s3://bucket/americast     # CI and production

Set it once in the environment. There is no per-module setting and
there should not be: two stores pointing at different roots is a
half-migrated system that looks like a working one.
"""

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ENV_VAR = "AMERICAST_DATA_ROOT"
DEFAULT_ROOT = "data"

# Objects under this prefix are the ones a browser fetches. The bucket
# policy grants anonymous GET here and nowhere else, so a file is
# public because of where it was written rather than because someone
# remembered to set an ACL.
PUBLIC_PREFIX = "public"


def root() -> str:
    """The configured data root, read fresh so tests can move it."""
    return os.environ.get(ENV_VAR, DEFAULT_ROOT)


def is_remote() -> bool:
    """True when the root is object storage rather than a disk."""
    return root().startswith("s3://")


def key(relative: str) -> Path | str:
    """Resolve a store's relative path against the root.

    Returns a `Path` locally and an `s3://` string remotely, so callers
    can keep their `path: Path = STORE_PATH` defaults and the value
    simply carries further when the root moves.
    """
    base = root().rstrip("/")
    return f"{base}/{relative}" if is_remote() else Path(base) / relative


def public(relative: str) -> Path | str:
    """Resolve a path under the browser-readable prefix."""
    return key(f"{PUBLIC_PREFIX}/{relative}")


def exists(location: Path | str) -> bool:
    """Does the object exist?"""
    filesystem, path = _resolve(location)
    return filesystem.get_file_info(path).type != pafs.FileType.NotFound


def read_parquet(location: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet store from wherever it lives."""
    filesystem, path = _resolve(location)
    return pq.read_table(path, columns=columns, filesystem=filesystem).to_pandas()


def read_schema(location: Path | str) -> pa.Schema:
    """The stored schema, without reading a row.

    Golden tests use this to decide whether a stored artifact still
    matches the declared schema, and skip rather than fail when it does
    not.
    """
    filesystem, path = _resolve(location)
    return pq.read_schema(path, filesystem=filesystem)


def write_parquet(table: pa.Table, location: Path | str) -> None:
    """Write a parquet store atomically, creating any parent directory.

    Object storage has no directories, so the mkdir is a local-only
    concern and is skipped remotely rather than emulated.

    **A reader sees the whole old file or the whole new one, never a
    half-written one.** On S3 that is free: `PutObject` is atomic and a
    GET during an overwrite returns the previous object. Locally it is
    not, so the write goes to a neighbouring temp file and is renamed —
    `os.replace` being atomic within a filesystem, which a sibling path
    is guaranteed to share.

    This matters because stores are read while they are rewritten. Every
    HRRR backfill worker re-reads the registry at the start of each run,
    and a dozen of them are running when a rebuild lands.
    """
    filesystem, path = _resolve(location)
    _ensure_parent(filesystem, path)

    if not isinstance(filesystem, pafs.LocalFileSystem):
        with filesystem.open_output_stream(path) as sink:
            pq.write_table(table, sink)
        return

    staged = f"{path}.tmp"
    with filesystem.open_output_stream(staged) as sink:
        pq.write_table(table, sink)
    os.replace(staged, path)


def write_text(location: Path | str, text: str) -> None:
    """Write a text file — the JSON contracts, and nothing else so far."""
    filesystem, path = _resolve(location)
    _ensure_parent(filesystem, path)
    with filesystem.open_output_stream(path) as sink:
        sink.write(text.encode())


def read_text(location: Path | str) -> str:
    """Read a text file back."""
    filesystem, path = _resolve(location)
    with filesystem.open_input_stream(path) as source:
        return source.readall().decode()


def listdir(location: Path | str, suffix: str = "") -> list[str]:
    """Every object directly under a prefix, sorted, optionally filtered.

    Replaces `Path.glob` for stores that are read as a set — the weather
    runs above all. Sorted because several callers depend on time order
    and get it from the filename.
    """
    filesystem, path = _resolve(location)
    selector = pafs.FileSelector(path, allow_not_found=True, recursive=False)
    found = [
        info.path
        for info in filesystem.get_file_info(selector)
        if info.type == pafs.FileType.File and info.path.endswith(suffix)
    ]
    return sorted(found)


def _resolve(location: Path | str) -> tuple[pafs.FileSystem, str]:
    """Split a location into the filesystem that holds it and its path.

    `FileSystem.from_uri` returns the path already stripped of the
    scheme and bucket, which is the form every pyarrow call wants.
    """
    text = str(location)
    if text.startswith("s3://"):
        return pafs.FileSystem.from_uri(text)
    return pafs.LocalFileSystem(), str(Path(text))


def _ensure_parent(filesystem: pafs.FileSystem, path: str) -> None:
    """Create the containing directory, where the concept applies."""
    if isinstance(filesystem, pafs.LocalFileSystem):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
