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

import gzip
import os
from functools import lru_cache
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


def child(prefix: Path | str, name: str) -> Path | str:
    """One object directly under a prefix, in the prefix's own idiom.

    `Path / name` is wrong for a remote store and string concatenation is
    wrong for a local one, and callers that guess have to guess in every
    branch. Path arithmetic is the specific trap: `Path("s3://b/k")`
    silently collapses the double slash to `s3:/b/k`, which then reads as
    a relative local path and fails naming a key that plainly exists.
    """
    if isinstance(prefix, Path):
        return prefix / name
    return f"{str(prefix).rstrip('/')}/{name}"


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


def write_text(
    location: Path | str, text: str, cache_control: str | None = None
) -> None:
    """Write a text file — the JSON contracts, and nothing else so far.

    A `.json` object is tagged `application/json` on the way out. S3
    defaults every upload to `application/octet-stream`, which a browser
    will still parse but will not preview, and which makes the object
    look like a download rather than an API response.

    `cache_control` reaches S3 as the object's own header, so a reader
    keeps its copy for exactly as long as the object is valid. Locally
    there is nowhere to put it and it is dropped, as Content-Type is.
    """
    filesystem, path = _resolve(location)
    _ensure_parent(filesystem, path)
    headers = _headers(path, cache_control)
    with filesystem.open_output_stream(
        path, compression=None, metadata=headers
    ) as sink:
        sink.write(text.encode())


def write_gzip(
    location: Path | str, text: str, cache_control: str | None = None
) -> None:
    """Write text compressed, for the objects large enough to need it.

    **There is no `Content-Encoding` header on this object.** pyarrow
    does not pass one through to S3 at any spelling — verified against
    the real bucket, not assumed, because arrow ignores metadata keys it
    does not know without saying so. The object is therefore named
    `.json.gz`, typed `application/gzip` honestly, and the browser
    decompresses it itself through `DecompressionStream('gzip')`.

    `mtime=0` matters: gzip writes the current time into its header, so
    without it the same text would produce different bytes every morning
    and every publish would look like a change.

    `compression=None` matters too: arrow's default is `detect`, which
    would see the `.gz` suffix and compress these bytes a second time.
    """
    filesystem, path = _resolve(location)
    _ensure_parent(filesystem, path)
    packed = gzip.compress(text.encode(), mtime=0)
    headers = _headers(path, cache_control)
    with filesystem.open_output_stream(
        path, compression=None, metadata=headers
    ) as sink:
        sink.write(packed)


def read_gzip(location: Path | str) -> str:
    """Read a compressed text object back."""
    filesystem, path = _resolve(location)
    with filesystem.open_input_stream(path, compression=None) as source:
        return gzip.decompress(source.readall()).decode()


def _headers(path: str, cache_control: str | None) -> dict[str, str] | None:
    """S3 object metadata for a write, or None where there is none.

    Only keys arrow is known to forward appear here. It drops the rest
    without complaining, which is how `Content-Encoding` was lost.
    """
    headers = {}
    if path.endswith(".json"):
        headers["Content-Type"] = "application/json"
    elif path.endswith(".gz"):
        headers["Content-Type"] = "application/gzip"
    if cache_control:
        headers["Cache-Control"] = cache_control
    return headers or None


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

    **The results go straight back into `read_parquet`.** pyarrow reports
    S3 paths as `bucket/key`, with no scheme, and `_resolve` reads a
    scheme-less string as a local path — so returning them raw produced
    a `FileNotFoundError` naming a key that plainly existed. The scheme
    is put back here, which is the only place that knows it was stripped.
    """
    filesystem, path = _resolve(location)
    selector = pafs.FileSelector(path, allow_not_found=True, recursive=False)
    found = [
        info.path
        for info in filesystem.get_file_info(selector)
        if info.type == pafs.FileType.File and info.path.endswith(suffix)
    ]
    remote = not isinstance(filesystem, pafs.LocalFileSystem)
    return sorted(f"s3://{item}" if remote else item for item in found)


def _resolve(location: Path | str) -> tuple[pafs.FileSystem, str]:
    """Split a location into the filesystem that holds it and its path.

    `S3FileSystem` is constructed directly rather than through
    `FileSystem.from_uri`, which would also work — both run the AWS
    SDK's normal credential chain. The reason is caching: `from_uri`
    builds a fresh filesystem, and therefore re-initialises the SDK,
    on every call. `_s3` builds one per region and keeps it.

    Paths lose the scheme and keep the bucket, which is the form every
    pyarrow call wants.
    """
    text = str(location)
    if not text.startswith("s3://"):
        return pafs.LocalFileSystem(), str(Path(text))
    return _s3(_region()), text[len("s3://") :]


def _region() -> str | None:
    """The region for S3 calls, or None to let the SDK work it out."""
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


@lru_cache(maxsize=4)
def _s3(region: str | None) -> pafs.S3FileSystem:
    """One filesystem per region, reused.

    Constructing an S3FileSystem initialises the AWS SDK and resolves
    credentials, which is far too expensive to repeat for every object
    in a fold over a thousand runs.

    Credentials come from the SDK's own chain, which reads the standard
    environment variables and `~/.aws/credentials`. Note that it does
    **not** honour `AWS_PROFILE` the way the `aws` CLI does, and it does
    not understand the session `aws login` writes — so a shell where the
    CLI works is not necessarily a shell where this works. Export real
    credentials into the environment, which is what CI does anyway.
    """
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()


def _ensure_parent(filesystem: pafs.FileSystem, path: str) -> None:
    """Create the containing directory, where the concept applies."""
    if isinstance(filesystem, pafs.LocalFileSystem):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
