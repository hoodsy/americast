# The published archive

What the bucket serves to a browser, how it is written, and what to do
when it looks wrong.

The design decision behind it is
`docs/superpowers/specs/2026-08-18-bucket-served-archive-design.md`.
This page is the operational half.

## What it is

One directory per model run, plus an index over them, under the
browser-readable prefix:

    public/regions.json                            catalogue of regions
    public/caiso/forecast.json                     the newest run
    public/caiso/scoreboard.json                   the rolling record
    public/caiso/plants.json.gz                    static plant metadata
    public/caiso/runs.json                         run index, newest first
    public/caiso/runs/20260818T06z/forecast.json   the curve as issued
    public/caiso/runs/20260818T06z/totals.json     zone and county curves
    public/caiso/runs/20260818T06z/plants.json.gz  per-plant values

The run key is `20260818T06z` — the same spelling as the weather file it
comes from, `hrrr_20260818_06z.parquet`, rather than a second spelling
of the same instant.

**No client builds a key.** `regions.json` points at a region's
`runs.json`, and every entry in `runs.json` carries its own `path`. A
second region, or a second run hour a day, then appears without a
frontend deploy.

## Nothing here is computed

Every object is a projection of a store that already exists:

| Object | Comes from |
|---|---|
| `forecast.json` | `live/forecasts.parquet`, `live/scores.parquet` |
| `totals.json` | `hrrr/hrrr_*.parquet` + the registry, through `api.frames` |
| `plants.json.gz` | the same, through `api.frames` |
| `runs.json` | `live/forecasts.parquet`, `live/scores.parquet` |

Three things follow. Re-running is safe. A sealed object can be rebuilt
byte-identically when a bug is found. And the JSON never becomes a
second source of truth that can disagree with the parquet.

The map halves go through `api.frames`, the same functions the local
FastAPI app serves, so the bucket and the API cannot drift into two
contracts sharing one name.

## Sealing

A run's forecast object gains actuals for a day or two, then stops
changing. It seals when **all 47 hours are graded, or when the run turns
`SEAL_AFTER_DAYS` (4) old**.

The age backstop is required, not defensive. `grade_daily` permanently
drops any hour backed by fewer than `MIN_INTERVALS` five-minute CAISO
readings, and CAISO does not re-send telemetry. A rule of "all 47
graded" alone would leave those runs open and rewritten every morning
forever.

## Cache headers

| Object | `Cache-Control` | Rewritten |
|---|---|---|
| `runs/{t}/plants.json.gz` | `public, max-age=31536000, immutable` | never |
| `runs/{t}/totals.json` | `public, max-age=31536000, immutable` | never |
| `runs/{t}/forecast.json`, open | `public, max-age=300` | daily |
| `runs/{t}/forecast.json`, sealed | `public, max-age=31536000, immutable` | never |
| `runs.json` | `public, max-age=300` | daily |
| `regions.json`, `forecast.json`, `scoreboard.json` | `public, max-age=300` | daily |
| `plants.json.gz` | `public, max-age=86400` | daily |

**`immutable` cannot be withdrawn.** A browser that has cached an
`immutable` object will not revalidate it, so rewriting one is invisible
to every reader who already has it. If a bug ever forces sealed objects
to be re-published, change the run key — do not overwrite in place.

## Compression, and the missing header

`plants.json.gz` and `caiso/plants.json.gz` are gzip, and they carry
**no `Content-Encoding` header**.

pyarrow does not forward one to S3, at any spelling. This was tested
against the real bucket rather than assumed, because arrow's documented
behaviour is to ignore metadata keys it does not recognise without
saying so. `Cache-Control` and `Content-Type` do get through; only
`Content-Encoding` is dropped.

boto3 would set it and was rejected: it is not in the locked dependency
list, and a second S3 client beside pyarrow's is what `storage.py`'s
"No new dependency" note exists to prevent.

So the reader decompresses. In the browser that is
`DecompressionStream('gzip')`. By hand it is:

```sh
curl -s https://americast-data.s3.us-west-2.amazonaws.com/americast/public/caiso/plants.json.gz \
  | gunzip | jq '.plants | length'
```

Every other object is plain JSON and reads with `curl ... | jq` directly.

## Running it

Inside the daily job, nothing needs running by hand:

- `run_daily` stores the weather run, publishes `forecast.json`, writes
  the run directory and the static plant metadata.
- `grade_daily` re-publishes every open run and rewrites `runs.json`.

To rebuild the archive's mutable half by hand:

```sh
export AMERICAST_DATA_ROOT=s3://americast-data/americast
export AWS_DEFAULT_REGION=us-west-2
eval "$(aws configure export-credentials --profile americast --format env)"
uv run python -m americast.daily.publish
```

It prints the run count, the sealed/open split, which runs it rewrote,
and anything `verify` found. It writes only open runs and the index, so
it is safe to run repeatedly.

Note that pyarrow's S3 client runs the AWS SDK credential chain and does
**not** honour `AWS_PROFILE` the way the CLI does. Export real
credentials into the environment, as above and as CI does.

## When it looks wrong

**`verify` reports missing objects.** A run is in the index but its
three objects are not all there — usually a job that died between
writing the forecast and building the map. Re-run the publisher; `write`
fills in whichever objects are absent and leaves the rest alone.

**`verify` reports short runs.** A run object holds fewer than 47 hours,
which means the weather archive had holes when it was built. The page
will have gaps in it. Check `hrrr/manifest.csv` for that run.

**A run never seals.** Expected for at most four days. Past that, check
that `grade_daily` is running — sealing is driven by run age, so a run
older than four days that is still open means the publisher has not run
since it aged out, not that grading failed.

**The page shows a stale forecast.** Read `generated_at` on
`caiso/forecast.json`. More than about 26 hours old means the cron
failed. `generated_at` is the moment the forecast was computed and is
carried forward across rewrites on purpose; `updated_at` is when the
object was last written and moves while actuals land.
