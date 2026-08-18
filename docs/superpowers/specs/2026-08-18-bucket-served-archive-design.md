# Bucket-served archive — design

Status: approved 2026-08-18. Replaces the delivery half of
`plans/SERVE.md`, which must be rewritten when this lands. The frontend
spec is `plans/WEB.md`; the consumer handoff is `docs/web_handoff.md`.

## Why this exists

Two things were true on 2026-08-18 and neither was tolerable.

**The map had no origin.** `americast-web` reads its statewide curve
from three static objects on S3, which work. It reads its map from
`http://localhost:8000`, which nobody deployed. Every map request fails
for a visitor. The web work stalled on this, not on the S3 data.

**The archive stopped growing.** `run_daily` builds its HRRR run in
memory and never stores it, so `hrrr/` in the bucket ends at
`hrrr_20260812_06z.parquet`. The weather archive had not grown in six
days, and the next retrain would have had a hole in it.

Logan's decision: serve the whole page off the bucket, and make the
bucket accumulate daily so a visitor can browse previous forecasts.

## What this reverses, and what it does not

`plans/SERVE.md` §6 rejected static files on object storage. The
reasoning was that the verification surface in its §5 — error by lead
hour, run-to-run convergence, worst misses over a season — is a
server-side join over the run archive and always will be, so static
files would mean maintaining two delivery paths to avoid maintaining
one.

That reasoning still holds. What changed is its own stated exception:
§6 says the conclusion is "worth revisiting only if this becomes a
read-only viewer with no verification surface". That is exactly what
shipped. The viewer needs the bulk data and does not need the queries.

So: the bucket takes the bulk data now. The FastAPI app stays in the
repo, runs locally, and is never deployed. When the verification views
are built, they get the origin `SERVE.md` §5 describes, and the bulk
path does not move to it. `SERVE.md` is rewritten to record this, not
deleted — the reasoning is worth more than the conclusion.

## Decisions

| Question | Decision |
|---|---|
| How deep is the archive | Forward only, from the first published day |
| Do past runs show actuals | Yes, added when grading reaches them |
| Does the weather archive grow daily | Yes, `run_daily` stores its run |
| How is the large object delivered | gzipped, decompressed in the browser |
| What happens to the FastAPI app | Kept local, never deployed |
| How is the archive laid out | One directory per run, plus an index |

**Forward only** matters for honesty. The bucket holds 2,120 stored
runs back to 2023, and objects could be generated for all of them. But
no forecast was published for those mornings, so a generated object
would be a reconstruction made with today's model, not a record of what
we said. Those are different claims. The archive contains only runs
that were genuinely issued.

## Scope

In:

- A publisher that projects existing stores into public objects.
- `run_daily` storing its HRRR run.
- `grade_daily` re-publishing runs as actuals arrive, and sealing them.
- `storage` gaining cache headers and a gzip writer.
- `americast-web` reading every object from the bucket.
- `plans/SERVE.md` rewritten.

Out:

- Backfilling the 2,120 stored runs. Decided against above.
- A CDN. Considered and declined; S3 with correct cache headers is the
  whole delivery path.
- The verification query surface. Still `SERVE.md` §5, still needs an
  origin, still not built.
- Deploying the FastAPI app.
- Hosting `americast-web`. A separate step, unblocked by this one.

## Measurements

Taken 2026-08-18 against stored run `2025-03-24 06z`, 788 plants over
47 hours, through the real response models:

| Object | Raw | Gzipped |
|---|---|---|
| per-plant values | 396.8 KB | 52.8 KB |
| totals, state + zone + county | 28.1 KB | 6.8 KB |
| plant metadata, static | 141.7 KB | 24.8 KB |
| statewide forecast | 5.2 KB | — |

S3 does not compress on the fly, so an uncompressed page load costs
about 425 KB per run opened. Compressed, about 60 KB.

## The pyarrow constraint

Verified 2026-08-18 by writing real objects to the bucket and reading
their headers back:

- `Cache-Control` set through `open_output_stream(metadata=...)` **does**
  reach S3. The sealing plan below depends on this.
- `Content-Encoding` does **not**, at any spelling tried
  (`Content-Encoding`, `content-encoding`, `CONTENT-ENCODING`,
  `ContentEncoding`). Arrow's documented behaviour is that unsupported
  metadata keys are ignored silently, which is why this was probed
  rather than assumed.

boto3 would set it, and was rejected: it is not in the Gate 0
dependency list, and a second S3 client beside pyarrow's is what
`storage.py`'s "No new dependency" note exists to prevent.

So a compressed object is published as `.json.gz` with no
`Content-Encoding`, and the browser decompresses it explicitly through
`DecompressionStream('gzip')`. The floor for that is Safari 16.4, which
is acceptable for this audience. `curl` needs a pipe through `gunzip`,
which the handoff doc must say.

## Published layout

    public/regions.json                            catalogue
    public/caiso/forecast.json                     the newest run
    public/caiso/scoreboard.json                   rolling record
    public/caiso/plants.json.gz                    static plant metadata
    public/caiso/runs.json                         run index, newest first
    public/caiso/runs/20260818T06z/forecast.json   the curve as issued
    public/caiso/runs/20260818T06z/totals.json     zone and county curves
    public/caiso/runs/20260818T06z/plants.json.gz  per-plant values

The run key is `20260818T06z`, matching the weather file it derives
from (`hrrr_20260818_06z.parquet`) rather than inventing a second
spelling of the same instant.

**No client builds a key.** `regions.json` points at a region's
`runs.json`, and each entry in `runs.json` carries its own `path`. This
is the rule that already stops `caiso/` being hardcoded, extended one
level down. A second region, or a second run hour per day, then appears
without a frontend deploy.

### `runs.json`

```json
{
  "schema_version": 1,
  "generated_at": "2026-08-18T09:45:02+00:00",
  "region": "caiso",
  "runs": [
    {
      "run_time": "2026-08-18T06:00:00+00:00",
      "path": "caiso/runs/20260818T06z/",
      "sealed": false,
      "peak_mw": 21077.9,
      "mae_mw": null
    }
  ]
}
```

`peak_mw` and `mae_mw` are carried so a run picker can show which days
were sunny and which days the model missed, without fetching 47 run
objects to find out. `mae_mw` is null until the run is graded.

**This object grows without bound**, by about 150 bytes a run and so
about 55 KB a year. That is fine for years at a five-minute TTL, and it
is stated here rather than capped, because a silently truncated archive
reads as a complete one.

### `runs/{key}/forecast.json`

The existing `forecast.json` contract, plus three fields:

| Field | Meaning |
|---|---|
| `observed_mw` | What CAISO actually reported, parallel to `valid_times`. Null per hour until graded, and permanently null for hours that never grade |
| `error` | That run's own score: `mae_mw`, `bias_mw`, `coverage`, `graded_hours`. Null until anything is graded |
| `sealed` | True when this object will never be written again |

`error` is that run's record. `accuracy`, which the existing contract
already carries, is the rolling 30-day figure as of `updated_at`. They
are different numbers and both belong: one says how this forecast did,
the other says how the model has been doing lately.

`generated_at` keeps its meaning — when the forecast was computed, and
it never changes. `updated_at` is added for when the object was last
written, which moves while actuals land.

`public/caiso/forecast.json` is a copy of the newest run's object, so
there is one shape and one TypeScript interface, not two.

### `runs/{key}/totals.json` and `runs/{key}/plants.json.gz`

The `Totals` and `PlantFrames` models from `api/models.py`, serialized
unchanged. They are what `GET /runs/{t}/totals` and
`GET /runs/{t}/plants` return today, which is what makes the local API
and the bucket the same contract rather than two that drift.

## The publisher

New module: `src/americast/daily/publish.py`.

**What it does.** Takes one run time and writes that run's three
objects, then rewrites the index.

**Inputs.** The run time, the region, and four stores that already
exist: `live/forecasts.parquet` for the curve, `live/scores.parquet`
for the observed values, `hrrr/hrrr_*.parquet` for the weather,
`registry/` for the fleet.

**Outputs.** The objects listed above.

**Why this design.** The publisher computes nothing. Every published
object is a projection of a store, which buys three things: re-running
is safe, a sealed run can be rebuilt byte-identically when a bug is
found, and the JSON never becomes a second source of truth that can
disagree with the parquet. It calls `api.frames.frames()` and
`api.frames.totals()` rather than reimplementing them, so the local API
and the bucket cannot drift.

Functions:

| Function | Does |
|---|---|
| `curve(run_time, region)` | Builds the statewide object from the forecast and score stores |
| `write(run_time, region)` | Writes one run's three objects with the right headers |
| `catalogue(region)` | Builds `runs.json` from the same two stores |
| `refresh(region)` | Rewrites every unsealed run, then the index |
| `verify(...)` | The house standard checks, before publication |

`catalogue` reads the stores rather than listing the bucket. The store
knows every run that was published; a listing knows every object that
happens to be there.

`refresh` rewrites **every** unsealed run, not a fixed window. Because
`SEAL_AFTER_DAYS` is 4, at most four runs are ever unsealed, so the
cost is bounded without a second rule to keep in step with the first.

`refresh` does **not** rewrite `public/caiso/forecast.json`. That
object is the newest run as issued, written once by `run_daily`, and
replaced the following morning by the next run. Its actuals arrive
after it has already stopped being the newest, so patching them into it
would be work nobody reads. A visitor who wants a graded run opens it
from `runs.json`.

## The daily sequence

    1  ingest.caiso        yesterday's actuals
    2  run_daily           build the HRRR run in memory
                           hrrr.write()          NEW, about 600 KB
                           forecast, append live/forecasts.parquet
                           publish public/caiso/forecast.json
                           publish.write()       NEW, the run directory
    3  grade_daily         grade, append live/scores.parquet
                           publish scoreboard.json
                           publish.refresh()     NEW, unsealed runs + index

`hrrr.write()` must precede `publish.write()`, because the publisher
reads the stored weather file. The uncommitted `storage` seam in
`api/frames.py` and `api/app.py` is what lets it read that from a
bucket; it lands as part of this work rather than separately.

The workflow file itself does not change. The steps it already runs
gain the writes.

## Sealing

A run seals when every one of its 47 hours is graded, **or** when the
run is `SEAL_AFTER_DAYS` old. That constant is 4.

The age backstop is required, not defensive. `grade_daily` drops any
hour backed by fewer than `MIN_INTERVALS` five-minute readings, and
that hour never becomes gradeable — CAISO does not re-send telemetry.
A rule of "all 47 graded" alone would leave those runs unsealed and
rewritten every morning forever.

## Cache headers

| Object | `Cache-Control` | Rewritten |
|---|---|---|
| `runs/{t}/plants.json.gz` | `public, max-age=31536000, immutable` | never |
| `runs/{t}/totals.json` | `public, max-age=31536000, immutable` | never |
| `runs/{t}/forecast.json`, unsealed | `public, max-age=300` | daily |
| `runs/{t}/forecast.json`, sealed | `public, max-age=31536000, immutable` | never |
| `runs.json` | `public, max-age=300` | daily |
| `regions.json` | `public, max-age=300` | daily |
| `forecast.json` | `public, max-age=300` | daily |
| `scoreboard.json` | `public, max-age=300` | daily |
| `plants.json.gz` | `public, max-age=86400` | on registry rebuild |

This keeps `SERVE.md` rule 2 exactly: `immutable` goes only on an
object that will never be written again. A run's totals and per-plant
values are physics over an immutable weather file, so they are
immutable from the moment they are written. Its forecast object is not,
until it seals.

The consequence to respect: an object written with `immutable` and
later rewritten is invisible to every client that cached it. If a bug
requires re-publishing sealed objects, the run key must change, or the
readers who cached them will never see the fix.

## Changes to `storage`

`write_text(location, text, cache_control=None)` — the header is passed
through to S3 metadata, and ignored locally, exactly as Content-Type is
ignored locally today.

`write_gzip(location, text, cache_control=None)` — compresses and
writes. Content-Type is `application/gzip`, because the bytes are gzip
and no `Content-Encoding` is claiming otherwise.

Both stay inside the one seam. Nothing outside `storage.py` learns
whether the target is a disk or a bucket.

## Changes to `americast-web`

`src/api/client.ts` swaps its base and its paths for object keys, and
gains a helper that pipes a response through
`DecompressionStream('gzip')` before parsing.

The larger change is a deletion. `fetchRunNear`, the hour-offset
matching in `App.tsx`, and the `canPlot` fallback all exist because the
curve and the map came from different runs on different schedules. They
now come from the same run, published in the same job, so the offset is
always zero and the fallback is unreachable. Remove them rather than
leave them as dead branches that hide a real misalignment later.

`runs.json` makes a run picker possible, carrying each run's peak and
error. Building it is optional and not part of this spec.

## Testing

- Unit: `curve()` projects a known forecast frame and score frame into
  the expected dict; the sealing rule at each boundary; `catalogue()`
  shape and ordering.
- Golden, in the `test_golden_api.py` mould: publish a stored run to
  `tmp_path`, then assert 47 hours, every array parallel to
  `valid_times`, the gzipped object round-tripping to identical JSON,
  and `immutable` present only on sealed objects.
- Idempotence: publishing the same run twice produces identical bytes,
  matching the test `run_daily` already has.
- `verify()` and `docs/publish.md`, per the standard the `ingest/`
  package sets.
- Web: one check that the `DecompressionStream` path parses a real
  published object, rather than a fixture that was never gzipped by
  pyarrow.

## Risks

**A silently ignored header.** The mechanism behind the whole caching
table is a metadata dict that drops unknown keys without complaint.
`Content-Encoding` was already lost this way. The golden test must read
headers back from a real write, not assert what was passed in.

**The first day is a one-entry archive.** The page must render a run
index with a single run, and `runs.json` must exist from the first
publication rather than appearing on day two.

**A longer daily job.** One weather write and four object writes are
added. Expected to cost seconds against a 45-minute timeout, but the
first scheduled run after this lands should be checked, not assumed.

## Build order

Each is a gate. Stop for review at the end of each.

1. `storage`: `cache_control` on `write_text`, and `write_gzip`. Tests.
2. `publish.py`: `curve`, `write`, `catalogue`, `refresh`, `verify`,
   with tests and `docs/publish.md`.
3. `run_daily`: store the HRRR run, then call `publish.write`.
4. `grade_daily`: call `publish.refresh`.
5. Run the daily job by hand against the bucket. Read the headers back.
6. `americast-web`: read from the bucket, delete the alignment code.
7. Rewrite `plans/SERVE.md` and update `docs/web_handoff.md`.
