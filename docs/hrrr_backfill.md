# HRRR backfill (Gate 3b)

How the historical HRRR extraction runs, why it is built this way, and
how to restart it.

## What it collects

One parquet per model run, at `data/hrrr/hrrr_<YYYYMMDD>_<HH>z.parquet`.
Each file holds forecast hours f01-f48 for every plant in the registry —
928 plants x 48 hours = 44,544 rows, about 530 KB.

Target window: 2023-01-01 to the present, for the 00z, 06z, 12z and 18z
runs. That is 1317 days x 4 runs = 5268 runs, about 2.8 GB in total.

The work is split into two passes:

| Pass | Runs | Count | Purpose |
|---|---|---|---|
| 1 | 06z only | 1287 | Gives Gates 4-5 the full history quickly |
| 2 | 00z, 12z, 18z | 3951 | Fills in lead-time diversity |

Pass 1 first, because it puts 3.5 years of data in place in about a day
instead of six. A schema mistake then costs one day, not six.

## Why the workers outnumber the cores

Measured on 2026-08-10, before any of this was written:

| Stage | Wall time per forecast hour |
|---|---|
| `fetch` (S3 + cfgrib decode) | 8.2 - 8.7 s |
| `extract` (sample at 928 plants) | 0.20 s |

Two fetches used 23.3 s of wall time but only 2.3 s of CPU. The cost is
waiting for S3 round-trips, not computing. Peak memory is 406 MB per
process.

Two conclusions:

1. `extract` needed no work. Herbie caches its BallTree already, so the
   nearest-gridpoint search is not the bottleneck. An earlier guess that
   it was is recorded here because the measurement is what settled it.
2. The worker count is limited by memory, not by cores. 12 workers use
   about 4.8 GB and well under one core of real work. Twelve workers
   turn pass 1 from 34 hours into roughly 12-15 hours.

## Design

```
runs()  ->  pending()  ->  ProcessPoolExecutor(12)
                                  |
                        worker: own scratch dir, own pid
                                  |
                        build() -> write() -> record
                                  |
                        driver appends manifest.csv
```

One worker takes one run and calls `build()` unchanged from the pilot.
The parallel unit is the run, so `extract`, `finalize` and `write` were
not touched.

**Scratch isolation.** Each worker uses `data/tmp/herbie/w<pid>` and
deletes it after every run. On 2026-08-07 two processes shared one
scratch directory: Herbie returned a lazy dataset to one process while
the other unlinked the file, and the second process died with
`FileNotFoundError`. Separate directories make that impossible. It also
keeps scratch under a megabyte per worker — a shared directory would
have accumulated about 4.3 GB of `.idx` files across the backfill.

**Holes in the archive.** Some runs and some forecast hours were never
archived. `attempt()` tries a forecast hour three times with backoff. A
hole costs its own hour and nothing else: the other 47 hours are still
written. When the first three hours all fail, the run is treated as
absent and abandoned, so a missing run costs 3 tries instead of 144.

**The manifest.** `data/hrrr/manifest.csv` has one line per attempted
run. Only the driver writes it, so lines never interleave.

| status | meaning | file written |
|---|---|---|
| `ok` | all 48 forecast hours | yes |
| `partial` | 1 to 47 hours | yes |
| `missing` | nothing in the archive | no |
| `failed` | our bug, not their gap | no |

The parquet files are the truth. The manifest is the log of how they got
there. `verify()` audits the files directly and does not trust the log.

## Restarting

The job is resumable. Re-running skips any run whose parquet exists —
which also covers the 30 pilot files, written before the manifest — and
skips runs already known to be `missing`. Runs marked `partial` or
`failed` are tried again, because those usually mean a bad hour on the
network rather than absent data.

```sh
# Pass 1: 06z across the whole window.
caffeinate -i uv run python -m americast.ingest.hrrr \
    --hours 6 --start 2023-01-01 --end 2026-08-09 --workers 12

# Pass 2, later.
caffeinate -i uv run python -m americast.ingest.hrrr \
    --hours 0,12,18 --start 2023-01-01 --end 2026-08-09 --workers 12
```

Use `caffeinate -i` so the machine does not sleep. Run only one backfill
process at a time.

Audit what is stored:

```python
from americast.ingest.hrrr import verify
verify()   # run_time, fhours, plants, rows — one line per stored file
```

## Known limitation: the registry is one snapshot

The registry is EIA-860 2025 Early Release, listing plants operating
now. The backfill therefore samples weather at 2025-era plant locations
for every year back to 2023, including plants that did not yet exist.

For the weather itself this is harmless — weather at a coordinate does
not depend on whether a plant stands there. It matters at Gate 4, where
capacity-weighted aggregation would weight 2023 with 2026 capacity. The
registry has no operating-date column, so date-aware weighting needs a
re-parse of EIA-860. That is cheap and needs no new download. Flagged
here rather than fixed, because it is a Gate 4 decision.
