# HRRR backfill

How the historical HRRR extraction runs, why it is built this way, and
how to restart it.

## What it collects

One parquet per model run, at `data/hrrr/hrrr_<YYYYMMDD>_<HH>z.parquet`.
Each file holds forecast hours f01-f48 for every plant in the registry —
928 plants x 48 hours = 44,544 rows.

Seven GRIB messages per forecast hour: DSWRF, VBDSF, VDDSF, TCDC,
TMP:2m, UGRD:10m and VGRD:10m. They become the ten columns of
`HRRR_WEATHER`.

Target window: 2023-01-01 to the present, for the 00z, 06z, 12z and 18z
runs. That is 1317 days x 4 runs = 5268 runs.

The work is split into three passes:

| Pass | Runs | Count | Purpose |
|---|---|---|---|
| 1 | 06z, June 2024 | 30 | Proves the schema and measures the rate |
| 2 | 06z, whole window | 1287 | Puts the full history in place quickly |
| 3 | 00z, 12z, 18z | 3951 | Fills in lead-time diversity |

Pass 1 is deliberately tiny. It is the trial month the feature work
reads, so a schema mistake surfaces in 25 minutes rather than after a
day of fetching. Pass 2 next, because it puts 3.5 years in place in
about a day instead of six.

`pending()` keys off file existence, so each pass automatically skips
what the earlier passes already stored. No bookkeeping between them.

## Why the workers outnumber the cores

Measured on 2026-08-10, before any of this was written:

| Stage | Wall time per forecast hour |
|---|---|
| `fetch` (S3 + cfgrib decode) | 8.2 - 8.7 s |
| `extract` (sample at 928 plants) | 0.20 s |

Two fetches used 23.3 s of wall time but only 2.3 s of CPU. The cost is
waiting for S3 round-trips, not computing.

Two conclusions:

1. `extract` needed no work. Herbie caches its BallTree already, so the
   nearest-gridpoint search is not the bottleneck. An earlier guess that
   it was is recorded here because the measurement is what settled it.
2. The worker count is limited by memory, not by cores.

That same benchmark also reported 406 MB per process, and that number
was wrong in the way that matters. See the next section.

## Memory: one run per worker

A worker peaks at 1.4-1.6 GB while it does one run, not the 406 MB
above. 406 MB was the peak across two fetches. A full run is 48.

Worse, the cost did not stop there. The whole fetch path leaves about
4 MB behind per forecast hour that the allocator never returns to the
OS. No single stage is at fault: decode, `extract` and `write` each
measured flat over hundreds of calls, and Python object counts stay
level. It is native heap the process keeps after allocating and freeing
~100 MB of grid arrays 48 times per run.

4 MB is invisible over the 2 fetches of the benchmark. A worker that
lives for 9 runs makes 432 fetches, and 432 x 4 MB is 1.7 GB on top of
its working set.

On 2026-08-10 that arithmetic crashed the machine. Twelve workers
reached 4.5 GB each, ~37 GB together, drove swap from nothing to 35 GB
on a 16 GB Mac, and the machine went down mid-backfill.

The fix is `max_tasks_per_child=1`: a worker does one run and is
replaced. Memory cannot compound past one run, and a fresh process costs
~3 s of imports against ~700 s of work. Measured after the change: three
workers peaked at 1503, 1627 and 1406 MB, and the three that replaced
them started at 209, 213 and 242 MB. Swap did not move. Run time per run
was unchanged at 658-670 s.

**Measure with the footprint, never with RSS.** macOS RSS excludes
compressed memory. During the crash the workers reported 63-442 MB RSS
while really holding 4.5 GB, ~94% of it compressed. RSS made a machine-
killing leak look like a healthy pool inside its budget:

```sh
top -l 1 -o mem -stats pid,command,mem,cmprs   # mem is the truth
sysctl vm.swapusage                            # whole-machine pressure
```

Also note that `pgrep -f americast.ingest.hrrr` does not match the
pool's spawn-mode children. Killing the driver leaves the workers alive
as orphans, still holding the memory. Kill the child pids directly.

## Design

```
runs()  ->  pending()  ->  ProcessPoolExecutor(4, max_tasks_per_child=1)
                                  |
                        worker: own scratch dir, own pid
                                  |
                        build() -> write() -> record
                                  |
                        driver appends manifest.csv
                                  |
                        worker exits, a fresh one takes the next run
```

One worker takes one run and calls `build()` unchanged from the pilot.
The parallel unit is the run, so `extract`, `finalize` and `write` were
not touched. A worker handles exactly one run and is then replaced; see
the memory section for why.

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
# Pass 1: the pilot month. Run the golden tests before going further.
caffeinate -i uv run python -m americast.ingest.hrrr \
    --hours 6 --start 2024-06-01 --end 2024-06-30 --workers 4

# Pass 2: 06z across the whole window.
caffeinate -i uv run python -m americast.ingest.hrrr \
    --hours 6 --start 2023-01-01 --end 2026-08-09 --workers 4

# Pass 3, later.
caffeinate -i uv run python -m americast.ingest.hrrr \
    --hours 0,12,18 --start 2023-01-01 --end 2026-08-09 --workers 4
```

Use `caffeinate -i` so the machine does not sleep. Run only one backfill
process at a time.

Four workers, not twelve: at ~1.6 GB each at peak that is ~6.4 GB, which
a 16 GB machine survives next to a browser and an editor. Raise the count
only after you watch `top -stats mem,cmprs` through a few runs.

## After a schema change, clear the store first

`pending()` treats an existing parquet as a finished run, so old-schema
files silently block their own refetch. Move them aside before pass 1:

```sh
mv data/hrrr data/hrrr_old && mkdir -p data/hrrr
```

Keep `data/hrrr_old` until the pilot month and its golden tests pass,
then delete it. The manifest goes with it; that costs nothing unless it
holds `missing` rows, which are the only entries `pending()` reads.

This happened on 2026-08-10, when VBDSF and VDDSF were added to capture
the beam/diffuse split. 248 stored runs were refetched. The lesson is
that the cheapest moment to change the schema is the earliest one — the
cost of a field decision grows with every run already fetched.

Audit what is stored:

```python
from americast.ingest.hrrr import verify
verify()   # run_time, fhours, plants, rows — one line per stored file
```

## The registry is one snapshot, and that is now handled

The registry is EIA-860 2025 Early Release, listing plants operating
now. The backfill therefore samples weather at 2025-era plant locations
for every year back to 2023, including plants that did not yet exist.

For the weather itself this is harmless — weather at a coordinate does
not depend on whether a plant stands there. It mattered for feature
building, where capacity-weighted aggregation would have weighted 2023
with 2026 capacity. That is not a small correction: only 71.2% of
today's CISO solar capacity was running at the start of 2023.

Fixed on 2026-08-10. The registry now carries `operating_date`, the
month a plant's first phase started generating, so the aggregation can
drop plants that did not exist yet. The aggregation must actually
filter on it — the column existing is not the same as the leak being
closed.

Residual: the date is per plant, not per phase, so a plant built in
stages counts at full size from its first phase. 27 plants holding
3.74 GW are staged, but only 8 (0.04 GW) stagger by more than two
years. Dating each generator separately is the fix if the residual
shows up in evaluation.
