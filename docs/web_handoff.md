# Connecting americast-web to live data

Instructions for whoever wires the frontend to the forecast. Written to
be self-contained: you should not need to read the backend to follow it.

The repositories are siblings:

```
~/Documents/Projects/solar/americast       backend, this repo
~/Documents/Projects/solar/americast-web   React + Vite + maplibre-gl
```

---

## 1. Read this first: there are two data products, not one

This is the single thing that will waste your time if you miss it.

**This section is now history.** As of 2026-08-18 both products are
static objects in the same bucket, written by the same daily job, and
`americast-web` reads both. There is no server in the read path at all.

It is kept because the distinction still matters when reading the code:

| | **Statewide forecast** | **The map** |
|---|---|---|
| What | one 48-hour curve for all of CISO | every plant, every hour |
| Shape | 3 static JSON objects | 2 objects per run, plus an index |
| Client | `src/api/forecast.ts` | `src/api/client.ts` |
| Graded | yes, against CAISO's published hourly output | no |
| Size | ~5 KB | ~60 KB gzipped per run |

The forecast is the graded product and decides whether the page works.
The map is the detail underneath it, and a run issued before the daily
job began storing its weather has none — the page then draws bare
geography, which is honest rather than broken.

---

## 2. What is live right now

Three objects, world-readable, CORS enabled, `application/json`:

```
https://americast-data.s3.us-west-2.amazonaws.com/americast/public/regions.json
https://americast-data.s3.us-west-2.amazonaws.com/americast/public/caiso/forecast.json
https://americast-data.s3.us-west-2.amazonaws.com/americast/public/caiso/scoreboard.json
```

plus, since 2026-08-18, the map and the archive of past runs:

```
.../americast/public/caiso/runs.json                         the run index
.../americast/public/caiso/plants.json.gz                    static plant metadata
.../americast/public/caiso/runs/20260818T06z/forecast.json   one past run
.../americast/public/caiso/runs/20260818T06z/totals.json     zone and county
.../americast/public/caiso/runs/20260818T06z/plants.json.gz  per-plant
```

Refreshed once a day by a GitHub Actions cron at 09:00 UTC. No auth, no
key, no rate limit worth thinking about. `curl` one and see.

**Resolve paths from the index, never build them.** `regions.json` names
each region's `runs.json`, and every entry in `runs.json` carries its own
`path`. That is what lets a second region — or a second run hour a day —
appear without a frontend deploy.

**The `.gz` objects carry no `Content-Encoding`.** pyarrow cannot set
that header, so nothing unpacks them for you:

```sh
curl -s .../caiso/plants.json.gz | gunzip | jq '.plants | length'
```

In the browser that is `res.body.pipeThrough(new DecompressionStream('gzip'))`.
A missing object answers **403, not 404** — the bucket grants anonymous
`GetObject` but not `ListBucket`, so S3 will not confirm a key is absent
to someone who cannot list.

### `regions.json` — the index

```json
{
  "schema_version": 1,
  "generated_at": "2026-08-16T22:02:22Z",
  "regions": [
    {
      "id": "caiso", "name": "California ISO", "kind": "iso",
      "timezone": "America/Los_Angeles", "graded": true,
      "forecast": "caiso/forecast.json",
      "scoreboard": "caiso/scoreboard.json",
      "runs": "caiso/runs.json",
      "plants": "caiso/plants.json.gz"
    }
  ]
}
```

Fetch this first and resolve the rest from it. One region today; the
paths are relative to `.../public/`. Do not hardcode `caiso/` — the
whole point of the index is that a second region appears here and the UI
picks it up without a deploy.

### `forecast.json` — the product

```json
{
  "schema_version": 1,
  "region": { "id": "caiso", "name": "California ISO", "kind": "iso",
              "timezone": "America/Los_Angeles", "graded": true },
  "units": "MW", "level": "state", "validated": true,
  "run_time":     "2026-08-16T06:00:00+00:00",
  "generated_at": "2026-08-16T22:02:22+00:00",

  "valid_times":  ["2026-08-16T07:00:00+00:00", ...],   // 47 entries
  "lead_hours":   [1, 2, ...],
  "p50_mw":       [...],
  "p10_mw":       [...],
  "p90_mw":       [...],
  "physical_mw":  [...],
  "clear_sky_mw": [...],

  "peak": { "valid_time": "...", "p50_mw": 21256.0 },
  "accuracy": null
}
```

**Every array is parallel to `valid_times`.** Index `i` describes the
hour that *starts* at `valid_times[i]`, as a mean over that hour — not
an instant. Index them together; never match on a timestamp.

| Field | What to do with it |
|---|---|
| `p50_mw` | the forecast line |
| `p10_mw`, `p90_mw` | the shaded band around it |
| `physical_mw` | pure physics, no learning. Useful as a faint reference line |
| `clear_sky_mw` | the ceiling — what a cloudless day would give. Good as a backdrop showing how much sun was lost |
| `peak` | the headline number. Already computed, do not recompute |
| `accuracy` | recent track record — **can be `null`**, see §4 |

### `scoreboard.json` — the record

A rolling 30-day summary plus a daily series (`days`, `daily_mae_mw`,
`daily_coverage`, `daily_hours`), all parallel arrays. Around 2 KB. A
view that only shows the forecast never needs to fetch it.

---

## 3. How the app is wired

Both phases are done. This is what is there now.

**`src/api/forecast.ts`** owns the statewide curve: `regions.json`, then
the region's `forecast.json`. Plain `fetch`, plain JSON.

**`src/api/client.ts`** owns the map: `runs.json` for the index,
`plants.json.gz` for the static plant list, and `{entry.path}plants.json.gz`
for one run's per-plant values. The two `.gz` reads go through
`DecompressionStream`; see §2.

The map is fetched for **the forecast's own run**, found by matching
`run_time` in the index. Both halves come from one job and one morning,
so index `i` means the same hour in both and there is no offset to
carry. If a run has no map — every run issued before 2026-08-18 — the
fetch 403s, it is caught, and the page draws bare geography.

### `runs.json` — the archive

```json
{
  "schema_version": 1,
  "generated_at": "2026-08-18T03:38:28+00:00",
  "region": "caiso",
  "runs": [
    { "run_time": "2026-08-17T06:00:00+00:00",
      "path": "caiso/runs/20260817T06z/",
      "sealed": false, "peak_mw": 21077.9, "mae_mw": null }
  ]
}
```

Newest first. `peak_mw` and `mae_mw` are there so a run picker can show
which days were sunny and which the model missed without fetching every
run object. `mae_mw` is `null` until the run is graded.

`sealed` says whether the run's forecast object will ever change again.
An open run gains actuals for a day or two; a sealed one is final and is
served `immutable` for a year. If you cache, key on `path` and treat a
sealed run as permanent.

**Caching:** S3 sends an `ETag` and now a `Cache-Control` that already
says the right thing, so a plain `fetch` does the correct amount of
work. `forecast.json` and `runs.json` are five minutes; a run's map
objects are a year. Poll `runs.json` if you want to notice a new run.

---

## 4. Things that will surprise you if nobody says them

**`accuracy` can be `null`, and is thin when it is not.** It populated
on 2026-08-17, one day after grading started, and read
`{"window_days": 30, "mae_mw": 1156.6, "bias_mw": -817.7,
"coverage": 0.333, "graded_hours": 9}`.

Nine hours is not a track record. Read `graded_hours` before showing
`mae_mw`, and keep rendering "not yet graded" until it is worth a
claim — the field being present is not the same as it meaning
something. It fills in on its own.

A past run object also carries `error`, which is that run's own score
rather than the rolling window, and `null` until the run is graded.

**The band is currently under-covering, and will fix itself.** It
promises 80% and delivers about 64% until calibration kicks in, then
around 80%. If you want to caveat it in the UI for now, that is honest;
just don't hardcode the caveat, because it expires.

**`validated: true` means something specific.** There is a public hourly
truth for this number and it is graded against it. A future region may
carry `graded: false` — forecast, but unverifiable. **Do not present
those identically.** The API draws the same distinction for county and
zone levels, which are estimates that sum to the graded state total.

**Timestamps are UTC; the region carries its own timezone.** Render
local using `region.timezone`. Do not hardcode `America/Los_Angeles` —
that is exactly what breaks when region two arrives.

**47 hours, not 48.** The last forecast hour has no successor to average
with, so it is dropped. Do not treat a 47-length array as truncated.

**The 06z run covers exactly two Pacific days** — midnight today through
23:00 the day after tomorrow. That is why it is the run we publish, and
it means "today and tomorrow" is a clean slice of the array rather than
an awkward offset.

**`generated_at` vs `run_time`.** `run_time` is the weather model's
cycle; `generated_at` is when we computed it. If `generated_at` is more
than ~26 hours old, the cron has failed — worth surfacing.

**The fleet is 833 plants and 24.2 GW AC, not 788 and 21.5.** The
registry moved from a state filter to a balancing-authority filter,
which brought in the Arizona and Nevada plants inside CISO's footprint.
Anything quoting the older pair is stale, `plans/WEB.md` included.

**There are six zones now, not five.** `src/api/types.ts` declares
`Zone` as five names; Arizona's `sonoran` was added when the plant
registry moved from a state filter to a balancing-authority filter.
Fix that type before phase 2 or the map will drop a zone silently.

---

## 5. Checking your work

```sh
curl -s https://americast-data.s3.us-west-2.amazonaws.com/americast/public/caiso/forecast.json | jq '.peak, .accuracy, (.valid_times|length)'
```

Expect a peak around 20–23 GW in summer, `accuracy` null for now, and
47. If `p50_mw` is all zeros, you are looking at night — check
`valid_times`, and remember the array starts at 07:00 UTC, which is
midnight Pacific.

Sanity bounds: the CAISO fleet is about 24 GW installed, and the
observed record is 23.2 GW. Anything above that is a bug, not a sunny
day.
