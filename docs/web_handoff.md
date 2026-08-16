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

| | **Statewide forecast** | **The map** |
|---|---|---|
| What | one 48-hour curve for all of CAISO | every plant, every hour |
| Shape | 3 static JSON files | a query API |
| Where | **live on S3 now** | `americast/api/`, **local only** |
| Infrastructure | none — plain HTTPS | needs a server deployed |
| Size | ~5 KB | ~110 KB gzipped per run |

**`americast-web` is currently built against the map API** — `src/api/client.ts`
points at `http://localhost:8000` and fetches `/runs`, `/plants`,
`/runs/{t}/plants`, `/runs/{t}/totals`.

Nothing is wrong with that. But the statewide forecast is live today and
needs no server, and the map needs an origin nobody has deployed yet. So
do them in that order.

---

## 2. What is live right now

Three objects, world-readable, CORS enabled, `application/json`:

```
https://americast-data.s3.us-west-2.amazonaws.com/americast/public/regions.json
https://americast-data.s3.us-west-2.amazonaws.com/americast/public/caiso/forecast.json
https://americast-data.s3.us-west-2.amazonaws.com/americast/public/caiso/scoreboard.json
```

Refreshed once a day by a GitHub Actions cron at 09:00 UTC. No auth, no
key, no rate limit worth thinking about. `curl` one and see.

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
      "scoreboard": "caiso/scoreboard.json"
    }
  ]
}
```

Fetch this first and resolve the other two from it. One region today;
the paths are relative to `.../public/`. Do not hardcode `caiso/` —
the whole point of the index is that a second region appears here and
the UI picks it up without a deploy.

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

## 3. Suggested plan

**Phase 1 — the statewide forecast, no backend.** Add a second client
alongside the existing one; do not modify `src/api/client.ts`, which
serves the map. Something like `src/api/forecast.ts`:

```ts
const PUBLIC_BASE =
  import.meta.env.VITE_FORECAST_BASE ??
  'https://americast-data.s3.us-west-2.amazonaws.com/americast/public';

export async function fetchRegions(): Promise<RegionsResponse> {
  return (await fetch(`${PUBLIC_BASE}/regions.json`)).json();
}
export async function fetchForecast(path: string): Promise<ForecastResponse> {
  return (await fetch(`${PUBLIC_BASE}/${path}`)).json();
}
```

Then a view: headline peak, the p50 line, the p10–p90 band, and the
accuracy strip. That is a complete, honest product with no server.

**Phase 2 — the map.** Needs `americast/api/` (FastAPI) deployed. Read
`docs/superpowers/specs/2026-08-12-map-api-design.md` for the contract
and `plans/SERVE.md` for how it should be served — the short version is
that a run is immutable, so per-run URLs get a one-year `immutable`
cache header and a CDN, while `/runs` and `/runs/latest/*` get 60
seconds. Do not put `immutable` on `latest`.

**Caching for phase 1:** S3 sends an `ETag`. `forecast.json` changes
once a day, so a 5–10 minute client TTL plus a conditional refetch is
plenty. Poll `regions.json` if you want to notice a new run.

---

## 4. Things that will surprise you if nobody says them

**`accuracy` is `null` for the first month.** The band is calibrated
from 30 days of graded history, and grading only started 2026-08-16.
Render "not yet graded", not a zero. It populates itself.

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
