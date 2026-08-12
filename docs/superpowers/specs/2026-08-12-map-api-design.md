# Map API — design

Status: approved 2026-08-12. Implements the read side that a separate
React repo consumes. The frontend spec lives in `plans/WEB.md`.

## Why this exists

The build plan defers the frontend until after Gate 6 and has it consume
"small JSON files: this JSON is the frozen contract the future frontend
consumes". Logan reordered that on 2026-08-12: build the consumer's
contract now, while the HRRR backfill finishes, rather than guess at it
later.

Building the consumer first is the point. Gate 6 then gets written
against a specification a real client already exercises, instead of a
schema invented in the abstract.

This repo serves the data. A separate repo renders the map in React.

## Scope

In:

- A FastAPI service, run locally, over the data already in this repo.
- Four endpoints plus a `latest` alias.
- A fix to the clear-sky ceiling, which the map's colour channel depends on.

Out:

- Hosting, CORS beyond localhost, auth, rate limits. Deferred as its own
  decision.
- Anything needing Gate 5 or Gate 6: model quantiles, accuracy scores,
  live runs. The response models leave room for quantiles; nothing
  fabricates them.
- The React app itself.

## What a "run" is

One HRRR model run, keyed by its initialisation time, one-to-one with a
file in the weather store:

    data/hrrr/hrrr_20240615_06z.parquet   ->  run_time 2024-06-15T06:00:00Z

`GET /runs/2024-06-15T06:00:00Z/plants` means "the forecast issued at
06:00 UTC on 15 June, for the 47 hours that follow".

The API is keyed on the run, not on the target date. A forecast has an
issue time, and that is what makes it gradeable — the same discipline
`run_time` / `valid_time` / `lead_hours` enforces in the training table.
A date-keyed API would silently stitch several runs together, and then
"what did we predict, at what lead time" has no answer. `/runs/latest`
exists so the client does not have to sort.

47 steps, not 48: the last forecast hour is dropped by the same hour
alignment the training table uses (below).

## Architecture

```
src/americast/api/
  app.py      FastAPI app and routes
  frames.py   builds one run's payload; owns the cache
  models.py   pydantic response models — these ARE the contract
```

```
data/hrrr/hrrr_YYYYMMDD_HHz.parquet
        |
        v  features.power.estimate  (~1.4 s, on demand)
   per-plant mw + clear_mw
        |
        +--> per-plant frames ---+
        +--> features.physical --+--> LRU cache keyed on run_time
                                 |
                                 v  FastAPI  -->  React repo
```

Nothing new is stored. The Gate 4 ruling that per-plant estimates are
computed on the fly stands: 1075 runs at per-plant resolution would be
~39M rows, and they rebuild in 1.4 seconds. First request for a run pays
that; the cache serves the rest.

New dependencies: `fastapi`, `uvicorn`, `httpx` (required by FastAPI's
TestClient). Outside the Gate 0 locked list, added by Logan's decision
on 2026-08-12.

## Hour alignment

The API applies the same trapezoid as `features.hourly()`: HRRR reports
instantaneous irradiance, the graded label is a mean over the hour, and
mixing them is wrong by a factor of 2.6 at dusk. Averaging is linear, so
applying it per plant and then summing equals summing and then
averaging.

Without this the map would display a different quantity from the
statewide forecast and the two would visibly disagree at the shoulders.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /runs` | available run times, newest first |
| `GET /plants` | 788 plants: id, name, lat, lon, capacity, county, zone |
| `GET /runs/{run_time}/plants` | per-plant `mw` and `clearness` arrays |
| `GET /runs/{run_time}/totals` | state, zone and county curves |
| `GET /runs/latest/...` | alias resolving to the newest stored run |

Plant metadata is split from plant values on purpose. Coordinates and
capacity never change, so the client fetches them once; scrubbing to
another run then pulls only numbers.

### Response shapes

```jsonc
// GET /plants
{"plants": [
  {"plant_id": 57234, "name": "Solar Star", "latitude": 34.83,
   "longitude": -118.39, "capacity_mw_ac": 585.9, "dc_capacity_mw": 747.0,
   "county": "Kern", "zone": "kern"}
]}

// GET /runs/2024-06-15T06:00:00Z/plants
{
  "run_time": "2024-06-15T06:00:00Z",
  "valid_times": ["2024-06-15T07:00:00Z", "..."],      // 47
  "plants": [
    {"plant_id": 57234,
     "mw":        [0.0, 0.0, 12.4, "..."],             // 47, >= 0
     "clearness": [null, null, 0.94, "..."]}           // 47, null before sun
  ]
}

// GET /runs/2024-06-15T06:00:00Z/totals
{
  "run_time": "2024-06-15T06:00:00Z",
  "valid_times": ["..."],
  "levels": [
    {"level": "state",  "name": "CISO", "validated": true,
     "mw": ["..."], "clear_mw": ["..."]},
    {"level": "zone",   "name": "kern", "validated": false, "...": "..."},
    {"level": "county", "name": "Kern", "validated": false, "...": "..."}
  ]
}
```

Every array in a response has the same length as `valid_times`.

### Two decisions inside the shape

**Clearness is computed server-side and is `null` below a light
threshold.** One definition of the metric in one place. It also contains
the ceiling's worst failure: the ratio is unusable at dawn and dusk, and
a plant that has not started generating is "dark", not "0% clear" —
different claims. The client renders nulls as unlit points, so a data
limit becomes a rendering rule instead of leaking to the frontend.

**Each level carries `validated`.** Only the statewide number is checked
against CAISO; county and zone are physically-derived estimates that sum
to it. Putting that in the payload makes it machine-readable rather than
a warning in a document the frontend author may never read.

## The clear-sky ceiling fix

A prerequisite, not a nicety. Colour is clearness, and clearness
currently exceeds 1.0 on **71.3% of daylight rows**, median ratio 1.059.
A map reporting "118% clear" depicts nothing.

Measured today:

| Local hour | 5 | 7 | 10-15 | 17 | 19 |
|---|---|---|---|---|---|
| median estimate / ceiling | 1.88 | 1.13 | ~1.05 | 1.11 | 1.42 |

Altitude is not the cause and was re-tested at the hours where it should
have mattered most: at 800 m the dawn ratio gets worse, 2.54 -> 2.93.
The suspect is the Linke turbidity climatology, which reads high over
clean dry air.

**The fix: scale Linke turbidity by a single fitted factor.** One number
for the fleet, chosen so the ceiling envelopes HRRR's own clear-sky
hours. Clearness is a ratio of two HRRR-derived quantities, so the
denominator should be consistent with the numerator's source — on a
clear June day HRRR's GHI runs about 10% above Ineichen's at every
midday hour.

Turbidity is scaled rather than the output irradiance, because that
stays physical: less turbidity is cleaner air, which shifts the
beam/diffuse split correctly and attenuates less at high airmass, so it
helps the shoulders as well as the level.

**No label enters the fit**, so there is nothing to leak into Gate 5's
test period. The factor is fitted against HRRR's own irradiance. The fit
is restricted to 2023-24 anyway, as a second line of defence.

Acceptance criteria:

1. On hours above the light threshold, clearness exceeds 1.0 on under 5%
   of rows.
2. On HRRR-clear hours, median clearness sits within 0.02 of 1.0.
3. The Gate 4 golden tests still pass.

Clearness is **not** capped at 1.0. Cloud enhancement is real, so a
value slightly above 1 is information; a cap would hide the bias instead
of fixing it.

This changes `features/power.py`, so it moves the training table's
`clear_mw` column and Gate 5's features too. The table must be rebuilt
after, and `docs/training_table.md` updated with the new numbers.

## What the data honestly supports

Facts that constrain any renderer, measured 2026-08-12:

| | |
|---|---|
| CISO plants modelled | 788 (21.52 GW AC) |
| Distinct coordinates | 751 |
| Distinct ~3 km HRRR cells sampled | **588** |
| Plants sharing a cell | 200, holding 53% of capacity |
| Counties with capacity | 42 |
| Median plant / largest plant | 3.0 MW / 585.9 MW |
| Capacity in the smallest half of plants | 2.9% |

Plants inside one 3 km cell receive **identical** irradiance by
construction, so their colours match exactly. That is sampling
resolution, not meteorology, and the renderer must not jitter it away.

751 dots would imply 751 independent observations; we have about 588.
The frontend spec carries this as a legend note rather than a feature.

## Error handling

- Unknown `run_time` -> 404.
- Malformed timestamp -> 422, from pydantic, free.
- A run file present but not matching `HRRR_WEATHER` is omitted from
  `/runs` rather than served, so a store mid-refetch degrades instead of
  lying. Same schema-checked pattern the golden tests already use.
- Registry missing at startup -> fail immediately and loudly, not on the
  first request.

## Testing

Matching the standard `ingest/` sets: a unit file and a golden file.

`tests/test_api.py` — TestClient against a tiny synthetic store, reusing
the fixture pattern from `test_features_table.py`:

- every array length equals `len(valid_times)`
- `/plants` count matches the registry
- 404 on an unknown run, 422 on a malformed timestamp
- a run file with the wrong schema is absent from `/runs`
- the second request for a run does not recompute (cache)
- `clearness` is null wherever the ceiling is below threshold

`tests/test_golden_api.py` — against the real store, skipped if absent:

- a real run returns 47 steps, leads 1..47
- zones and counties each sum to the state curve
- `validated` is true for state and false for everything else
- clearness meets the acceptance criteria above
- `/plants` coordinates all fall inside a California bounding box

## Build order

1. Fix the clear-sky ceiling; rebuild the training table; update
   `docs/training_table.md`.
2. `api/models.py` — the contract, with tests that pin the shapes.
3. `api/frames.py` — payload assembly and cache.
4. `api/app.py` — routes and error handling.
5. Golden tests against the real store.
6. Hand `plans/WEB.md` to the frontend repo.

Step 1 is genuinely first: everything downstream displays its output.
