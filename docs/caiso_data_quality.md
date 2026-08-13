# CAISO solar store — data quality notes

QA run 2026-08-06 over `data/caiso/solar_5min.parquet`
(2023-01-01 → 2026-08-05, 377,853 rows). Checks: 5-minute grid gaps,
DST interval counts, large negatives, frozen values
(`americast.ingest.caiso.qa_report`).

## Real feed gaps (missing from CAISO's published history)

| Local day | Missing intervals | Notes |
|---|---|---|
| 2024-01-11 | 113 (~9.4 h) | largest outage in the window |
| 2024-06-27 | 33 (21:15–23:55) | evening, solar ≈ 0 |
| 2024-08-29 | 34 | |
| 2024-09-24 | 34 | |

Total: 214 genuinely missing intervals in 3.6 years → 99.98% complete.
`to_hourly` surfaces these as `n_intervals < 12`; the training table
excludes incomplete hours rather than interpolating them.

## DST conventions (expected, not data loss)

- Spring-forward days (2023-03-12, 2024-03-10, 2025-03-09, 2026-03-08):
  276 intervals — correct 23-hour days, UTC grid unbroken.
- Fall-back days (2023-11-05, 2024-11-03, 2025-11-02): the history CSV
  publishes 288 rows, not 300 — the repeated 1–2 AM local hour appears
  once. Shows up as 12 "missing" UTC night intervals per fall-back day
  (solar ≈ 0; immaterial, but systematic).

## Value checks

- Large negatives: 7 rows in −104…−321 MW, all night/dusk hours —
  station service draw, kept as-is (schema documents this).
- Frozen runs (same positive value ≥ 1 h): 3,930 rows, **all** small
  night/dawn plateaus; zero daytime occurrences above 1 GW. Benign.

## Known edge

The store's last day trails "now" by a few hours: CAISO's history CSV for
the current day is published with a lag. Harmless — backfill always
refetches the most recent stored day, and the daily forecast loop reads
the live feed instead.

## Curtailment: the sunlight the grid refused

`data/caiso/curtailment_hourly.parquet` — hourly solar curtailment,
2023-01-01 to present, 19,557 rows, 13,652 GWh withheld in total.

The fuel mix measures what the grid *accepted*. The physical model
estimates what the panels *could produce*. Curtailment is the
difference, and it is an economic decision no irradiance model can see.

```sh
uv run python -m americast.ingest.curtailment 2023-01-01 2026-08-12
```

Two report formats, and two traps in them.

**gridstatus returns the legacy columns as strings.** Summing object
dtype concatenates digits instead of adding them, so `"17"` and `"8"`
become 178. That produced 30,251 MW of curtailment for one July day —
more than the whole fleet, which is the only reason it was caught.
`verify()` now fails any hour above 30 GW.

**The peak-MW column is not comparable across the formats.** The legacy
report sometimes gives an hour's energy with a blank power reading, so
summing categories yields an hour whose "peak" sits below its own mean.
Only the energy figure is stored; it is a mean power over a one-hour
interval, directly addable to the hourly label.

### What it showed, and what it did not

Adding curtailment back to the label makes the unfitted physical model
**almost exactly right in 2023**: median `(solar + curtailed) / physics`
is **1.0021**. That is a strong independent validation of the whole
per-plant chain.

It does **not** explain the drift. The residual climbs +8.0% from 2023
to 2025 whether curtailment is added back or not — the correction moves
the level, never the slope. Curtailment runs at a near-constant 5.7-6.2%
of predicted output across 2023-2025.

| Year | resid | resid + curtailment | curtailed share |
|---|---|---|---|
| 2023 | 0.9469 | **1.0021** | 5.7% |
| 2024 | 0.9882 | 1.0531 | 6.1% |
| 2025 | 1.0229 | 1.0824 | 6.2% |

So the physics was right in 2023 and under-predicts by 8% by 2025. See
`model.md` for what is still unexplained.
