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
