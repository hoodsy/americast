# Plant registry — build notes

Built 2026-08-07 from EIA-860 **2025 Early Release** (published
2026-06-09; final vintage lands September 2026) by
`python -m americast.ingest.eia860`.

Filter: state == CA, status == OP, technology == Solar
Photovoltaic. Generator-level rows aggregated to plants;
tracking type is capacity-dominant. All 928 plants carried coordinates —
nothing dropped.

## Headline numbers

| Slice | Plants | GW AC |
|---|---|---|
| California total | 928 | 23.88 |
| CISO (CAISO BA) | 788 | 21.52 |
| LADWP | 55 | 1.38 |
| IID | 27 | 0.59 |
| BANC | 53 | 0.39 |
| PACW + WALC | 5 | 0.01 |

Tracking: single-axis 20.21 GW (85%), fixed 3.24, dual-axis 0.26,
unknown 0.18. Largest plant: Topaz Solar Farm, 585.9 MW. Top counties:
Kern 6.60 GW, Riverside 4.49, Imperial 2.16.

## The capacity delta, explained

The labels (CAISO fuel mix) measure the **balancing authority**, not
the state. EIA-860 shows CISO-BA solar PV outside California:
Arizona 1.81 GW (10 plants), Nevada 0.67 GW (6 plants) → CISO total
24.0 GW. Observed all-time fuel-mix peak (2026-08-03) is 23.35 GW:

- vs CA-only CISO slice (21.52 GW): ratio 1.085 — *impossible* for
  installed capacity, proving the labels include plants beyond it.
- vs full CISO BA (24.0 GW): ratio 0.97 — physically sensible.

Residual gap sources: plants energized in 2026 (invisible to the 2025
filing until the next vintage) — CAISO has been adding roughly 2–4 GW of
solar per year.

## Open decision

Whether weather sampling should use CA-only plants (current registry) or
the full CISO fleet including AZ/NV (matches what the labels measure;
~10% of capacity). Registry filter is one line either
way; the schema already carries balancing_authority.
