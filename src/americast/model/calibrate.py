"""Aim the confidence band using how wrong it recently was.

The three boosters produce a band that does not cover what it claims:
63.6% of hours land inside a p10-p90 range that promises 80%. This
module fixes that, and the reason it works is worth stating, because
the obvious repair is the wrong one.

## The band was never too narrow

Measured on the test period, a correctly-centred band needs about
3,200 MW to reach 80% coverage. The trained band is 4,000 MW — a
quarter wider — and still misses. Width was never the problem.

## What is actually wrong: the seasons do not match

Uncertainty in this system is strongly seasonal. Spring is when
California curtails the most solar, so spring forecasts are wrong in a
different way, and by a different amount, than summer ones. The trained
band inherits whichever season the fit saw.

That shows up as a clean reversal. On the validation period, February
to May, 18.8% of hours fall **below** the band. On the test period, May
to August, 22.5% fall **above** it. The same band is too high in one
season and too low in the next.

Two things follow, and both were measured rather than assumed:

- **Training the tails harder makes it worse.** Forcing 600 rounds
  instead of early stopping tightened the band to 1,669 MW and dropped
  coverage to 46.6%. A tighter fit to the wrong season is worse than a
  loose one. Early stopping was not a bug; it was the only thing
  keeping the band wide enough to be nearly useful.
- **Calibrating once, on validation, overshoots.** It lands at 91%
  coverage, because it corrects a spring band toward spring and applies
  it to summer.

## The fix: recalibrate continuously

Take the residuals from the last 30 graded days, divide each by the
clear-sky ceiling so seasons are comparable, and read off the 10th and
90th percentiles. The band becomes `p50 + q * clear_mw`.

Nothing looks forward. Every day is calibrated from days already
graded, which is precisely what the daily loop can do — `grade_daily`
writes those residuals every morning. As the season turns, the
calibration follows it without anyone deciding to intervene.

Walk-forward over the test period, calibrating each day from the 30
before it:

| window | coverage | below p10 | above p90 | width |
|---|---|---|---|---|
| 14 days | 78.4% | 10.0% | 11.5% | 3,673 MW |
| **30 days** | **79.6%** | 8.4% | 12.0% | **3,877 MW** |
| 60 days | 82.0% | 7.2% | 10.7% | 4,239 MW |

Thirty days reaches nominal coverage with a **narrower** band than the
uncalibrated one. Fourteen is jumpier and sixty lags the season.

## What this does not touch

The point forecast. `p50` is unchanged, and so is the 1,039 MW MAE.
This moves the two edges of the band and nothing else.
"""

import numpy as np
import pandas as pd

# The trailing window, in days. Thirty is the measured optimum above:
# long enough to hold several hundred graded hours, short enough to
# follow a season rather than average across one.
WINDOW_DAYS = 30

# The band's nominal edges. These must match model.QUANTILES, or the
# calibration would be correcting toward a different promise than the
# one the contract publishes.
LOW, HIGH = 0.10, 0.90

# Below this many graded hours the quantiles are noise. A fortnight of
# daylight hours is roughly 200, so this refuses to calibrate on less
# than about a week.
MIN_HOURS = 150


def offsets(
    scores: pd.DataFrame,
    days: int = WINDOW_DAYS,
    low: float = LOW,
    high: float = HIGH,
) -> tuple[float, float] | None:
    """Relative band edges from recently graded hours, or None.

    Returns `(low, high)` as fractions of the clear-sky ceiling, to be
    multiplied by each hour's own ceiling. Relative rather than
    absolute because the error scales with what the fleet could make: a
    600 MW miss is ordinary at midsummer noon and enormous in December.

    Returns None when there is not enough history, which is the honest
    state for the first weeks of a new region. The caller publishes the
    uncalibrated band and says so, rather than inventing an offset.
    """
    if scores.empty:
        return None
    recent = scores[
        scores["valid_time"] > scores["valid_time"].max() - pd.Timedelta(days=days)
    ]
    lit = recent[recent["fleet_clear_mw"] > 0.0]
    if len(lit) < MIN_HOURS:
        return None

    # error_mw is p50 - actual, so the residual is its negative.
    relative = -lit["error_mw"] / lit["fleet_clear_mw"]
    return float(np.quantile(relative, low)), float(np.quantile(relative, high))


def apply(frame: pd.DataFrame, band: tuple[float, float] | None) -> pd.DataFrame:
    """Re-aim p10 and p90 around the existing p50.

    The median is not touched. Neither is any row where the ceiling is
    zero — the sun is down, every prediction is already zero, and a
    band around zero would be an invention.

    The result is clipped at zero and re-sorted, the same two physical
    guards `model.predict` applies, because a calibration offset can
    push a low edge below zero on a dim hour.
    """
    if band is None:
        return frame
    low, high = band

    out = frame.copy()
    ceiling = out["fleet_clear_mw"].to_numpy()
    lit = ceiling > 0.0

    edges = np.column_stack(
        [
            np.where(lit, out["p50_mw"] + low * ceiling, out["p10_mw"]),
            out["p50_mw"],
            np.where(lit, out["p50_mw"] + high * ceiling, out["p90_mw"]),
        ]
    )
    edges = np.clip(np.sort(edges, axis=1), 0.0, None)
    out["p10_mw"], out["p50_mw"], out["p90_mw"] = edges[:, 0], edges[:, 1], edges[:, 2]
    return out


def verify(frame: pd.DataFrame, band: tuple[float, float] | None) -> dict:
    """What the calibration did, for the record and for the report."""
    if band is None:
        return {"calibrated": False, "reason": "not enough graded history"}
    before = (frame["p90_mw"] - frame["p10_mw"]).mean()
    after = apply(frame, band)
    return {
        "calibrated": True,
        "low_offset": band[0],
        "high_offset": band[1],
        "window_days": WINDOW_DAYS,
        "width_before_mw": float(before),
        "width_after_mw": float((after["p90_mw"] - after["p10_mw"]).mean()),
        "p50_unchanged": bool(np.allclose(after["p50_mw"], frame["p50_mw"])),
    }
