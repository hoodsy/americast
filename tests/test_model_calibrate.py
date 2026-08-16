"""Band calibration: does it re-aim, and does it leave the median alone?

Hand-built scoreboards, so the right answer is arithmetic rather than
a property of California.
"""

import numpy as np
import pandas as pd
import pytest

from americast.model import calibrate


def scoreboard(
    residuals: list[float], ceiling: float = 10_000.0, days: int = 1
) -> pd.DataFrame:
    """A scoreboard whose residuals are exactly what is passed in.

    `error_mw` is p50 - actual, so a positive residual (the fleet beat
    the forecast) is a negative error.
    """
    n = len(residuals)
    stamps = pd.date_range("2026-06-01", periods=n, freq=f"{days * 24 / n:.6f}h", tz="UTC")
    return pd.DataFrame(
        {
            "run_time": stamps - pd.Timedelta(hours=6),
            "valid_time": stamps,
            "lead_hours": np.int32(6),
            "p10_mw": 0.0,
            "p50_mw": 5000.0,
            "p90_mw": 0.0,
            "solar_mw": [5000.0 + r for r in residuals],
            "error_mw": [-r for r in residuals],
            "inside_band": False,
            "fleet_clear_mw": ceiling,
        }
    )


def forecast(n: int = 24, ceiling: float = 10_000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "p10_mw": np.full(n, 4000.0),
            "p50_mw": np.full(n, 5000.0),
            "p90_mw": np.full(n, 6000.0),
            "fleet_clear_mw": np.full(n, ceiling),
        }
    )


# --- reading the offsets ---------------------------------------------


def test_offsets_are_the_residual_quantiles_over_the_ceiling() -> None:
    """A residual of +1000 MW on a 10 GW ceiling is +0.1."""
    residuals = list(np.linspace(-2000.0, 2000.0, 400))
    low, high = calibrate.offsets(scoreboard(residuals))
    assert low == pytest.approx(np.quantile(residuals, 0.10) / 10_000.0, rel=1e-6)
    assert high == pytest.approx(np.quantile(residuals, 0.90) / 10_000.0, rel=1e-6)


def test_offsets_are_asymmetric_when_the_error_is() -> None:
    """The whole point: a skewed error needs a lopsided band.

    A symmetric band cannot cover a distribution with a long tail on
    one side, which is what curtailment produces.
    """
    skewed = [-200.0] * 300 + [3000.0] * 100
    low, high = calibrate.offsets(scoreboard(skewed))
    assert abs(high) > abs(low) * 2


def test_too_little_history_returns_none_rather_than_a_guess() -> None:
    """Day one of a new region has no error to measure."""
    assert calibrate.offsets(scoreboard([100.0] * 10)) is None
    assert calibrate.offsets(pd.DataFrame()) is None


def test_only_the_recent_window_counts() -> None:
    """A season that ended is not evidence about the one running."""
    old = scoreboard([5000.0] * 400)
    old["valid_time"] = old["valid_time"] - pd.Timedelta(days=120)
    recent = scoreboard([100.0] * 400)
    both = pd.concat([old, recent], ignore_index=True)

    _, high = calibrate.offsets(both)
    assert high < 0.05, "the 120-day-old residuals must not reach the band"


def test_night_rows_do_not_enter_the_calibration() -> None:
    """A zero ceiling would divide a residual by nothing."""
    lit = scoreboard([500.0] * 300)
    dark = scoreboard([0.0] * 300, ceiling=0.0)
    both = pd.concat([lit, dark], ignore_index=True)
    _, high = calibrate.offsets(both)
    assert high == pytest.approx(0.05, rel=1e-6)


# --- applying them ----------------------------------------------------


def test_apply_moves_the_edges_to_the_measured_offsets() -> None:
    out = calibrate.apply(forecast(), (-0.10, 0.20))
    assert out["p10_mw"].iloc[0] == pytest.approx(5000.0 - 1000.0)
    assert out["p90_mw"].iloc[0] == pytest.approx(5000.0 + 2000.0)


def test_apply_never_touches_the_median() -> None:
    """The point forecast is not what is broken."""
    before = forecast()
    after = calibrate.apply(before, (-0.30, 0.30))
    assert (after["p50_mw"] == before["p50_mw"]).all()


def test_apply_scales_with_each_hour_s_own_ceiling() -> None:
    """A 600 MW miss is ordinary at midsummer noon and huge in December."""
    frame = forecast(n=2)
    frame.loc[0, "fleet_clear_mw"] = 20_000.0
    frame.loc[1, "fleet_clear_mw"] = 2_000.0
    out = calibrate.apply(frame, (-0.10, 0.10))
    assert (out.loc[0, "p90_mw"] - out.loc[0, "p50_mw"]) == pytest.approx(2000.0)
    assert (out.loc[1, "p90_mw"] - out.loc[1, "p50_mw"]) == pytest.approx(200.0)


def test_a_dark_hour_keeps_its_zero_band() -> None:
    """No ceiling, no band. A range around zero would be an invention."""
    frame = forecast(n=1)
    frame.loc[0, ["p10_mw", "p50_mw", "p90_mw", "fleet_clear_mw"]] = 0.0
    out = calibrate.apply(frame, (-0.20, 0.20))
    assert out.loc[0, "p10_mw"] == 0.0
    assert out.loc[0, "p90_mw"] == 0.0


def test_the_band_never_goes_negative() -> None:
    """A large low offset on a dim hour would otherwise cross zero."""
    frame = forecast(n=1)
    frame.loc[0, "p50_mw"] = 200.0
    out = calibrate.apply(frame, (-0.90, 0.10))
    assert out["p10_mw"].iloc[0] >= 0.0


def test_the_band_stays_ordered() -> None:
    out = calibrate.apply(forecast(), (0.30, -0.30))
    assert (out["p10_mw"] <= out["p50_mw"]).all()
    assert (out["p50_mw"] <= out["p90_mw"]).all()


def test_no_offsets_means_the_forecast_passes_through() -> None:
    before = forecast()
    assert calibrate.apply(before, None).equals(before)


# --- verify -----------------------------------------------------------


def test_verify_reports_what_changed() -> None:
    audit = calibrate.verify(forecast(), (-0.10, 0.20))
    assert audit["calibrated"]
    assert audit["p50_unchanged"]
    assert audit["width_before_mw"] == pytest.approx(2000.0)
    assert audit["width_after_mw"] == pytest.approx(3000.0)


def test_verify_says_so_when_it_did_nothing() -> None:
    audit = calibrate.verify(forecast(), None)
    assert not audit["calibrated"]
    assert "history" in audit["reason"]
