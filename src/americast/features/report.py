"""Gate 4 EDA: does the table describe the days it claims to describe?

Reads the training table and writes one self-contained HTML page. No
notebook, so the figures are reproducible from a command and reviewable
in a diff.

The page answers three questions. How close is the physical model to
CAISO, before any learning at all? Do the two baselines beat a naive
zero, and by how much? And what do the best and worst days actually
look like — because a mean error hides whether a model is a little
wrong every day or badly wrong on a few.

Everything is scored on daylight hours only. Night is trivially
correct for every predictor here, and including it would flatter all of
them equally while hiding the differences that matter.

Colors are the validated three-slot categorical palette; the page
commits to a light surface and paints it explicitly. The aqua slot sits
below 3:1 on that surface, so the summary table is not decoration — it
is the required relief, and every number in the figures appears there
in text.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from americast.features.baselines import DAYLIGHT_MW
from americast.features.table import load
from americast.region import CAISO_CA

REPORT_PATH = Path("data/reports/gate4.html")

# Categorical slots 1-3, light mode. Assigned in fixed order by entity,
# never by rank, so a filtered chart never repaints the survivors.
SERIES = {
    "solar_mw": ("CAISO actual", "#2a78d6"),
    "fleet_ac_mw": ("Physical model", "#eb6834"),
    "fleet_clear_mw": ("Clear-sky ceiling", "#1baf7a"),
}
PREDICTORS = {
    "fleet_ac_mw": ("Physical model", "#2a78d6"),
    "baseline_clear_sky_mw": ("Clear-sky persistence", "#eb6834"),
    "baseline_smart_mw": ("Smart persistence", "#1baf7a"),
}

# Chart chrome, light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

LEAD_BUCKETS = [0, 6, 12, 18, 24, 30, 36, 42, 48]


def graded(table: pd.DataFrame) -> pd.DataFrame:
    """Daylight rows where the label and every predictor exist.

    The comparison has to be like for like. A baseline that is null for
    its first week would otherwise be scored on an easier subset than
    the physical model, which is never null, and would look better for
    having skipped the hard days.
    """
    columns = ["solar_mw", *PREDICTORS]
    complete = table.dropna(subset=columns)
    return complete[complete["fleet_clear_mw"] > DAYLIGHT_MW].copy()


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error and bias per predictor, plus naive zero."""
    scored = []
    for column, (name, _) in PREDICTORS.items():
        error = rows[column] - rows["solar_mw"]
        scored.append({"predictor": name, "mae": error.abs().mean(), "bias": error.mean()})
    zero = -rows["solar_mw"]
    scored.append({"predictor": "Naive zero", "mae": zero.abs().mean(), "bias": zero.mean()})
    return pd.DataFrame(scored)


def by_lead(rows: pd.DataFrame) -> go.Figure:
    """Error against how far ahead the forecast reached."""
    bucketed = rows.copy()
    bucketed["bucket"] = pd.cut(bucketed["lead_hours"], LEAD_BUCKETS)

    figure = go.Figure()
    for column, (name, color) in PREDICTORS.items():
        error = (bucketed[column] - bucketed["solar_mw"]).abs()
        mae = error.groupby(bucketed["bucket"], observed=True).mean()
        labels = [f"{int(b.left) + 1}-{int(b.right)}h" for b in mae.index]
        figure.add_bar(
            x=labels,
            y=mae.to_numpy(),
            name=name,
            marker_color=color,
            marker_line_width=2,
            marker_line_color=SURFACE,
            hovertemplate="%{x}<br>%{y:,.0f} MW<extra>" + name + "</extra>",
        )
    return _chrome(figure, "Error by lead time", "Mean absolute error (MW)")


def by_hour(rows: pd.DataFrame) -> go.Figure:
    """Error against the hour of the local day."""
    figure = go.Figure()
    for column, (name, color) in PREDICTORS.items():
        error = (rows[column] - rows["solar_mw"]).abs()
        mae = error.groupby(rows["local_hour"]).mean()
        figure.add_scatter(
            x=mae.index,
            y=mae.to_numpy(),
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8, line=dict(width=2, color=SURFACE)),
            hovertemplate="%{x}:00<br>%{y:,.0f} MW<extra>" + name + "</extra>",
        )
    chart = _chrome(figure, "Error by local hour", "Mean absolute error (MW)")
    chart.update_xaxes(title_text="Hour, America/Los_Angeles", dtick=2)
    return chart


def days(table: pd.DataFrame, rows: pd.DataFrame, region=CAISO_CA) -> list[go.Figure]:
    """The best and worst day the physical model had, drawn in full.

    Local days, not UTC ones, so that a day is a day: a UTC date splits
    a California afternoon from its own morning. Only days with a full
    complement of daylight hours are eligible — the first and last days
    of the table are partial, and a partial day wins "best" on having
    fewer hours to be wrong in rather than on being right.

    One run per day, so the curve is a single forecast rather than a
    stitched-together best-of. The shortest lead available is chosen,
    which is the forecast a day-ahead product would publish.
    """
    local = rows["valid_time"].dt.tz_convert(region.timezone)
    error = (rows["fleet_ac_mw"] - rows["solar_mw"]).abs()
    per_day = error.groupby(local.dt.date).agg(["mean", "size"])
    whole = per_day[per_day["size"] >= per_day["size"].max() - 1]
    ranked = whole["mean"].sort_values()
    picked = [(ranked.index[0], "Best day"), (ranked.index[-1], "Worst day")]

    table_local = table["valid_time"].dt.tz_convert(region.timezone)
    figures = []
    for date, title in picked:
        same_day = table[table_local.dt.date == date]
        run = same_day.loc[same_day["lead_hours"].idxmin(), "run_time"]
        curve = same_day[same_day["run_time"] == run].sort_values("valid_time")
        clock = curve["valid_time"].dt.tz_convert(region.timezone)

        figure = go.Figure()
        for column, (name, color) in SERIES.items():
            figure.add_scatter(
                x=clock,
                y=curve[column],
                name=name,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{y:,.0f} MW<extra>" + name + "</extra>",
            )
        # No x-axis title: plotly stacks a date under the time ticks on
        # a day-long axis, and a title below that lands on the legend.
        chart = _chrome(figure, f"{title} — {date} (Pacific)", "Statewide solar (MW)")
        chart.update_layout(hovermode="x unified")
        figures.append(chart)
    return figures


def render(table: pd.DataFrame) -> str:
    """Build the whole page as one HTML string."""
    rows = graded(table)
    scores = summarize(rows)
    figures = [by_lead(rows), by_hour(rows), *days(table, rows)]

    blocks = []
    for index, figure in enumerate(figures):
        blocks.append(
            figure.to_html(
                full_html=False, include_plotlyjs="cdn" if index == 0 else False
            )
        )
    span = f"{table['valid_time'].min():%Y-%m-%d} to {table['valid_time'].max():%Y-%m-%d}"
    return _page(scores, blocks, span, len(rows))


def write(table: pd.DataFrame, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(table))


def _chrome(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    """One recessive, legible look for every figure on the page."""
    figure.update_layout(
        title=dict(text=title, font=dict(size=17, color=INK)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif", color=INK, size=13),
        legend=dict(orientation="h", y=-0.18, font=dict(color=INK)),
        margin=dict(t=56, b=80, l=72, r=24),
        height=420,
        bargap=0.28,
    )
    figure.update_yaxes(
        title_text=y_title,
        gridcolor=GRID,
        zerolinecolor=GRID,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED),
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=GRID,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED),
    )
    return figure


def _page(scores: pd.DataFrame, blocks: list[str], span: str, n_rows: int) -> str:
    best = scores[scores["predictor"] != "Naive zero"]["mae"].min()
    naive = scores.loc[scores["predictor"] == "Naive zero", "mae"].iloc[0]

    table_rows = "\n".join(
        f"<tr><td>{row.predictor}</td><td>{row.mae:,.0f}</td>"
        f"<td>{row.bias:+,.0f}</td></tr>"
        for row in scores.itertuples()
    )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>americast — Gate 4</title>
<style>
  body {{ background:#f9f9f7; color:{INK}; margin:0; padding:32px;
         font-family:system-ui, -apple-system, sans-serif; }}
  main {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  p.sub {{ color:{MUTED}; margin:0 0 28px; }}
  p.note {{ color:{MUTED}; font-size:13px; line-height:1.5;
            margin:-8px 0 20px; max-width:70ch; }}
  .figure {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.10);
             border-radius:8px; margin-bottom:20px; overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; background:{SURFACE};
           border:1px solid rgba(11,11,11,0.10); border-radius:8px;
           margin-bottom:28px; font-variant-numeric:tabular-nums; }}
  th, td {{ text-align:right; padding:10px 14px;
            border-bottom:1px solid {GRID}; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:{MUTED}; font-weight:600; font-size:13px; }}
  .hero {{ font-size:40px; font-weight:600; }}
</style>
<main>
  <h1>Gate 4 — features and baselines</h1>
  <p class="sub">{span} · {n_rows:,} graded daylight hours · statewide CAISO solar</p>
  <p class="hero">{best:,.0f} MW</p>
  <p class="sub">best mean absolute error, against {naive:,.0f} MW for a naive
     zero forecast. No model has been trained yet.</p>
  <table>
    <tr><th>Predictor</th><th>MAE (MW)</th><th>Bias (MW)</th></tr>
    {table_rows}
  </table>
  <div class="figure">{blocks[0]}</div>
  <p class="note">Read the lead-time chart with care while only 06z runs are
     stored. Every lead hour then lands on a fixed time of day, so the low
     buckets at 19-30h are not skill — they are the hours where California is
     barely generating. Pass 3 of the backfill adds the 00z, 12z and 18z runs,
     which is what separates lead time from time of day.</p>
  <div class="figure">{blocks[1]}</div>
  <div class="figure">{blocks[2]}</div>
  <div class="figure">{blocks[3]}</div>
</main>
"""


if __name__ == "__main__":
    built = load()
    write(built)
    print(f"wrote {REPORT_PATH}")
    print(summarize(graded(built)).round(0).to_string(index=False))
