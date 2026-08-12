"""Gate 5's report: did the model beat the baselines, and where did it not?

Reads the stored boosters and the training table, scores the test
period, and writes one HTML page. No notebook, so every figure is
reproducible from a command and reviewable in a diff.

Open it with `open data/reports/gate5.html`.

The page is built to be readable by somebody looking for the catch. It
leads with the number the build plan asked for, then spends most of its
length on the three places the model is weaker than that number
suggests: the lead-time axis is confounded while only 06z runs are
stored, the confidence band does not cover what it claims, and the fleet
the model was fitted to is not the fleet it was graded on.

Colors are categorical slots 1-4 of the validated palette, assigned in
fixed order by entity. Slot 3 (aqua) and slot 4 (yellow) sit below 3:1
on this surface, so the score table is not decoration — it is the
required relief, and every number in every figure appears there as text.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from americast.features.table import load
from americast.model import eval as scoring
from americast.model import model as boosters
from americast.model.split import TRAIN_END, VAL_END, graded, split
from americast.region import CAISO_CA

REPORT_PATH = Path("data/reports/gate5.html")

# Categorical slots 1-4, light mode, assigned in declaration order.
# The model takes slot 1 because it is the subject of this page; the
# three references follow in the order the build plan names them.
SERIES = {
    "p50_mw": ("Model (p50)", "#2a78d6"),
    "fleet_ac_mw": ("Physical model", "#eb6834"),
    "baseline_clear_sky_mw": ("Clear-sky persistence", "#1baf7a"),
    "baseline_smart_mw": ("Smart persistence", "#eda100"),
}

# Chart chrome, light mode. Same values as the Gate 4 page.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BAND = "rgba(42,120,214,0.18)"


def by_lead(rows: pd.DataFrame) -> go.Figure:
    """Error against how far ahead the forecast reached."""
    table = scoring.by_lead(rows)
    figure = go.Figure()
    for name, color in SERIES.values():
        part = table[table["predictor"] == name]
        labels = [f"{int(b.left) + 1}-{int(b.right)}h" for b in part["group"]]
        figure.add_bar(
            x=labels,
            y=part["mae"].to_numpy(),
            name=name,
            marker_color=color,
            marker_line_width=2,
            marker_line_color=SURFACE,
            hovertemplate="%{x}<br>%{y:,.0f} MW<extra>" + name + "</extra>",
        )
    return _chrome(figure, "Error by lead time", "Mean absolute error (MW)")


def by_hour(rows: pd.DataFrame) -> go.Figure:
    """Error against the hour of the local day."""
    table = scoring.by_hour(rows)
    figure = go.Figure()
    for name, color in SERIES.values():
        part = table[table["predictor"] == name].sort_values("group")
        figure.add_scatter(
            x=part["group"],
            y=part["mae"].to_numpy(),
            name=name,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": SURFACE}},
            hovertemplate="%{x}:00<br>%{y:,.0f} MW<extra>" + name + "</extra>",
        )
    chart = _chrome(figure, "Error by local hour", "Mean absolute error (MW)")
    chart.update_xaxes(title_text="Hour, America/Los_Angeles", dtick=2)
    return chart


def drift(parts: dict[str, pd.DataFrame]) -> go.Figure:
    """CAISO against the physics, quarter by quarter — the bias explained.

    One series, so no legend: the title names it. The reference line at
    1.0 is where the fleet delivers exactly what the physics predicts,
    and the three shaded regions are the periods, so the reader can see
    the fit period sitting below the line and the grading period sitting
    above it.
    """
    lit = pd.concat([graded(part) for part in parts.values()])
    lit = lit[lit["fleet_ac_mw"] > 500.0]
    # tz dropped deliberately: a quarter is a calendar bucket, and
    # pandas warns rather than assumes when converting an aware stamp.
    quarter = lit["valid_time"].dt.tz_localize(None).dt.to_period("Q")
    residual = (lit["solar_mw"] / lit["fleet_ac_mw"]).groupby(quarter).median()
    stamps = [q.start_time for q in residual.index]

    figure = go.Figure()
    figure.add_scatter(
        x=stamps,
        y=residual.to_numpy(),
        mode="lines+markers",
        name="CAISO / physics",
        line={"color": "#2a78d6", "width": 2},
        marker={"size": 8, "line": {"width": 2, "color": SURFACE}},
        hovertemplate="%{x|%Y Q%q}<br>%{y:.3f}<extra>CAISO / physics</extra>",
    )
    figure.add_hline(y=1.0, line={"color": MUTED, "width": 1, "dash": "dot"})

    # Boundaries as lines, not as three adjacent shaded blocks. Equal
    # shading on abutting regions merges into one grey field that says
    # nothing; a rule at each cut says exactly where the periods change.
    # The test period alone keeps a wash, because it is the one the
    # scores are computed on.
    graded_test = graded(parts["test"])
    figure.add_vrect(
        x0=graded_test["valid_time"].min(),
        x1=graded_test["valid_time"].max(),
        fillcolor=INK,
        opacity=0.04,
        line_width=0,
    )
    for edge, label in ((TRAIN_END, "validate"), (VAL_END, "test")):
        figure.add_vline(
            x=edge,
            line={"color": MUTED, "width": 1},
            annotation_text=f"{label} →",
            annotation_position="top right",
            annotation_font={"color": MUTED, "size": 12},
        )
    figure.add_annotation(
        x=graded(parts["train"])["valid_time"].min(),
        y=1.0,
        yref="paper",
        text="← train",
        showarrow=False,
        xanchor="left",
        font={"color": MUTED, "size": 12},
    )
    chart = _chrome(
        figure,
        "What CAISO delivered, against what the physics predicted",
        "Median ratio (1.0 = the physics is right)",
    )
    chart.update_layout(showlegend=False)
    return chart


def days(rows: pd.DataFrame, region=CAISO_CA) -> list[go.Figure]:
    """The model's best and worst test day, with the band drawn.

    One run per day, at the shortest lead available, so the curve is a
    single forecast rather than a stitched-together best-of. The band is
    what makes these worth plotting: a bad day where the truth stays
    inside p10-p90 is a model that knew it was uncertain, which is a
    different failure from one that was confidently wrong.
    """
    ranked = scoring.days(rows, region.timezone)
    picked = [(ranked.index[0], "Best day"), (ranked.index[-1], "Worst day")]
    local = rows["valid_time"].dt.tz_convert(region.timezone)

    figures = []
    for date, title in picked:
        same_day = rows[local.dt.date == date]
        run = same_day.loc[same_day["lead_hours"].idxmin(), "run_time"]
        curve = same_day[same_day["run_time"] == run].sort_values("valid_time")
        clock = curve["valid_time"].dt.tz_convert(region.timezone)

        figure = go.Figure()
        figure.add_scatter(
            x=clock, y=curve["p90_mw"], mode="lines", line={"width": 0},
            name="p90", hoverinfo="skip", showlegend=False,
        )
        figure.add_scatter(
            x=clock, y=curve["p10_mw"], mode="lines", line={"width": 0},
            fill="tonexty", fillcolor=BAND, name="p10-p90 band",
            hovertemplate="%{y:,.0f} MW<extra>p10</extra>",
        )
        figure.add_scatter(
            x=clock, y=curve["p50_mw"], mode="lines", name="Model (p50)",
            line={"color": "#2a78d6", "width": 2},
            hovertemplate="%{y:,.0f} MW<extra>Model (p50)</extra>",
        )
        figure.add_scatter(
            x=clock, y=curve["solar_mw"], mode="lines", name="CAISO actual",
            line={"color": INK, "width": 2},
            hovertemplate="%{y:,.0f} MW<extra>CAISO actual</extra>",
        )
        error = (curve["p50_mw"] - curve["solar_mw"]).abs().mean()
        chart = _chrome(
            figure,
            f"{title} — {date} (Pacific) · {error:,.0f} MW mean error",
            "Statewide solar (MW)",
        )
        chart.update_layout(hovermode="x unified")
        figures.append(chart)
    return figures


def render(rows: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> str:
    """Build the whole page as one HTML string."""
    audit = scoring.verify(rows)
    figures = [by_lead(rows), by_hour(rows), *days(rows), drift(parts)]

    blocks = []
    for index, figure in enumerate(figures):
        blocks.append(
            figure.to_html(full_html=False, include_plotlyjs="cdn" if index == 0 else False)
        )
    return _page(audit, scoring.drift(parts), blocks)


def write(rows: pd.DataFrame, parts: dict[str, pd.DataFrame], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(rows, parts))


def _chrome(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    """One recessive, legible look for every figure on the page."""
    figure.update_layout(
        title={"text": title, "font": {"size": 17, "color": INK}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, sans-serif", "color": INK, "size": 13},
        legend={"orientation": "h", "y": -0.18, "font": {"color": INK}},
        margin={"t": 56, "b": 80, "l": 72, "r": 24},
        height=420,
        bargap=0.28,
    )
    figure.update_yaxes(
        title_text=y_title,
        gridcolor=GRID,
        zerolinecolor=GRID,
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
        # Wide tick labels otherwise push the axis title past the fixed
        # left margin and it is clipped by the figure's own border.
        automargin=True,
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=GRID,
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
    )
    return figure


def _page(audit: dict, drifted: pd.DataFrame, blocks: list[str]) -> str:
    scores = audit["scores"]
    skill = audit["skill_vs_clear_sky"]
    criterion = audit["criterion"]
    band = audit["coverage"]
    confound = audit["confounded"]

    model_mae = scores.loc[scores["column"] == "p50_mw", "mae"].iloc[0]
    verdict = "MET" if criterion["passed"] else "NOT MET"
    verdict_color = "#1baf7a" if criterion["passed"] else "#e34948"

    score_rows = "\n".join(
        f"<tr><td>{row.predictor}</td><td>{row.mae:,.0f}</td><td>{row.rmse:,.0f}</td>"
        f"<td>{row.bias:+,.0f}</td>"
        f"<td>{skill.loc[row.predictor, 'mae_skill']:+.3f}</td></tr>"
        for row in scores.itertuples()
    )
    drift_rows = "\n".join(
        f"<tr><td>{row.period}</td><td>{row.start:%Y-%m-%d}</td><td>{row.end:%Y-%m-%d}</td>"
        f"<td>{row.residual_median:.4f}</td><td>{row.n:,}</td></tr>"
        for row in drifted.itertuples()
    )
    span = f"{audit['span'][0]:%Y-%m-%d} to {audit['span'][1]:%Y-%m-%d}"

    return f"""<!doctype html>
<meta charset="utf-8">
<title>americast — Gate 5</title>
<style>
  body {{ background:#f9f9f7; color:{INK}; margin:0; padding:32px;
         font-family:system-ui, -apple-system, sans-serif; }}
  main {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:36px 0 10px; }}
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
  .hero {{ font-size:40px; font-weight:600; margin:0; }}
  .verdict {{ display:inline-block; padding:3px 10px; border-radius:999px;
              font-size:13px; font-weight:600; color:#fff;
              background:{verdict_color}; }}
</style>
<main>
  <h1>Gate 5 — model and honest evaluation</h1>
  <p class="sub">Test period {span} · {audit['n_rows']:,} graded daylight hours ·
     statewide CAISO solar</p>

  <p class="hero">{model_mae:,.0f} MW</p>
  <p class="sub">mean absolute error, against {criterion['reference_mae']:,.0f} MW for
     clear-sky persistence — {skill.loc['Model (p50)', 'mae_skill']:.1%} skill.
     Exit criterion <span class="verdict">{verdict}</span>: the model wins
     {criterion['buckets_won']} of {criterion['buckets_total']} lead buckets at
     {criterion['lead_floor']}h and beyond.</p>

  <h2>Every predictor, on identical rows</h2>
  <table>
    <tr><th>Predictor</th><th>MAE (MW)</th><th>RMSE (MW)</th><th>Bias (MW)</th>
        <th>MAE skill</th></tr>
    {score_rows}
  </table>
  <p class="note">Skill is against clear-sky persistence: 1 − error/reference. The
     physical model is the honest bar — an unfitted calculation carrying no learned
     parameters — and the gap between it and the model is what three years of
     training bought.</p>

  <h2>Where the model wins</h2>
  <div class="figure">{blocks[0]}</div>
  <p class="note"><strong>Read the lead-time axis with care.</strong> Only the 06z
     run is stored, so each lead hour reaches just
     {confound['local_hours_per_lead']:.2f} distinct local hours — and those two are
     one hour apart, chosen by the daylight-saving calendar rather than by anything
     the forecast knows. Lead time and time of day are therefore the same axis here,
     and this chart is close to a re-lettered copy of the next one. A bucket that
     looks easy is a bucket that falls at night. Pass 3 of the backfill adds the
     00z, 12z and 18z runs, which is what separates them.</p>
  <div class="figure">{blocks[1]}</div>

  <h2>Best and worst day</h2>
  <div class="figure">{blocks[2]}</div>
  <div class="figure">{blocks[3]}</div>

  <h2>The band does not cover what it claims</h2>
  <p class="note">The p10–p90 band should hold the truth
     {band['nominal']:.0%} of the time. It holds it
     <strong>{band['coverage']:.1%}</strong> of the time, and it fails
     asymmetrically: {band['below_p10']:.1%} of hours fall below p10 against a
     nominal 10%, but {band['above_p90']:.1%} fall above p90. The band is not
     merely narrow — it sits too low. Mean width is
     {band['width_mw']:,.0f} MW. The cause is the drift below, and the fix is not
     a wider band.</p>

  <h2>The fleet was not the same fleet</h2>
  <div class="figure">{blocks[4]}</div>
  <table>
    <tr><th>Period</th><th>Start</th><th>End</th>
        <th>CAISO / physics (median)</th><th>Hours</th></tr>
    {drift_rows}
  </table>
  <p class="note">With the weather divided out, CAISO delivered
     {drifted.loc[0, 'residual_median']:.3f}× the physics during training and
     {drifted.loc[2, 'residual_median']:.3f}× during the test period. The model
     learned the first number and was graded against the second, which is the whole
     of its {scores.loc[scores['column'] == 'p50_mw', 'bias'].iloc[0]:+,.0f} MW bias
     — the unfitted physics, having learned nothing, is nearly unbiased instead.
     The registry's newest plant is dated 2025-12, so every plant commissioned
     during the test period generates real megawatts and contributes no ceiling.
     This is a stale input, not a modelling error, and no amount of retuning
     fixes it.</p>
</main>
"""


if __name__ == "__main__":
    parts = split(load())
    models, _ = boosters.load()
    test = boosters.attach(models, graded(parts["test"]))
    write(test, parts)
    print(f"wrote {REPORT_PATH}")
