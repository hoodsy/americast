# americast

Statewide day-ahead solar generation forecast for the California grid (CAISO),
built to extend to other US regions later. A tree model on weather-model
features, graded daily against published actuals.

**Scope:** utility-scale solar generation only. Rooftop (behind-the-meter)
solar is explicitly out of scope — the target is CAISO's reported
utility-scale solar output, hourly, in MW.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```sh
uv sync
uv run pytest
```
