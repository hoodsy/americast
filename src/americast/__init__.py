"""Statewide day-ahead solar generation forecasting."""

import pandas as pd

# gridstatus caps pandas <3, so we opt into pandas-3 semantics on 2.x:
# the eventual upgrade is then just a version bump.
pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True
