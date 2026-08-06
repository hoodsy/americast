import pandas as pd

import americast


def test_package_imports() -> None:
    assert americast.__doc__ is not None


def test_pandas_future_flags_enabled() -> None:
    assert pd.get_option("mode.copy_on_write") is True
    assert pd.get_option("future.infer_string") is True
