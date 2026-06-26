"""Shared pytest fixtures for the test-suite.

The backtest module caches every parquet Close read process-wide
(``_read_close_series``) so each asset / FX-pair file is fetched once per
process. Tests patch ``src.backtest.pd.read_parquet`` and assert how often it is
called, so a cache surviving across tests would leak reads and make those counts
unpredictable. The autouse fixture below clears the cache around every test,
restoring full isolation.
"""

import pytest

import src.backtest as backtest


@pytest.fixture(autouse=True)
def _clear_parquet_cache():
    """Reset the process-wide parquet Close cache before and after each test."""
    backtest.clear_parquet_cache()
    yield
    backtest.clear_parquet_cache()
