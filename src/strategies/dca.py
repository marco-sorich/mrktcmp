# ---------------------------------------------------------------------------
# strategies/dca.py – Dollar-Cost Averaging strategy plugin
#
# This is the first (and currently only) strategy plugin.  It wraps the
# existing utility functions in backtest.py so all computation logic stays
# in one place and the plugin is a thin orchestration layer.
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

from src.backtest import (
    MONTHLY_INVESTMENT,
    _window_by_month,
    compute_metrics,
    load_daily_closes,
    simulate_dca,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register


@register
class DCAStrategy(BacktestStrategy):
    """Dollar-Cost Averaging: invest a fixed monthly amount across all basket assets."""

    @classmethod
    def get_name(cls) -> str:
        return "DCA"

    @classmethod
    def get_icon(cls) -> str:
        return "bi-calendar-month"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Dollar-Cost Averaging: invest a fixed amount each month, "
            "split equally across all basket assets."
        )

    @classmethod
    def get_config_schema(cls) -> list[ConfigParam]:
        return [
            ConfigParam(
                key='monthly_investment',
                label='Monthly Investment (€)',
                type='float',
                # Share the module constant from backtest.py so both code paths
                # always start from the same default value.
                default=float(MONTHLY_INVESTMENT),
                min_value=1.0,
                max_value=1_000_000.0,
            ),
        ]

    def run(
        self,
        base_url: str,
        filenames: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        df_meta: pd.DataFrame,
        params: dict[str, int | float | str],
    ) -> tuple[pd.Series | None, dict[str, str] | None]:
        # resolve_params merges caller-supplied values with schema defaults so
        # that passing params={} is equivalent to using all declared defaults.
        resolved = self.resolve_params(params)

        monthly_investment = float(resolved['monthly_investment'])

        # Load the full daily history, then keep only the calendar months of
        # the requested window so the chosen start/end *months* are fully
        # included regardless of exact trading-day boundaries.
        price_df = load_daily_closes(base_url, filenames, df_meta)
        if price_df.empty:
            return None, None

        price_df = _window_by_month(price_df, start_date, end_date)
        if price_df.empty:
            return None, None

        # Forward-fill short price gaps (≤5 trading days, ≈ one week) to handle
        # weekends, exchange holidays or delayed data without distorting the
        # simulation or carrying delisted assets indefinitely.
        price_df = price_df.ffill(limit=5)

        portfolio, total_invested = simulate_dca(price_df, monthly_investment)
        metrics = compute_metrics(portfolio, total_invested)

        # compute_metrics returns {} when portfolio is too short (< 3 months)
        # or total_invested is zero.  Treat that as a failure so callers always
        # receive either a fully-populated result or (None, None).
        if not metrics:
            return None, None

        return portfolio, metrics
