# ---------------------------------------------------------------------------
# strategies/dca.py – Dollar-Cost Averaging strategy plugin
#
# This is the first (and currently only) strategy plugin.  It wraps the
# existing utility functions in backtest.py so all computation logic stays
# in one place and the plugin is a thin orchestration layer.
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

from src.backtest import compute_metrics, load_monthly_closes, simulate_dca
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register


@register
class DCAStrategy(BacktestStrategy):
    """Dollar-Cost Averaging: invest a fixed monthly amount across all basket assets."""

    @classmethod
    def get_name(cls) -> str:
        return "DCA"

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
                default=1000.0,
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
        # Merge schema defaults with caller-supplied params so that missing
        # keys always fall back to their declared default values.  This means
        # passing params={} is equivalent to using all defaults.
        schema_defaults = {p.key: p.default for p in self.get_config_schema()}
        resolved = {**schema_defaults, **params}

        monthly_investment = float(resolved['monthly_investment'])

        price_df = load_monthly_closes(base_url, filenames, df_meta)
        if price_df.empty:
            return None, None

        # Restrict to the requested date window (both bounds inclusive).
        mask = (price_df.index >= start_date) & (price_df.index <= end_date)
        price_df = price_df.loc[mask].dropna(how='all', axis=1)
        if price_df.empty:
            return None, None

        # Forward-fill short price gaps (≤3 months) to handle exchange
        # holidays or delayed data without distorting the simulation.
        price_df = price_df.ffill(limit=3)

        portfolio, total_invested = simulate_dca(price_df, monthly_investment)
        return portfolio, compute_metrics(portfolio, total_invested)
