# ---------------------------------------------------------------------------
# strategies/riskoff.py – Risk-Off signal strategy plugin
#
# A tactical asset-allocation strategy.  Unlike DCA there is no recurring
# savings rate: a single lump sum is provided as cash at the start and is
# shifted between the basket and cash each month according to three market
# signals evaluated on the basket as a whole (see backtest.py for the maths).
# This plugin is a thin orchestration layer over the pure functions in
# backtest.py so all computation logic stays in one place.
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

from src.backtest import (
    INITIAL_INVESTMENT,
    build_equal_weight_index,
    compute_metrics,
    compute_riskoff_signals,
    load_daily_closes,
    load_monthly_closes,
    simulate_riskoff,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register


@register
class RiskOffStrategy(BacktestStrategy):
    """Risk-Off: invest a lump sum tactically, shifting to cash on weak signals."""

    @classmethod
    def get_name(cls) -> str:
        return "Risk-Off Signale"

    @classmethod
    def get_icon(cls) -> str:
        return "bi-shield-check"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Risk-Off Signale: invest a one-off lump sum and shift between the "
            "basket and cash each month based on three market signals (200-day "
            "trend, year-to-date return, January barometer)."
        )

    @classmethod
    def get_config_schema(cls) -> list[ConfigParam]:
        return [
            ConfigParam(
                key='initial_investment',
                label='Initial Investment (€)',
                type='float',
                # Share the module constant so both code paths start from the
                # same default lump-sum value.
                default=float(INITIAL_INVESTMENT),
                min_value=1.0,
                max_value=1_000_000_000.0,
            ),
            ConfigParam(
                key='sma_window',
                label='Trend SMA Window (trading days)',
                type='int',
                default=200,
                min_value=2,
                max_value=500,
            ),
            ConfigParam(
                key='first_n_days',
                label='January Barometer (trading days)',
                type='int',
                default=10,
                min_value=1,
                max_value=60,
            ),
        ]

    @staticmethod
    def _fraction_at(signals: pd.Series, month_end: pd.Timestamp) -> float:
        """Target invested fraction at *month_end* = positive signals / 3.

        Uses Series.asof to pick the most recent signal at or before the
        month-end (so only information available by then is used – no
        look-ahead).  Before any signal history exists the value is NaN, which
        we map to 0.0 (fully in cash) as a conservative default.
        """
        value = signals.asof(month_end)
        if pd.isna(value):
            return 0.0
        return float(value) / 3.0

    def run(
        self,
        base_url: str,
        filenames: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        df_meta: pd.DataFrame,
        params: dict[str, int | float | str],
    ) -> tuple[pd.Series | None, dict[str, str] | None]:
        # Merge caller-supplied values with schema defaults.
        resolved = self.resolve_params(params)

        # Defensive casts: values round-tripped through dcc.Store/JSON can arrive
        # as floats, but rolling(window=...) and head(n) require real ints.
        initial_investment = float(resolved['initial_investment'])
        sma_window = int(resolved['sma_window'])
        first_n_days = int(resolved['first_n_days'])

        # 1. Full daily history (no window) → equal-weight index → signals.
        #    The extra history gives the long look-back signals enough warm-up.
        daily_df = load_daily_closes(base_url, filenames, df_meta)
        if daily_df.empty:
            return None, None

        index = build_equal_weight_index(daily_df)
        signals = compute_riskoff_signals(index, sma_window, first_n_days)

        # Normalise signal timestamps to midnight so Series.asof aligns cleanly
        # with the month-end labels (which are at midnight) and never picks the
        # previous day because of an intraday timestamp.
        assert isinstance(signals.index, pd.DatetimeIndex)
        signals.index = signals.index.normalize()

        # 2. Monthly closes restricted to the requested window (bounds inclusive).
        price_df = load_monthly_closes(base_url, filenames, df_meta)
        if price_df.empty:
            return None, None

        mask = (price_df.index >= start_date) & (price_df.index <= end_date)
        price_df = price_df.loc[mask].dropna(how='all', axis=1)
        if price_df.empty:
            return None, None

        # Forward-fill short price gaps (≤3 months), as in the DCA path.
        price_df = price_df.ffill(limit=3)

        # 3. Target invested fraction per month-end = positive signals / 3.
        inv_frac = pd.Series(
            [self._fraction_at(signals, me) for me in price_df.index],
            index=price_df.index,
        )

        portfolio, total_invested = simulate_riskoff(price_df, inv_frac, initial_investment)
        metrics = compute_metrics(portfolio, total_invested)

        # compute_metrics returns {} when the series is too short (<3 months);
        # treat that as a failure so callers get either a full result or (None, None).
        if not metrics:
            return None, None

        return portfolio, metrics
