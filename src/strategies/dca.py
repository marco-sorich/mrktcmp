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
    OrderEvent,
    OrderRow,
    _is_month_end_trading_day,
    _portfolio_value,
    _window_by_month,
    build_order_log,
    compute_metrics,
    load_daily_closes,
    simulate_dca,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register


def _dca_order_events(
    price_df: pd.DataFrame, monthly_investment: float = MONTHLY_INVESTMENT
) -> list[OrderEvent]:
    """Record one Buy OrderEvent per monthly DCA contribution.

    This is the DCA-specific half of the order log; the generic derived columns
    are added afterwards by ``backtest.build_order_log``.  It mirrors the
    contribution cadence of ``simulate_dca`` (which returns only the daily value
    curve): a contribution lands on each calendar month's last trading day,
    split equally across the assets priced that day.  For every such day it
    captures the portfolio worth just before the buy, the fixed cash inflow, and
    the resulting holdings value — DCA is always fully invested, so cash_after
    is 0.

    Parameters
    ----------
    price_df           – windowed daily closes, one column per asset.
    monthly_investment – fixed amount contributed each month.

    Returns
    -------
    Chronological list of OrderEvents (empty when price_df is empty).
    """
    if price_df.empty:
        return []

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # Units held per asset; grows on every contribution.  Cash is always 0 for
    # DCA because each contribution is immediately and fully invested.
    holdings: dict[str, float] = {str(col): 0.0 for col in price_df.columns}

    # True on each month's last trading day → the single day we contribute.
    contribute_day = _is_month_end_trading_day(price_df.index)

    events: list[OrderEvent] = []
    for i, (_, prices) in enumerate(price_df.iterrows()):
        if not contribute_day[i]:
            continue

        # Assets actually buyable this day (valid, positive price).
        available = {
            str(c): float(p) for c, p in prices.items() if pd.notna(p) and p > 0
        }
        if not available:
            continue

        # Worth before the buy: last month's holdings valued at today's prices.
        value_before = _portfolio_value(holdings, 0.0, prices)

        # Split the contribution equally and add the bought units.
        per_asset = monthly_investment / len(available)
        for col, price in available.items():
            holdings[col] += per_asset / price

        events.append(OrderEvent(
            date=price_df.index[i],
            side='Buy',
            value_before=value_before,
            inflow=monthly_investment,
            assets_after=_portfolio_value(holdings, 0.0, prices),
            cash_after=0.0,
        ))

    return events


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
    ) -> tuple[pd.Series | None, dict[str, str] | None, list[OrderRow] | None]:
        # resolve_params merges caller-supplied values with schema defaults so
        # that passing params={} is equivalent to using all declared defaults.
        resolved = self.resolve_params(params)

        monthly_investment = float(resolved['monthly_investment'])

        # Load the full daily history, then keep only the calendar months of
        # the requested window so the chosen start/end *months* are fully
        # included regardless of exact trading-day boundaries.
        price_df = load_daily_closes(base_url, filenames, df_meta)
        if price_df.empty:
            return None, None, None

        price_df = _window_by_month(price_df, start_date, end_date)
        if price_df.empty:
            return None, None, None

        # Forward-fill short price gaps (≤5 trading days, ≈ one week) to handle
        # weekends, exchange holidays or delayed data without distorting the
        # simulation or carrying delisted assets indefinitely.
        price_df = price_df.ffill(limit=5)

        portfolio, total_invested = simulate_dca(price_df, monthly_investment)
        metrics = compute_metrics(portfolio, total_invested)

        # compute_metrics returns {} when portfolio is too short (< 3 months)
        # or total_invested is zero.  Treat that as a failure so callers always
        # receive either a fully-populated result or (None, None, None).
        if not metrics:
            return None, None, None

        # Build this strategy's order log: DCA-specific events (above) handed to
        # the generic builder.  DCA seeds no initial capital — all money enters
        # through the per-contribution inflows — so initial_capital is 0.
        order_log = build_order_log(
            _dca_order_events(price_df, monthly_investment), initial_capital=0.0
        )

        return portfolio, metrics, order_log
