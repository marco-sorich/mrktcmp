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
    FxColumns,
    OrderEvent,
    OrderRow,
    _asset_prices,
    _asset_prices_local,
    _asset_values,
    _fx_rate_values,
    _is_month_end_trading_day,
    _portfolio_value,
    _window_by_month,
    build_equal_weight_index,
    build_fx_columns,
    build_order_log,
    compute_metrics,
    load_daily_closes,
    simulate_dca,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register
from src.utils import log_duration


def _dca_order_events(
    price_df: pd.DataFrame,
    monthly_investment: float = MONTHLY_INVESTMENT,
    fx: FxColumns | None = None,
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
    fx                 – optional FX context (see backtest.FxColumns); when given,
                         each event also carries the trading-currency quote and the
                         per-pair FX rates so the order table can show conversions.

    Returns
    -------
    Chronological list of OrderEvents (empty when price_df is empty).
    """
    if price_df.empty:
        return []

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # FX side-tables (empty when no conversion applies) used to add the
    # trading-currency price and per-pair rate columns to each order event.
    asset_rate = fx.asset_rate if fx else {}
    pair_rate = fx.pair_rate if fx else {}

    # Units held per asset; grows on every contribution.  Cash is always 0 for
    # DCA because each contribution is immediately and fully invested.
    holdings: dict[str, float] = {str(col): 0.0 for col in price_df.columns}

    # Iterate only the month-end rows (the single contribution day each month)
    # rather than every trading day: nothing changes in between, so the days we
    # skipped emitted no event anyway.
    month_end_rows = price_df.loc[_is_month_end_trading_day(price_df.index)]
    assert isinstance(month_end_rows.index, pd.DatetimeIndex)

    events: list[OrderEvent] = []
    for i, (_, prices) in enumerate(month_end_rows.iterrows()):
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

        date = month_end_rows.index[i]
        events.append(OrderEvent(
            date=date,
            side='Buy',
            value_before=value_before,
            inflow=monthly_investment,
            assets_after=_portfolio_value(holdings, 0.0, prices),
            cash_after=0.0,
            asset_values=_asset_values(holdings, prices),  # per-asset worth
            asset_prices=_asset_prices(prices),            # per-asset close (base ccy)
            asset_prices_local=_asset_prices_local(prices, asset_rate, date),  # trading-ccy close
            fx_rates=_fx_rate_values(pair_rate, date),     # per-pair FX rate this day
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
                label='Monthly Investment',
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
        base_currency: str = 'EUR',
    ) -> tuple[pd.Series | None, dict[str, str] | None, list[OrderRow] | None]:
        # resolve_params merges caller-supplied values with schema defaults so
        # that passing params={} is equivalent to using all declared defaults.
        resolved = self.resolve_params(params)

        monthly_investment = float(resolved['monthly_investment'])

        # Load the full daily history, then keep only the calendar months of
        # the requested window so the chosen start/end *months* are fully
        # included regardless of exact trading-day boundaries.
        with log_duration('dca: load_daily_closes'):
            price_df = load_daily_closes(base_url, filenames, df_meta, base_currency)
        if price_df.empty:
            return None, None, None

        price_df = _window_by_month(price_df, start_date, end_date)
        if price_df.empty:
            return None, None, None

        # Forward-fill short price gaps (≤5 trading days, ≈ one week) to handle
        # weekends, exchange holidays or delayed data without distorting the
        # simulation or carrying delisted assets indefinitely.
        price_df = price_df.ffill(limit=5)

        # FX side-tables aligned to the windowed index, so the order log can show
        # each asset's trading-currency price and the rate it was converted at.
        fx = build_fx_columns(base_url, filenames, df_meta, base_currency, price_df.index)

        # The windowed row count is exactly the number of points later plotted.
        with log_duration(f'dca: simulate+metrics+orderlog ({price_df.shape[0]} rows)'):
            portfolio, total_invested = simulate_dca(price_df, monthly_investment)
            metrics = compute_metrics(portfolio, total_invested)

            # Normalised equal-weight index (first value = 1.0) used as the B&H
            # benchmark column in the order log.
            _bh_raw = build_equal_weight_index(price_df)
            bh_index = (_bh_raw / _bh_raw.iloc[0]) if not _bh_raw.empty else None

            # compute_metrics returns {} when portfolio is too short (< 3 months)
            # or total_invested is zero; build the order log only for a valid run.
            # DCA seeds no initial capital — all money enters through the
            # per-contribution inflows — so initial_capital is 0.
            order_log = (
                build_order_log(
                    _dca_order_events(price_df, monthly_investment, fx),
                    initial_capital=0.0,
                    bh_index=bh_index,
                )
                if metrics else None
            )

        # Treat empty metrics as a failure so callers always receive either a
        # fully-populated result or (None, None, None).
        if not metrics:
            return None, None, None

        return portfolio, metrics, order_log
