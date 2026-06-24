# ---------------------------------------------------------------------------
# strategies/lumpsum.py – Buy & Hold (one single initial investment) plugin
#
# The simplest possible strategy: invest a one-off lump sum in full on the very
# first trading day, split equally across the basket, and then hold the position
# unchanged until the end — no further orders, no rebalancing, no contributions.
# Like the other plugins this is a thin orchestration layer over backtest.py, so
# all simulation logic stays in one place.
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

from src.backtest import (
    INITIAL_INVESTMENT,
    FxColumns,
    OrderEvent,
    OrderRow,
    _asset_prices,
    _asset_prices_local,
    _asset_values,
    _fx_rate_values,
    _portfolio_value,
    _rebalance_to_target,
    _window_by_month,
    build_equal_weight_index,
    build_fx_columns,
    build_order_log,
    compute_metrics,
    load_daily_closes,
    simulate_lumpsum,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register
from src.utils import log_duration


def _lumpsum_order_events(
    price_df: pd.DataFrame,
    initial_investment: float = INITIAL_INVESTMENT,
    fx: FxColumns | None = None,
) -> list[OrderEvent]:
    """Record the single Buy OrderEvent of a lump-sum buy-and-hold run.

    This is the Buy & Hold half of the order log; the generic derived columns are
    added afterwards by ``backtest.build_order_log``.  It mirrors
    ``simulate_lumpsum`` (which returns only the daily value curve): the whole
    lump sum is deployed once, on the first trading day with at least one buyable
    asset, split equally across the assets priced that day.  Because the entire
    amount is invested there is no remaining cash, and as nothing is ever traded
    again exactly one event is produced for the whole window.

    Parameters
    ----------
    price_df           – windowed daily closes, one column per asset.
    initial_investment – one-off lump sum invested on the first buyable day.
    fx                 – optional FX context (see backtest.FxColumns); when given,
                         each event also carries the trading-currency quote and the
                         per-pair FX rates so the order table can show conversions.

    Returns
    -------
    A single-element list with the day-one Buy (empty when price_df is empty or
    no asset is ever buyable).
    """
    if price_df.empty:
        return []

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # FX side-tables (empty when no conversion applies) used to add the
    # trading-currency price and per-pair rate columns to the order event.
    asset_rate = fx.asset_rate if fx else {}
    pair_rate = fx.pair_rate if fx else {}

    # Find the first day on which at least one asset has a valid, positive price;
    # that is the day the lump sum is deployed.
    holdings: dict[str, float] = {str(col): 0.0 for col in price_df.columns}
    for i in range(len(price_df)):
        prices = price_df.iloc[i]
        # Assets actually buyable this day (valid, positive price).
        if not any(pd.notna(p) and p > 0 for p in prices):
            continue

        # Deploy 100% equal-weight, reusing the shared rebalance primitive so the
        # holdings match simulate_lumpsum exactly (cash_after is ~0 = fully invested).
        holdings, _, _ = _rebalance_to_target(holdings, float(initial_investment), prices, 1.0)

        date = price_df.index[i]
        return [OrderEvent(
            date=date,
            side='Buy',
            value_before=0.0,                 # nothing held before the first buy
            inflow=initial_investment,        # the whole lump sum enters here
            assets_after=_portfolio_value(holdings, 0.0, prices),
            cash_after=0.0,                   # fully invested, no residual cash
            asset_values=_asset_values(holdings, prices),  # per-asset worth
            asset_prices=_asset_prices(prices),            # per-asset close (base ccy)
            asset_prices_local=_asset_prices_local(prices, asset_rate, date),  # trading-ccy close
            fx_rates=_fx_rate_values(pair_rate, date),     # per-pair FX rate this day
        )]

    return []


@register
class BuyHoldStrategy(BacktestStrategy):
    """Buy & Hold: invest a single lump sum on day one and hold until the end."""

    @classmethod
    def get_name(cls) -> str:
        return "Buy & Hold"

    @classmethod
    def get_icon(cls) -> str:
        return "bi-bag-check"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Buy & Hold: invest a single lump sum on the first day, "
            "split equally across all basket assets, and hold until the end."
        )

    @classmethod
    def get_config_schema(cls) -> list[ConfigParam]:
        return [
            ConfigParam(
                key='initial_investment',
                label='Initial Investment',
                type='float',
                # Share the module constant from backtest.py so both code paths
                # always start from the same default value.
                default=float(INITIAL_INVESTMENT),
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

        initial_investment = float(resolved['initial_investment'])

        # Load the full daily history, then keep only the calendar months of the
        # requested window so the chosen start/end *months* are fully included
        # regardless of exact trading-day boundaries.
        with log_duration('lumpsum: load_daily_closes'):
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
        with log_duration(f'lumpsum: simulate+metrics+orderlog ({price_df.shape[0]} rows)'):
            portfolio, total_invested = simulate_lumpsum(price_df, initial_investment)
            metrics = compute_metrics(portfolio, total_invested)

            # Normalised equal-weight index (first value = 1.0) used as the B&H
            # benchmark column in the order log.
            _bh_raw = build_equal_weight_index(price_df)
            bh_index = (_bh_raw / _bh_raw.iloc[0]) if not _bh_raw.empty else None

            # compute_metrics returns {} when portfolio is too short (< 3 months)
            # or total_invested is zero; build the order log only for a valid run.
            # The whole lump sum is present from the start, so it seeds the
            # net-deposits tally as initial_capital (like Risk-Off).
            order_log = (
                build_order_log(
                    _lumpsum_order_events(price_df, initial_investment, fx),
                    initial_capital=initial_investment,
                    bh_index=bh_index,
                )
                if metrics else None
            )

        # Treat empty metrics as a failure so callers always receive either a
        # fully-populated result or (None, None, None).
        if not metrics:
            return None, None, None

        return portfolio, metrics, order_log
