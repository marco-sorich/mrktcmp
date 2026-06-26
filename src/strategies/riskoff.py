# ---------------------------------------------------------------------------
# strategies/riskoff.py – Risk-Off signal strategy plugin
#
# A tactical asset-allocation strategy.  Unlike DCA there is no recurring
# savings rate: a single lump sum is provided as cash at the start and is
# shifted between the basket and cash according to three market signals
# evaluated on the basket as a whole (see backtest.py for the maths).  The
# signals are evaluated *daily* and the basket is bought/sold to the new target
# whenever the signal changes (then held until the next change); the portfolio
# is valued every trading day.  This plugin is a thin orchestration layer over
# the pure functions in backtest.py so all computation logic stays in one place.
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
    compute_riskoff_signals,
    load_daily_closes,
    simulate_riskoff,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register
from src.utils import log_duration


def _riskoff_order_events(
    price_df: pd.DataFrame,
    target_fraction: pd.Series,
    initial_investment: float = INITIAL_INVESTMENT,
    fx: FxColumns | None = None,
) -> list[OrderEvent]:
    """Record one OrderEvent per Risk-Off rebalance (each time the target changes).

    This is the Risk-Off-specific half of the order log; the generic derived
    columns are added afterwards by ``backtest.build_order_log``.  It mirrors the
    change-driven trading of ``simulate_riskoff``: the lump sum starts as cash
    and, on every day the daily target invested fraction *changes*, the basket is
    rebalanced to it.  Each such day is captured as an event whose side is 'Buy'
    when the target rose (more invested) or 'Sell' when it fell (more cash);
    there is no fresh money, so inflow is always 0.

    Parameters
    ----------
    price_df           – windowed daily closes, one column per asset.
    target_fraction    – per-day target invested fraction (0..1); reindexed onto
                         price_df exactly as simulate_riskoff does.
    initial_investment – one-off lump sum held as cash at the start.
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

    holdings: dict[str, float] = {str(col): 0.0 for col in price_df.columns}
    cash = initial_investment

    # Align the daily targets to the price index; unknown days stay in cash.
    target = target_fraction.reindex(price_df.index).fillna(0.0)

    # Trade only on days the target changes (the starting fraction is 0.0 = all
    # cash); between changes the position is held, emitting no order.  Iterating
    # just these change-days avoids a full daily pass over a held position.
    changed = target.ne(target.shift(fill_value=0.0)).to_numpy()
    change_rows = price_df.loc[changed]
    change_targets = target.loc[changed]
    assert isinstance(change_rows.index, pd.DatetimeIndex)

    # Fraction currently allocated; starts at 0.0 (all cash) so the first
    # non-zero target triggers the initial deployment.
    current = 0.0

    events: list[OrderEvent] = []
    for i, (_, prices) in enumerate(change_rows.iterrows()):
        frac = float(change_targets.iloc[i])

        # Worth before the trade, and the direction of the rebalance.
        value_before = _portfolio_value(holdings, cash, prices)
        side = 'Buy' if frac > current else 'Sell'

        # Rebalance to the new target at today's prices, then record the split
        # (assets_after excludes cash; cash_after is the post-trade cash).
        holdings, cash, _ = _rebalance_to_target(holdings, cash, prices, frac)
        current = frac

        date = change_rows.index[i]
        events.append(OrderEvent(
            date=date,
            side=side,
            value_before=value_before,
            inflow=0.0,
            assets_after=_portfolio_value(holdings, 0.0, prices),
            cash_after=cash,
            asset_values=_asset_values(holdings, prices),  # per-asset worth
            asset_prices=_asset_prices(prices),            # per-asset close (base ccy)
            asset_prices_local=_asset_prices_local(prices, asset_rate, date),  # trading-ccy close
            fx_rates=_fx_rate_values(pair_rate, date),     # per-pair FX rate this day
        ))

    return events


@register
class RiskOffStrategy(BacktestStrategy):
    """Risk-Off: invest a lump sum tactically, shifting to cash on weak signals."""

    @classmethod
    def get_name(cls) -> str:
        return "Risk-Off"

    @classmethod
    def get_icon(cls) -> str:
        return "bi-shield-check"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Risk-Off: invest a one-off lump sum and shift between the "
            "basket and cash based on three daily market signals (200-day trend, "
            "year-to-date return, January barometer); the basket is bought/sold to "
            "the new target whenever the signal changes."
        )

    @classmethod
    def get_long_description(cls) -> str:
        return (
            "## Risk-Off\n\n"
            "A **tactical** lump-sum strategy: stay invested in the basket while "
            "the market looks healthy, and step aside into cash when it does not. "
            "Every day three simple market signals are evaluated and summed into a "
            "**0–3 positive-signal count**; dividing by three gives the day's "
            "target invested fraction (0%, 33%, 67% or 100%).\n\n"
            "**The three signals** (each contributes one point when positive):\n\n"
            "- **Trend** — the equal-weight basket index trades above its long-term "
            "moving average (default 200 trading days).\n"
            "- **Year-to-date return** — the index is up on the year so far.\n"
            "- **January barometer** — the index's return over the first trading "
            "days of the year (default 10) is positive.\n\n"
            "**How it trades:** the basket is bought or sold to the new target "
            "**only on the day the target changes**, holding (and letting the "
            "fraction drift) in between; any uninvested portion sits in cash.\n\n"
            "**Good for:** dampening large drawdowns in falling markets at the cost "
            "of lagging in strong, choppy uptrends.\n\n"
            "**Parameters**\n\n"
            "- **Initial Investment** — the one-off lump sum available to deploy.\n"
            "- **Trend SMA Window** — moving-average length (trading days) for the "
            "trend signal; larger is slower and steadier.\n"
            "- **January Barometer** — number of opening trading days of the year "
            "used for the January-barometer signal."
        )

    @classmethod
    def get_config_schema(cls) -> list[ConfigParam]:
        return [
            ConfigParam(
                key='initial_investment',
                label='Initial Investment',
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
        # Merge caller-supplied values with schema defaults.
        resolved = self.resolve_params(params)

        # Defensive casts: values round-tripped through dcc.Store/JSON can arrive
        # as floats, but rolling(window=...) and head(n) require real ints.
        initial_investment = float(resolved['initial_investment'])
        sma_window = int(resolved['sma_window'])
        first_n_days = int(resolved['first_n_days'])

        # 1. Full daily history (no window) → equal-weight index → signals.
        #    The extra history gives the long look-back signals enough warm-up.
        with log_duration('riskoff: load_daily_closes'):
            daily_df = load_daily_closes(base_url, filenames, df_meta, base_currency)
        if daily_df.empty:
            return None, None, None

        # Signals run over the FULL history (not the window), so their cost scales
        # with the raw row count — logged here to separate it from data loading.
        with log_duration(f'riskoff: index+signals ({daily_df.shape[0]} rows)'):
            index = build_equal_weight_index(daily_df)
            signals = compute_riskoff_signals(index, sma_window, first_n_days)

        # 2. Daily closes restricted to the requested window's calendar months.
        #    Derived from the *same* daily_df so its index aligns exactly with
        #    the signals computed above.
        price_df = _window_by_month(daily_df, start_date, end_date)
        if price_df.empty:
            return None, None, None

        # Forward-fill short price gaps (≤5 trading days, ≈ one week), as in DCA.
        price_df = price_df.ffill(limit=5)

        # FX side-tables aligned to the windowed index, so the order log can show
        # each asset's trading-currency price and the rate it was converted at.
        fx = build_fx_columns(base_url, filenames, df_meta, base_currency, price_df.index)

        # 3. Daily target invested fraction = positive signals / 3, aligned to
        #    the windowed price index. Days before any signal history (NaN) map
        #    to 0.0 (fully in cash). simulate_riskoff trades only when this
        #    fraction changes (same-day execution), holding in between.
        #    The windowed row count is exactly the number of points later plotted.
        with log_duration(f'riskoff: simulate+metrics+orderlog ({price_df.shape[0]} rows)'):
            target_fraction = (signals / 3.0).reindex(price_df.index).fillna(0.0)
            portfolio, total_invested = simulate_riskoff(price_df, target_fraction, initial_investment)
            metrics = compute_metrics(portfolio, total_invested)

            # Normalised equal-weight index for the windowed price_df (first value
            # = 1.0), used as the B&H benchmark column in the order log.  This is
            # separate from the full-history `index` used for signal computation.
            _bh_raw = build_equal_weight_index(price_df)
            bh_index = (_bh_raw / _bh_raw.iloc[0]) if not _bh_raw.empty else None

            # Build the order log only for a valid run.  The whole lump sum is
            # present from the start, so it seeds net-deposits as initial_capital.
            order_log = (
                build_order_log(
                    _riskoff_order_events(price_df, target_fraction, initial_investment, fx),
                    initial_capital=initial_investment,
                    bh_index=bh_index,
                )
                if metrics else None
            )

        # compute_metrics returns {} when the series is too short (<3 months);
        # treat that as a failure so callers get a full result or (None, None, None).
        if not metrics:
            return None, None, None

        return portfolio, metrics, order_log
