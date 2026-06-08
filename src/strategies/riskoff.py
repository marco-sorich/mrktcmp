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
    OrderEvent,
    OrderRow,
    _portfolio_value,
    _rebalance_to_target,
    _window_by_month,
    build_equal_weight_index,
    build_order_log,
    compute_metrics,
    compute_riskoff_signals,
    load_daily_closes,
    simulate_riskoff,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register


def _riskoff_order_events(
    price_df: pd.DataFrame,
    target_fraction: pd.Series,
    initial_investment: float = INITIAL_INVESTMENT,
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

    Returns
    -------
    Chronological list of OrderEvents (empty when price_df is empty).
    """
    if price_df.empty:
        return []

    assert isinstance(price_df.index, pd.DatetimeIndex)

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

        events.append(OrderEvent(
            date=change_rows.index[i],
            side=side,
            value_before=value_before,
            inflow=0.0,
            assets_after=_portfolio_value(holdings, 0.0, prices),
            cash_after=cash,
        ))

    return events


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
            "basket and cash based on three daily market signals (200-day trend, "
            "year-to-date return, January barometer); the basket is bought/sold to "
            "the new target whenever the signal changes."
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

    def run(
        self,
        base_url: str,
        filenames: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        df_meta: pd.DataFrame,
        params: dict[str, int | float | str],
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
        daily_df = load_daily_closes(base_url, filenames, df_meta)
        if daily_df.empty:
            return None, None, None

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

        # 3. Daily target invested fraction = positive signals / 3, aligned to
        #    the windowed price index. Days before any signal history (NaN) map
        #    to 0.0 (fully in cash). simulate_riskoff trades only when this
        #    fraction changes (same-day execution), holding in between.
        target_fraction = (signals / 3.0).reindex(price_df.index).fillna(0.0)

        portfolio, total_invested = simulate_riskoff(price_df, target_fraction, initial_investment)
        metrics = compute_metrics(portfolio, total_invested)

        # compute_metrics returns {} when the series is too short (<3 months);
        # treat that as a failure so callers get a full result or (None, None, None).
        if not metrics:
            return None, None, None

        # Build this strategy's order log: Risk-Off-specific rebalance events
        # (above) handed to the generic builder.  The whole lump sum is present
        # from the start, so it seeds the net-deposits basis as initial_capital.
        order_log = build_order_log(
            _riskoff_order_events(price_df, target_fraction, initial_investment),
            initial_capital=initial_investment,
        )

        return portfolio, metrics, order_log
