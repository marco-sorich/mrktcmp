# ---------------------------------------------------------------------------
# strategies/summergap.py – "Summer Gap" seasonal strategy plugin
#
# A calendar-driven tactical strategy modelled on the German stock-market adage
# "Sommerloch" (the quiet summer lull).  A single lump sum is invested up front
# and then stepped entirely into cash for a fixed seasonal window each year –
# by default sold at the start of August and bought back at the start of
# October – repeating every year of the backtest window.
#
# Mechanically this is identical to the Risk-Off strategy: both shift a lump
# sum between the basket and cash via a daily *target invested fraction*,
# trading only on the day the target changes (and letting the position drift in
# between).  The only difference is the source of that target: Summer Gap
# derives it from the *calendar* (1.0 invested, 0.0 inside the out-of-market
# window) instead of from market signals.  We therefore reuse the very same
# pure engine (simulate_riskoff), all-priced gate (gate_target_until_all_priced)
# and order-event generator (_riskoff_order_events, which is fully generic over
# an arbitrary target series) from backtest.py / riskoff.py, so all computation
# logic stays in one place.
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import (
    INITIAL_INVESTMENT,
    OrderRow,
    _window_by_month,
    build_equal_weight_index,
    build_fx_columns,
    build_order_log,
    compute_metrics,
    gate_target_until_all_priced,
    load_daily_closes,
    simulate_riskoff,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register
# The order-event generator is strategy-agnostic – it records one event per day
# the target fraction changes – so Summer Gap reuses Risk-Off's verbatim rather
# than duplicating it.
from src.strategies.riskoff import _riskoff_order_events
from src.utils import log_duration

# Month names for the GUI dropdowns; index + 1 is the calendar month number.
_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]


def _seasonal_target(
    index: pd.DatetimeIndex,
    sell_month: int,
    sell_day: int,
    buy_month: int,
    buy_day: int,
) -> pd.Series:
    """Build the daily target invested fraction for the seasonal swap.

    The result is 1.0 (fully invested) on every day *outside* the out-of-market
    window and 0.0 (fully in cash) on every day *inside* it.  The window is the
    half-open calendar range ``[sell-date, buy-date)``: a day on/after the sell
    date but strictly before the buy date is "out".  Feeding this series to
    ``simulate_riskoff`` (which trades only when the target changes) therefore
    produces a **Sell** on the first trading day on/after the sell date and a
    **Buy** on the first trading day on/after the buy date, every year.

    Dates are compared as month-day ordinals (``month * 100 + day``) so the
    comparison is purely seasonal and independent of the year.

    Parameters
    ----------
    index                – daily price index to evaluate the target on.
    sell_month, sell_day – calendar date the basket is sold (out-of-market start).
    buy_month, buy_day   – calendar date the basket is bought back (window end).

    Returns
    -------
    A ``pd.Series`` of 1.0/0.0 floats aligned to *index*.
    """
    # Seasonal (year-independent) ordinal for each day, e.g. 1 Aug -> 801.
    ords = np.asarray(index.month) * 100 + np.asarray(index.day)
    sell_ord = sell_month * 100 + sell_day
    buy_ord = buy_month * 100 + buy_day

    if sell_ord < buy_ord:
        # Window lies within one calendar year (the default Aug -> Oct case).
        out = (ords >= sell_ord) & (ords < buy_ord)
    elif sell_ord > buy_ord:
        # Window wraps the year end (e.g. sell in November, buy back in February).
        out = (ords >= sell_ord) | (ords < buy_ord)
    else:
        # Degenerate same-date configuration: there is no out-of-market window,
        # so the strategy stays fully invested throughout.
        out = np.zeros(len(index), dtype=bool)

    return pd.Series(np.where(out, 0.0, 1.0), index=index)


@register
class SummerGapStrategy(BacktestStrategy):
    """Summer Gap: invest a lump sum but step aside into cash for the summer."""

    @classmethod
    def get_name(cls) -> str:
        return "Summer Gap"

    @classmethod
    def get_icon(cls) -> str:
        return "bi-sun"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Summer Gap: invest a one-off lump sum, then sell the whole basket "
            "into cash for a fixed seasonal window each year (sold at the start "
            "of August and bought back at the start of October by default)."
        )

    @classmethod
    def get_long_description(cls) -> str:
        return (
            "## Summer Gap\n\n"
            "A **seasonal** lump-sum strategy inspired by the *“Sell in May”* / "
            "*Sommerloch* market adage: stay invested in the basket for most of "
            "the year, but step entirely aside into cash during the traditionally "
            "weak late-summer stretch, then buy back in.\n\n"
            "**How it trades:** a one-off lump sum is deployed into the basket and "
            "then, every year, the **whole basket is sold to cash** on the first "
            "trading day on/after the configured sell date (default 1 August) and "
            "**bought back** on the first trading day on/after the configured buy "
            "date (default 1 October).  In between those swaps the position is "
            "simply held (its value drifting with the market); the trade happens "
            "**only on the day the in/out target changes**.\n\n"
            "The lump sum stays fully in cash until every basket asset is priced, "
            "so the first deployment is split across the whole basket by the "
            "per-asset weights rather than over-weighting whichever asset listed "
            "first.\n\n"
            "**Good for:** sidestepping a recurring seasonal soft patch; it adds "
            "value only in years where that window is genuinely weak and will lag "
            "a plain buy-and-hold in summers that rally.\n\n"
            "**Parameters**\n\n"
            "- **Initial Investment** — the one-off lump sum available to deploy.\n"
            "- **Sell Month / Sell Day** — the calendar date each year the basket "
            "is sold into cash (the out-of-market window starts here).\n"
            "- **Buy Month / Buy Day** — the calendar date each year the basket is "
            "bought back (the out-of-market window ends here)."
        )

    @classmethod
    def get_config_schema(cls) -> list[ConfigParam]:
        return [
            ConfigParam(
                key='initial_investment',
                label='Initial Investment',
                type='float',
                # Share the module constant so this and Buy & Hold / Risk-Off all
                # start from the same default lump-sum value.
                default=float(INITIAL_INVESTMENT),
                min_value=1.0,
                max_value=1_000_000_000.0,
            ),
            ConfigParam(
                key='sell_month',
                label='Sell Month',
                type='select',
                default='August',
                options=list(_MONTHS),
            ),
            ConfigParam(
                key='sell_day',
                label='Sell Day',
                type='int',
                default=1,
                min_value=1,
                max_value=31,
            ),
            ConfigParam(
                key='buy_month',
                label='Buy Month',
                type='select',
                default='October',
                options=list(_MONTHS),
            ),
            ConfigParam(
                key='buy_day',
                label='Buy Day',
                type='int',
                default=1,
                min_value=1,
                max_value=31,
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
        weights: dict[str, float] | None = None,
    ) -> tuple[pd.Series | None, dict[str, str] | None, list[OrderRow] | None]:
        # Merge caller-supplied values with schema defaults.
        resolved = self.resolve_params(params)

        # Defensive casts: values round-tripped through dcc.Store/JSON can arrive
        # as floats/strings, but the arithmetic below needs concrete int/float.
        initial_investment = float(resolved['initial_investment'])
        sell_day = int(resolved['sell_day'])
        buy_day = int(resolved['buy_day'])
        # Map the human-readable month names back to calendar month numbers.
        sell_month = _MONTHS.index(str(resolved['sell_month'])) + 1
        buy_month = _MONTHS.index(str(resolved['buy_month'])) + 1

        # 1. Daily closes (already FX-converted to the reporting currency).  No
        #    extra look-back warm-up is needed: unlike Risk-Off there are no
        #    rolling signals, the target depends only on the calendar.
        with log_duration('summergap: load_daily_closes'):
            daily_df = load_daily_closes(base_url, filenames, df_meta, base_currency)
        if daily_df.empty:
            return None, None, None

        # 2. Restrict to the requested window's calendar months.
        price_df = _window_by_month(daily_df, start_date, end_date)
        if price_df.empty:
            return None, None, None

        # Forward-fill short price gaps (≤5 trading days, ≈ one week), as elsewhere.
        price_df = price_df.ffill(limit=5)

        # FX side-tables aligned to the windowed index, so the order log can show
        # each asset's trading-currency price and the rate it was converted at.
        fx = build_fx_columns(base_url, filenames, df_meta, base_currency, price_df.index)

        assert isinstance(price_df.index, pd.DatetimeIndex)

        with log_duration(f'summergap: simulate+metrics+orderlog ({price_df.shape[0]} rows)'):
            # 3. Seasonal daily target (1.0 invested / 0.0 in cash), then hold the
            #    lump sum in cash until the whole basket is priced so the first
            #    deployment splits the investment across all (positively weighted)
            #    assets by their weights.  The gate is applied to the single
            #    target series both the simulation and the order log consume,
            #    keeping them in lockstep.
            target_fraction = _seasonal_target(
                price_df.index, sell_month, sell_day, buy_month, buy_day
            )
            target_fraction = gate_target_until_all_priced(price_df, target_fraction, weights)

            # 4. Reuse Risk-Off's change-driven engine: trade only when the target
            #    flips (sell at the window start, buy at its end), hold in between.
            portfolio, total_invested = simulate_riskoff(
                price_df, target_fraction, initial_investment, weights
            )
            metrics = compute_metrics(portfolio, total_invested)

            # Normalised (weighted) basket index for the windowed price_df (first
            # value = 1.0), used as the Buy & Hold benchmark column in the order log.
            _bh_raw = build_equal_weight_index(price_df, weights)
            bh_index = (_bh_raw / _bh_raw.iloc[0]) if not _bh_raw.empty else None

            # 5. Build the order log only for a valid run.  The whole lump sum is
            #    present from the start, so it seeds net-deposits as initial_capital.
            order_log = (
                build_order_log(
                    _riskoff_order_events(
                        price_df, target_fraction, initial_investment, fx, weights
                    ),
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
