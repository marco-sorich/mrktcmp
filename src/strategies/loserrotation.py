# ---------------------------------------------------------------------------
# strategies/loserrotation.py – "Loser Rotation" seasonal strategy plugin
#
# A calendar-driven, cross-sectional strategy modelled on the well-known
# "losers of the year rebound in July" effect: at the start of the third
# quarter, funds and ETFs commonly rebalance out of the year's winners into its
# most beaten-down names, producing a short-lived rebound in a basket's worst
# performers. Once a year, on a configurable buy date, every basket asset's
# year-to-date return (from the first trading day of that calendar year up to
# the buy date) is ranked and the lump sum is bought into only the worst-
# performing N assets; the position is held until a configurable sell date,
# when it is sold back to cash until next year's buy date repeats the process
# with a freshly ranked selection.
#
# Unlike Risk-Off/Summer Gap — whose per-asset weights are one static dict for
# the whole run and only the aggregate invested fraction varies day to day —
# this strategy's asset *selection* itself changes every year, so it uses the
# new backtest.simulate_rotation/_rotation_order_events engine (an explicit
# list of (date, target_fraction, weights) events) instead of the fixed-weight
# simulate_riskoff/_riskoff_order_events pair.
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

from src.backtest import (
    INITIAL_INVESTMENT,
    OrderRow,
    _rotation_order_events,
    _window_by_month,
    build_equal_weight_index,
    build_fx_columns,
    build_order_log,
    compute_metrics,
    load_daily_closes,
    simulate_rotation,
)
from src.strategies.base import BacktestStrategy, ConfigParam
from src.strategies.registry import register
from src.utils import log_duration

# Month names for the GUI dropdowns; index + 1 is the calendar month number.
_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]


def _first_trading_day_of_year(index: pd.DatetimeIndex, year: int) -> pd.Timestamp | None:
    """The earliest trading day of *year* present in *index*, or None if absent."""
    year_rows = index[index.year == year]
    return year_rows.min() if len(year_rows) else None


def _first_trading_day_on_or_after(
    index: pd.DatetimeIndex, year: int, month: int, day: int
) -> pd.Timestamp | None:
    """First trading day of *year* in *index* on/after the (month, day) date.

    Unlike ``summergap._seasonal_target`` (which only needs a year-independent
    boolean mask), a rotation event must trade on a concrete calendar day, so
    this returns the actual matching ``Timestamp`` — the same "first day on/after
    a target ordinal" idea, restricted to a single year and resolved to a date
    rather than a mask.  Returns None when *year* has no rows in *index*, or none
    of them fall on/after the target date (e.g. it lies after the year's last
    trading day).
    """
    year_rows = index[index.year == year]
    if year_rows.empty:
        return None
    ords = year_rows.month * 100 + year_rows.day
    on_or_after = year_rows[ords >= (month * 100 + day)]
    return on_or_after.min() if len(on_or_after) else None


def _select_losers(
    daily_df: pd.DataFrame,
    year_start: pd.Timestamp,
    buy_date: pd.Timestamp,
    n_losers: int,
) -> list[str]:
    """The *n_losers* symbols with the lowest return from *year_start* to *buy_date*.

    Only symbols priced (valid, positive close) at **both** dates are eligible for
    ranking — an asset missing either price cannot have its year-to-date return
    computed. Ties are broken by column order (pandas' stable sort). Returns
    fewer than *n_losers* symbols when fewer are eligible that year, and an empty
    list when none are.
    """
    start_prices = daily_df.loc[year_start]
    buy_prices = daily_df.loc[buy_date]
    assert isinstance(start_prices, pd.Series)
    assert isinstance(buy_prices, pd.Series)
    eligible = start_prices.notna() & buy_prices.notna() & (start_prices > 0)
    if not eligible.any():
        return []
    ytd_return = (buy_prices[eligible] / start_prices[eligible]) - 1.0
    ranked = ytd_return.sort_values(ascending=True, kind='stable')
    return [str(s) for s in ranked.index[:n_losers]]


def _build_rotation_events(
    daily_df: pd.DataFrame,
    price_df: pd.DataFrame,
    buy_month: int,
    buy_day: int,
    sell_month: int,
    sell_day: int,
    n_losers: int,
    weights: dict[str, float] | None,
) -> list[tuple[pd.Timestamp, float, dict[str, float] | None]]:
    """Build the annual buy/sell rotation events for every year spanned by *price_df*.

    Uses the **full** *daily_df* (unwindowed) to find each year's 1 January
    anchor, buy date and YTD-return ranking, so the ranking is correct even when
    the requested window starts mid-year — the same full-history-for-signals
    split ``riskoff.py`` uses for its long look-back signals. Only events whose
    date actually falls inside *price_df*'s (windowed) index are returned, since
    those are the only ones ``simulate_rotation``/``_rotation_order_events`` can
    trade on.

    The sell date is searched in the *same* year as the buy date when the sell
    ordinal (month, day) falls after the buy ordinal (the default: buy in July,
    sell in October), and in the *following* year otherwise — mirroring
    ``summergap._seasonal_target``'s wrap-around handling for windows that cross
    the year boundary (e.g. buy in November, sell in February).

    Each selected year contributes a ``(buy_date, 1.0, year_weights)`` event and,
    when a sell date is found, a ``(sell_date, 0.0, year_weights)`` event —
    *year_weights* gives the selected losers the caller's relative weight (or 1.0
    each, i.e. equal split) and implicitly zero-weights every other basket
    symbol, so the lump sum is only ever invested in that year's selection.

    Returns
    -------
    Events in chronological order (not necessarily alternating strictly Buy/Sell
    if a year has no eligible losers — that year simply contributes no events).
    """
    if price_df.empty:
        return []
    assert isinstance(price_df.index, pd.DatetimeIndex)
    assert isinstance(daily_df.index, pd.DatetimeIndex)

    windowed_dates = set(price_df.index)
    buy_ord = buy_month * 100 + buy_day
    sell_ord = sell_month * 100 + sell_day

    events: list[tuple[pd.Timestamp, float, dict[str, float] | None]] = []

    for year in range(int(price_df.index.year.min()), int(price_df.index.year.max()) + 1):
        year_start = _first_trading_day_of_year(daily_df.index, year)
        buy_date = _first_trading_day_on_or_after(daily_df.index, year, buy_month, buy_day)
        if year_start is None or buy_date is None:
            continue

        selected = _select_losers(daily_df, year_start, buy_date, n_losers)
        if not selected:
            continue

        # Selected losers keep the caller's relative weight (or 1.0 = equal
        # split); every other basket symbol is implicitly 0 (absent from the
        # dict), so _rebalance_to_target only ever allocates to this selection.
        year_weights = {sym: (float(weights.get(sym, 1.0)) if weights else 1.0) for sym in selected}

        sell_year = year if sell_ord > buy_ord else year + 1
        sell_date = _first_trading_day_on_or_after(daily_df.index, sell_year, sell_month, sell_day)

        if buy_date in windowed_dates:
            events.append((buy_date, 1.0, year_weights))
        if sell_date is not None and sell_date in windowed_dates:
            events.append((sell_date, 0.0, year_weights))

    return events


@register
class LoserRotationStrategy(BacktestStrategy):
    """Loser Rotation: annually rotate a lump sum into the basket's worst YTD performers."""

    @classmethod
    def get_name(cls) -> str:
        return "Loser Rotation"

    @classmethod
    def get_icon(cls) -> str:
        return "bi-arrow-repeat"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Loser Rotation: once a year, invest a lump sum into the basket's "
            "worst year-to-date performers (a fixed count), buying at a "
            "configurable date and selling back to cash at another; repeats "
            "every year with a freshly ranked selection."
        )

    @classmethod
    def get_long_description(cls) -> str:
        return (
            "## Loser Rotation\n\n"
            "A **seasonal, cross-sectional** strategy inspired by the well-known "
            "“losers of the year rebound in July” effect: at the start of "
            "the third quarter, funds and ETFs commonly rebalance out of the "
            "year's winners into its most beaten-down, fundamentally cheap "
            "names, producing a short-lived rebound in a basket's worst "
            "performers.\n\n"
            "**How it trades:** once a year, on the configured buy date, every "
            "basket asset's return from the first trading day of that calendar "
            "year up to the buy date is ranked, and the lump sum is bought "
            "**only into the worst-performing N assets** (split by their "
            "per-asset weights, equal by default). The position is held until "
            "the configured sell date, when it is sold entirely back to cash; "
            "the basket then sits in cash until next year's buy date, when the "
            "worst performers are re-selected from scratch. There is no "
            "rebalancing between the buy and sell dates — the position simply "
            "drifts with the market.\n\n"
            "**Good for:** exploring a mean-reversion / sector-rotation effect on "
            "a basket's own laggards; it adds value only in years the rebound "
            "actually materialises and can lag a plain buy-and-hold otherwise.\n\n"
            "**Parameters**\n\n"
            "- **Initial Investment** — the one-off lump sum available to "
            "deploy.\n"
            "- **Number of Losers** — how many of the basket's worst "
            "year-to-date performers to buy each year (fewer are bought if the "
            "basket doesn't have that many eligible assets).\n"
            "- **Buy Month / Buy Day** — the calendar date each year the "
            "worst performers are ranked and bought.\n"
            "- **Sell Month / Sell Day** — the calendar date each year the "
            "position is sold back to cash."
        )

    @classmethod
    def get_config_schema(cls) -> list[ConfigParam]:
        return [
            ConfigParam(
                key='initial_investment',
                label='Initial Investment',
                type='float',
                # Share the module constant so this and Buy & Hold / Risk-Off /
                # Summer Gap all start from the same default lump-sum value.
                default=float(INITIAL_INVESTMENT),
                min_value=1.0,
                max_value=1_000_000_000.0,
            ),
            ConfigParam(
                key='n_losers',
                label='Number of Losers',
                type='int',
                default=3,
                min_value=1,
                max_value=20,
            ),
            ConfigParam(
                key='buy_month',
                label='Buy Month',
                type='select',
                default='July',
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
            ConfigParam(
                key='sell_month',
                label='Sell Month',
                type='select',
                default='October',
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
        # as floats/strings, but the ranking/rotation logic needs concrete types.
        initial_investment = float(resolved['initial_investment'])
        n_losers = int(resolved['n_losers'])
        buy_day = int(resolved['buy_day'])
        sell_day = int(resolved['sell_day'])
        # Map the human-readable month names back to calendar month numbers.
        buy_month = _MONTHS.index(str(resolved['buy_month'])) + 1
        sell_month = _MONTHS.index(str(resolved['sell_month'])) + 1

        # 1. Full daily history (no window): each year's 1 January anchor must be
        #    available even when the requested window starts mid-year.
        with log_duration('loserrotation: load_daily_closes'):
            daily_df = load_daily_closes(base_url, filenames, df_meta, base_currency)
        if daily_df.empty:
            return None, None, None

        # 2. Daily closes restricted to the requested window's calendar months.
        price_df = _window_by_month(daily_df, start_date, end_date)
        if price_df.empty:
            return None, None, None

        # Forward-fill short price gaps (≤5 trading days, ≈ one week), as elsewhere.
        price_df = price_df.ffill(limit=5)

        # FX side-tables aligned to the windowed index, so the order log can show
        # each asset's trading-currency price and the rate it was converted at.
        fx = build_fx_columns(base_url, filenames, df_meta, base_currency, price_df.index)

        assert isinstance(price_df.index, pd.DatetimeIndex)

        with log_duration(f'loserrotation: simulate+metrics+orderlog ({price_df.shape[0]} rows)'):
            # 3. One (buy, sell) event pair per year, each carrying that year's
            #    freshly ranked bottom-N weights (built from the full history).
            events = _build_rotation_events(
                daily_df, price_df, buy_month, buy_day, sell_month, sell_day, n_losers, weights
            )

            # 4. Rebalance to each event's own fraction AND weights, holding
            #    (and drifting with the market) in between.
            portfolio, total_invested = simulate_rotation(price_df, events, initial_investment)
            metrics = compute_metrics(portfolio, total_invested)

            # Normalised (weighted) basket index for the windowed price_df (first
            # value = 1.0), used as the Buy & Hold benchmark column in the order log.
            _bh_raw = build_equal_weight_index(price_df, weights)
            bh_index = (_bh_raw / _bh_raw.iloc[0]) if not _bh_raw.empty else None

            # 5. Build the order log only for a valid run. The whole lump sum is
            #    present from the start, so it seeds net-deposits as initial_capital.
            order_log = (
                build_order_log(
                    _rotation_order_events(price_df, events, initial_investment, fx),
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
