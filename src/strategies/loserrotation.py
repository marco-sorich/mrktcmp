# ---------------------------------------------------------------------------
# strategies/loserrotation.py – "Loser Rotation" seasonal strategy plugin
#
# A buy-and-hold strategy with an annual, calendar-driven tactical tilt,
# modelled on the well-known "losers of the year rebound in July" effect: at
# the start of the third quarter, funds and ETFs commonly rebalance out of the
# year's winners into its most beaten-down names, producing a short-lived
# rebound in a basket's worst performers.
#
# The lump sum is deployed up front across the whole basket at its per-asset
# weights, exactly like Buy & Hold, and stays fully invested throughout — the
# strategy never steps into cash. Once a year, on a configurable buy date, every
# basket asset's year-to-date return (from the first trading day of that
# calendar year up to the buy date) is ranked and a configurable share of the
# portfolio value is shifted into the worst-performing N assets (split equally
# among them; the remainder stays at the original weighting). On a configurable
# sell date the original weighting is fully restored. This repeats every year
# with a freshly ranked selection.
#
# Unlike Risk-Off/Summer Gap — whose per-asset weights are one static dict for
# the whole run and only the aggregate invested fraction varies day to day —
# this strategy's asset *allocation* itself changes over time, so it uses the
# backtest.simulate_rotation/_rotation_order_events engine (an explicit list of
# RotationEvents, each with its own weights) instead of the fixed-weight
# simulate_riskoff/_riskoff_order_events pair.
# ---------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

from src.backtest import (
    INITIAL_INVESTMENT,
    OrderRow,
    RotationEvent,
    _first_all_priced_pos,
    _normalised_weights,
    _rotation_order_events,
    _weight_array,
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

    *daily_df* should already be forward-filled (see ``_build_rotation_events``):
    *year_start* and *buy_date* are picked from the basket's **combined** trading
    calendar (the union of every asset's own trading days), so on either date some
    basket members may not have traded that exact day (e.g. a market holiday
    specific to their own exchange) even though they were priced a day or two
    earlier. Without filling first, those assets would be wrongly marked
    ineligible for the whole year purely because of a calendar mismatch, not
    because they were actually unpriced.
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
    shift_frac: float,
    weights: dict[str, float] | None,
) -> list[RotationEvent]:
    """Build the initial deployment plus the annual tilt/reset RotationEvents.

    The portfolio is **always fully invested** (target fraction 1.0 on every
    event); what changes is the per-asset allocation:

    1. **Initial deployment** — exactly like Buy & Hold: on the first day every
       positively-weighted basket asset is priced (``_first_all_priced_pos``),
       the lump sum is deployed at the basket's original per-asset weights.
       When no such day exists, no events are emitted and the lump sum stays in
       cash, mirroring ``simulate_lumpsum``'s never-fully-priced behaviour.
    2. **Annual tilt (Buy)** — on each year's buy date, *shift_frac* of the
       portfolio value moves into that year's *n_losers* worst YTD performers
       (split **equally** among them — the basket weights deliberately play no
       role here, so even an asset the user weighted 0 can be a rotation
       target); the remaining ``1 - shift_frac`` stays at the original weights.
       Expressed as one blended weights dict per year.
    3. **Annual reset (Sell)** — on each year's sell date the original
       weighting is fully restored, regardless of *shift_frac*.

    Uses the **full** *daily_df* (unwindowed) to find each year's 1 January
    anchor, buy date and YTD-return ranking, so the ranking is correct even when
    the requested window starts mid-year — the same full-history-for-signals
    split ``riskoff.py`` uses for its long look-back signals. Only events whose
    date actually falls inside *price_df*'s (windowed) index — and not before
    the initial deployment day — are returned, since those are the only ones
    ``simulate_rotation``/``_rotation_order_events`` can trade on.

    *year_start*/*buy_date* are located on the **combined** trading calendar (the
    union of every basket asset's own trading days), so a basket mixing markets
    with different holiday calendars (e.g. a US-listed and a Xetra-listed asset)
    will often find that day isn't a trading day for every member. Ranking is
    therefore done against a forward-filled copy of *daily_df* (bounded to the
    same ≤5-trading-day tolerance every plugin already applies to its windowed
    ``price_df`` for weekends/holidays/delayed data), so an asset merely silent on
    the exact anchor day still contributes its last known close instead of being
    wrongly excluded from that year's ranking altogether.

    The sell date is searched in the *same* year as the buy date when the sell
    ordinal (month, day) falls after the buy ordinal (the default: buy in July,
    sell in October), and in the *following* year otherwise — mirroring
    ``summergap._seasonal_target``'s wrap-around handling for windows that cross
    the year boundary (e.g. buy in November, sell in February).

    A year with no eligible losers contributes no tilt — but its reset is still
    emitted, so a tilt from a *previous* wrap-around window still ends on
    schedule and, at worst, the reset re-affirms the original weighting.

    Returns
    -------
    RotationEvents in chronological order.
    """
    if price_df.empty:
        return []
    assert isinstance(price_df.index, pd.DatetimeIndex)
    assert isinstance(daily_df.index, pd.DatetimeIndex)

    columns = [str(c) for c in price_df.columns]

    # The basket's original allocation (equal weight when no weights map is
    # given) — deployed on day one and fully restored on every sell date.
    original_shares = _normalised_weights(columns, weights)

    # Initial deployment day: first day every positively-weighted asset is
    # priced, exactly like Buy & Hold. Without one, the lump sum stays in cash.
    weight_vec = _weight_array(columns, weights)
    deploy_pos = _first_all_priced_pos(price_df.to_numpy(dtype=float), weight_vec)
    if deploy_pos is None:
        return []
    deploy_date = price_df.index[deploy_pos]

    events: list[RotationEvent] = [
        RotationEvent(deploy_date, 1.0, original_shares, 'Buy')
    ]

    # Yearly tilt/reset events are only tradable inside the window and only
    # once the basket is deployed.
    windowed_dates = {d for d in price_df.index if d > deploy_date}
    buy_ord = buy_month * 100 + buy_day
    sell_ord = sell_month * 100 + sell_day

    # See the docstring above: bridges calendar mismatches between the basket's
    # own markets (e.g. US vs. Xetra holidays) so an asset silent on the exact
    # combined-calendar anchor day isn't wrongly excluded from ranking.
    ranking_df = daily_df.ffill(limit=5)

    for year in range(int(price_df.index.year.min()), int(price_df.index.year.max()) + 1):
        year_start = _first_trading_day_of_year(daily_df.index, year)
        buy_date = _first_trading_day_on_or_after(daily_df.index, year, buy_month, buy_day)
        if year_start is None or buy_date is None:
            continue

        selected = _select_losers(ranking_df, year_start, buy_date, n_losers)

        if selected and buy_date in windowed_dates:
            # Tilt: shift_frac of the portfolio into the losers (equal split,
            # independent of the basket weights — see docstring), the rest
            # stays at the original allocation. Both parts sum to 1.0.
            loser_share = shift_frac / len(selected)
            blended = {
                sym: (loser_share if sym in selected else 0.0)
                + (1.0 - shift_frac) * original_shares.get(sym, 0.0)
                for sym in columns
            }
            events.append(RotationEvent(buy_date, 1.0, blended, 'Buy'))

        sell_year = year if sell_ord > buy_ord else year + 1
        sell_date = _first_trading_day_on_or_after(daily_df.index, sell_year, sell_month, sell_day)
        if sell_date is not None and sell_date in windowed_dates:
            # Reset: fully restore the original weighting (independent of
            # shift_frac).
            events.append(RotationEvent(sell_date, 1.0, original_shares, 'Sell'))

    events.sort(key=lambda e: e.date)
    return events


@register
class LoserRotationStrategy(BacktestStrategy):
    """Loser Rotation: buy & hold with an annual tactical tilt into the worst YTD performers."""

    @classmethod
    def get_name(cls) -> str:
        return "Loser Rotation"

    @classmethod
    def get_icon(cls) -> str:
        return "bi-arrow-repeat"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Loser Rotation: invest a lump sum like Buy & Hold, then once a "
            "year shift a configurable share of the portfolio into the "
            "basket's worst year-to-date performers (a fixed count) at a "
            "configurable date, and restore the original weighting at another; "
            "always fully invested, never in cash."
        )

    @classmethod
    def get_long_description(cls) -> str:
        return (
            "## Loser Rotation\n\n"
            "A **buy-and-hold strategy with an annual tactical tilt**, inspired "
            "by the well-known “losers of the year rebound in July” effect: at "
            "the start of the third quarter, funds and ETFs commonly rebalance "
            "out of the year's winners into its most beaten-down, fundamentally "
            "cheap names, producing a short-lived rebound in a basket's worst "
            "performers.\n\n"
            "**How it trades:** the lump sum is deployed up front across the "
            "whole basket at its per-asset weights, exactly like Buy & Hold, "
            "and stays **fully invested throughout** — the strategy never steps "
            "into cash. Once a year, on the configured buy date, every basket "
            "asset's return from the first trading day of that calendar year up "
            "to the buy date is ranked, and the configured **shift percentage** "
            "of the portfolio value is moved into the **worst-performing N "
            "assets** (split equally among them — even an asset weighted 0 in "
            "the basket can be a rotation target); the remainder stays at the "
            "original weighting. On the configured sell date the **original "
            "weighting is fully restored**. In between the trades the position "
            "simply drifts with the market; next year the worst performers are "
            "re-selected from scratch.\n\n"
            "**Good for:** exploring a mean-reversion / sector-rotation effect "
            "on a basket's own laggards while staying invested; it adds value "
            "only in years the rebound actually materialises and can lag a "
            "plain buy-and-hold otherwise.\n\n"
            "**Parameters**\n\n"
            "- **Initial Investment** — the one-off lump sum deployed at the "
            "start.\n"
            "- **Number of Losers** — how many of the basket's worst "
            "year-to-date performers to tilt into each year (fewer if the "
            "basket doesn't have that many eligible assets).\n"
            "- **Shift into Losers (%)** — the share of the portfolio value "
            "moved into the losers on each buy date (100% = the whole "
            "portfolio; the rest stays at the original weighting).\n"
            "- **Buy Month / Buy Day** — the calendar date each year the worst "
            "performers are ranked and the tilt is applied.\n"
            "- **Sell Month / Sell Day** — the calendar date each year the "
            "original weighting is restored."
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
                key='shift_percent',
                label='Shift into Losers (%)',
                type='float',
                default=100.0,
                min_value=0.0,
                max_value=100.0,
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
        # GUI percentage → fraction, clamped so a stray out-of-range value can
        # never push the blended weights negative.
        shift_frac = min(max(float(resolved['shift_percent']) / 100.0, 0.0), 1.0)
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
            # 3. The initial Buy & Hold deployment plus one tilt/reset event pair
            #    per year, each tilt carrying that year's freshly ranked bottom-N
            #    blend (built from the full history).
            events = _build_rotation_events(
                daily_df, price_df, buy_month, buy_day, sell_month, sell_day,
                n_losers, shift_frac, weights,
            )

            # 4. Rebalance to each event's own weights (always fully invested),
            #    holding (and drifting with the market) in between.
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
