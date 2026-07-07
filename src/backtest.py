# ---------------------------------------------------------------------------
# backtest.py – DCA (Dollar-Cost Averaging) simulation engine
#
# Dollar-Cost Averaging means investing a fixed amount of money at regular
# intervals (here: once per month) regardless of the current price. Over time
# this automatically buys more units when prices are low and fewer when prices
# are high, smoothing out the effect of volatility.
#
# Although DCA only *contributes* monthly, the simulation runs on daily close
# prices: each contribution lands on its month's last trading day, but the
# portfolio is valued on every trading day so the value curve and the risk
# metrics are daily.  (A second, tactical Risk-Off engine lives lower down.)
# ---------------------------------------------------------------------------

# logging: Python's standard library for recording messages at different
# severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). Using a named
# logger (instead of print) lets callers control output format and level.
import logging

# functools: lru_cache memoises the per-file parquet reads process-wide so each
# asset / FX-pair file is fetched from BASE_URL at most once per process.
import functools

# os: read the BASE_CURRENCY environment variable so the default reporting
# currency can be configured without code changes (mirrors how the GUI default
# is sourced in config.py — both read the same env var, one source of truth).
import os

# time: high-resolution timers (perf_counter) used to log how long the parquet
# reads take — these are the network-I/O hot spots, invisible to @log_time.
import time

# TYPE_CHECKING guard avoids a circular import at runtime: strategies/dca.py
# imports from backtest.py, so importing BacktestStrategy here unconditionally
# would create a cycle.  Under TYPE_CHECKING the import is only evaluated by
# static analysis tools (mypy), not at runtime.
from typing import TYPE_CHECKING, Iterable, NamedTuple, NotRequired, TypedDict
if TYPE_CHECKING:
    from src.strategies.base import BacktestStrategy

# numpy: fast numerical array operations; used here for vectorised maths
# such as annualising standard deviations.
import numpy as np

# pandas: the core data-analysis library. A DataFrame is a 2-D table with
# labelled rows (index) and columns. A Series is a single labelled column.
import pandas as pd

# Create a module-level logger. __name__ resolves to 'backtest' at runtime,
# so log messages will be prefixed with the module name automatically.
log = logging.getLogger(__name__)

# The fixed amount (in the portfolio's base currency, e.g. EUR) invested
# into each basket every month. Defined as a module constant so it is easy
# to find and change in one place.
MONTHLY_INVESTMENT = 1000.0

# The default *reporting* (base) currency every asset's prices are converted
# into before simulation.  The GUI lets the user pick another currency at run
# time (the chosen value is threaded down as the `base_currency` argument); this
# constant is only the fallback used when no explicit value is supplied (e.g. by
# the test suite or the legacy run_backtest entry point).  Configurable via the
# BASE_CURRENCY environment variable so deployments can change it without code
# edits — config.py reads the same variable for the dropdown's default.
BASE_CURRENCY = os.getenv('BASE_CURRENCY', 'EUR')

# The one-off lump sum (in the portfolio's base currency, e.g. EUR) made
# available as cash at the very start of the Risk-Off strategy.  Unlike DCA
# there are no recurring contributions: this amount is the entire capital,
# tactically shifted between the basket and cash each month.
INITIAL_INVESTMENT = 10_000.0


def _is_month_end_trading_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: True on the last present trading day of each calendar month.

    Used by the DCA simulation to place exactly one contribution per month even
    when iterating a *daily* price index.  Months are identified by the integer
    ``year * 12 + month`` (avoids the timezone warning that ``to_period`` would
    raise on a tz-aware index); ``duplicated(keep='last')`` flags every row of a
    month except its last, so negating yields True only on that final row.

    For a *monthly* index every row is already its month's only entry, so the
    mask is all-True and DCA behaves exactly like the original monthly engine.
    """
    month_ids = index.year * 12 + index.month
    return ~month_ids.duplicated(keep='last')


def _portfolio_value(holdings: dict[str, float], cash: float, prices: pd.Series) -> float:
    """Total worth = cash + Σ (units × price) over assets with a valid price.

    Assets whose price is missing (NaN) on this row are skipped from the
    valuation but keep their units, so they contribute again once a price
    reappears.  Shared by the daily valuation branches of both simulations and
    by ``_rebalance_to_target``.
    """
    invested = sum(
        holdings.get(str(c), 0.0) * float(p)
        for c, p in prices.items()
        if pd.notna(p)
    )
    return cash + invested


def _normalised_weights(
    symbols: Iterable[str], weights: dict[str, float] | None
) -> dict[str, float]:
    """Each symbol's share of an allocation, summing to 1.0 across *symbols*.

    This is the single place the per-asset **weighting** rule lives for the
    per-trade allocators (``_rebalance_to_target`` and the DCA order-event
    generator).  With no *weights* map (or an empty one) the split is **equal** —
    every symbol gets ``1/len(symbols)`` — which reproduces exactly the
    equal-weight behaviour every engine had before weighting existed.  When a
    *weights* map is given each symbol's share is its (non-negative) relative
    weight divided by the sum of the relative weights of *these* symbols, so the
    user's weights are renormalised over whatever subset is actually tradable on a
    given day (a symbol absent from the map counts as 0).

    If every weight in the subset is zero the result is all-zeros (nothing is
    allocated to this subset); the caller decides what that means — the allocators
    here interpret it as "stay in cash / skip the contribution", keeping them in
    lockstep with the vectorised engines, which simply make no purchase when the
    day's tradable weights sum to zero.

    Parameters
    ----------
    symbols – the asset symbols to split an allocation across (e.g. the priced
              assets on one trading day).
    weights – optional symbol → relative weight (non-negative); None/empty means
              equal weight.

    Returns
    -------
    dict mapping every input symbol to its share (the shares sum to 1.0, or to
    0.0 when a non-empty weight map zeroes the whole subset).
    """
    syms = [str(s) for s in symbols]
    if not syms:
        return {}
    # No weights supplied → equal split (the historical default).
    if not weights:
        equal = 1.0 / len(syms)
        return {s: equal for s in syms}
    # Renormalise the supplied weights over just these symbols.
    w = {s: max(float(weights.get(s, 0.0)), 0.0) for s in syms}
    total = sum(w.values())
    if total <= 0.0:
        return {s: 0.0 for s in syms}
    return {s: w[s] / total for s in syms}


def _weight_array(columns: Iterable[str], weights: dict[str, float] | None) -> np.ndarray:
    """Per-column relative weights aligned to *columns*, as a NumPy vector.

    The vectorised counterpart of ``_normalised_weights`` for the matrix engines
    (``simulate_dca``) and the all-priced gate (``_first_all_priced_pos``).
    Returns an all-ones vector (equal weight) when no *weights* map is given, so
    those engines keep their original behaviour byte-for-byte.  With a map each
    column takes its non-negative relative weight (a column absent from the map →
    0); a degenerate all-zero result falls back to equal weight so an engine never
    divides by a zero total across the *whole* basket (per-day zero subsets are
    still handled by the engines themselves).
    """
    cols = [str(c) for c in columns]
    if not weights:
        return np.ones(len(cols), dtype=float)
    arr = np.array([max(float(weights.get(c, 0.0)), 0.0) for c in cols], dtype=float)
    # Whole-basket all-zero → equal weight (a single asset zeroed still keeps the
    # rest; only an entirely-zero vector triggers this safety fallback).
    if not np.any(arr > 0):
        return np.ones(len(cols), dtype=float)
    return arr


def _asset_values(holdings: dict[str, float], prices: pd.Series) -> dict[str, float]:
    """Per-asset worth = units × price, one entry per asset column.

    Mirrors the individual terms summed by ``_portfolio_value``: an asset with a
    valid price contributes ``units × price``; an asset whose price is missing
    (NaN) this day contributes 0.0 (it keeps its units but has no current value).
    *Every* column of ``prices`` is included so the returned dict has the same,
    stable key set on every trade day — these become the per-asset value columns
    the order tables add (keyed by the asset's symbol, the price-frame column).
    The values therefore sum to the assets portion of ``_portfolio_value``.
    """
    return {
        str(c): (holdings.get(str(c), 0.0) * float(p) if pd.notna(p) else 0.0)
        for c, p in prices.items()
    }


def _asset_prices(prices: pd.Series) -> dict[str, float | None]:
    """Per-asset market price (close) on a trade day, one entry per asset column.

    The companion of ``_asset_values``: where that reports each asset's *worth*
    (units × price), this reports the bare *price* — the asset's exchange close
    that day, in the **reporting (base) currency** (prices arrive already
    converted, see ``load_daily_closes``) — so the order tables can show, alongside
    the value, the quote at which the trade was struck.  A missing price (NaN) maps
    to ``None`` (rendered as an em-dash); the price is reported for every column
    regardless of whether any units are held, so it stays meaningful even after a
    position is sold to 0.
    """
    return {
        str(c): (float(p) if pd.notna(p) else None)
        for c, p in prices.items()
    }


def _asset_prices_local(
    prices: pd.Series,
    asset_rate: dict[str, "pd.Series"],
    date: pd.Timestamp,
) -> dict[str, float | None]:
    """Per-asset close in the asset's own *trading* currency on a trade day.

    The companion of ``_asset_prices`` (which reports the base-currency close):
    where an FX rate applies — ``asset_rate[symbol]`` gives that pair's daily rate
    (base units per local unit) and ``date`` selects the trade day — the base
    close is divided back by the rate to recover the original local-currency
    close.  Assets already quoted in the base currency (or of unknown currency)
    have no entry in *asset_rate* and pass through unchanged, so their local and
    base price coincide.  A missing price (NaN) maps to ``None``, mirroring
    ``_asset_prices``.
    """
    out: dict[str, float | None] = {}
    for c, p in prices.items():
        sym = str(c)
        if pd.isna(p):
            out[sym] = None
            continue
        series = asset_rate.get(sym)
        rate = series.get(date) if series is not None else None
        out[sym] = (
            float(p) / float(rate)
            if rate is not None and pd.notna(rate) and float(rate) != 0.0
            else float(p)
        )
    return out


def _fx_rate_values(
    pair_rate: dict[str, "pd.Series"], date: pd.Timestamp
) -> dict[str, float | None]:
    """The day's FX rate for each affected currency pair, keyed by pair symbol.

    For every ``{LOCAL}{BASE}=X`` pair used to convert one of the basket's
    currencies, return its rate (base units per local unit) on *date* so the
    order table can show the exact rate each trade was converted at.  A missing
    rate (NaN) maps to ``None`` (rendered as an em-dash).
    """
    out: dict[str, float | None] = {}
    for pair, series in pair_rate.items():
        v = series.get(date)
        out[pair] = float(v) if v is not None and pd.notna(v) else None
    return out


def _window_by_month(
    price_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Restrict a daily price frame to the calendar months of [start, end].

    Months are compared as integer ``year * 12 + month`` so the selected
    start/end *months* are always fully included regardless of the exact
    trading day on which each month begins or ends (the date slider is
    month-granular), and no timezone conversion is needed.  Columns that are
    entirely empty within the window are dropped.

    Parameters
    ----------
    price_df   – daily closes, one column per asset (DatetimeIndex).
    start_date – first month to include (inclusive); only its month matters.
    end_date   – last month to include (inclusive); only its month matters.

    Returns
    -------
    The row/column subset of *price_df* falling inside the month window (which
    is empty when the window contains no data or start month > end month).
    """
    if price_df.empty:
        return price_df
    assert isinstance(price_df.index, pd.DatetimeIndex)
    month_ids = price_df.index.year * 12 + price_df.index.month
    start_m = start_date.year * 12 + start_date.month
    end_m = end_date.year * 12 + end_date.month
    mask = (month_ids >= start_m) & (month_ids <= end_m)
    return price_df.loc[np.asarray(mask)].dropna(how='all', axis=1)


def simulate_dca(
    price_df: pd.DataFrame,
    monthly_investment: float = MONTHLY_INVESTMENT,
    weights: dict[str, float] | None = None,
) -> tuple[pd.Series, float]:
    """Simulate monthly DCA on a *daily* price series.

    A fixed amount is contributed once per calendar month — on that month's last
    trading day — split across all assets with a valid price that day **in
    proportion to their per-asset weights** (equal weight when no *weights* map is
    given, renormalised over the day's buyable assets otherwise).  The portfolio
    is then valued on *every* trading day, producing a dense daily value curve
    even though money only goes in monthly.

    For a monthly-cadence price_df every row is its month's only entry, so a
    contribution is made on every row and the result is identical to the
    original monthly engine (this keeps the monthly-input unit tests valid).

    Implemented with vectorised numpy array maths (no per-day Python loop): the
    purchases form a (days × assets) matrix whose cumulative sum down the days is
    the running holdings, valued against the price matrix in one shot.  This is
    numerically identical to the day-by-day ``_portfolio_value`` valuation it
    replaces but ~200× faster on multi-year daily windows.

    Parameters
    ----------
    price_df           – daily (or monthly) close prices, one column per asset.
    monthly_investment – total amount (reporting currency) invested per month across the basket.
    weights            – optional symbol → relative weight; None/empty means equal
                         weight (the historical default).  Each contribution is
                         split across the day's buyable assets in proportion to
                         these weights (renormalised over that subset); a
                         zero-weight asset receives nothing.

    Returns
    -------
    (portfolio_series, total_invested)
      portfolio_series – portfolio value on every row of price_df.
      total_invested   – cumulative amount (reporting currency) contributed (months with no data skipped).
    """
    # Guard the empty case first so the month-end logic below never runs on a
    # non-datetime index (e.g. the RangeIndex of an empty DataFrame).
    if price_df.empty:
        return pd.Series(dtype=float), 0.0

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # Price matrix (days × assets); NaN marks a missing close on that day.
    prices = price_df.to_numpy(dtype=float)

    # Assets buyable on a given day need a valid, strictly positive price. A NaN
    # price means a data gap; a zero/negative price means suspended/delisted.
    buyable = ~np.isnan(prices) & (prices > 0)

    # Per-asset relative weights (all-ones = equal weight); a contribution is
    # allocated only to assets that are both buyable AND positively weighted, in
    # proportion to those weights renormalised over that day's allocatable set.
    weight_vec = _weight_array([str(c) for c in price_df.columns], weights)
    allocatable = buyable & (weight_vec[None, :] > 0)
    weight_mat = np.where(allocatable, weight_vec[None, :], 0.0)
    weight_sum = weight_mat.sum(axis=1)

    # Contribute once per calendar month — on its last trading day — but only on
    # month-ends that have at least one allocatable (buyable, positive-weight) asset.
    contribute_day = _is_month_end_trading_day(price_df.index) & (weight_sum > 0)

    # Money put into each allocatable asset on a contribution day = the monthly
    # amount times that asset's share of the day's weights (0 € on every other day
    # and for any zero-weight or non-buyable asset).
    per_asset = np.zeros_like(prices)
    per_asset[contribute_day] = (
        monthly_investment * weight_mat[contribute_day] / weight_sum[contribute_day][:, None]
    )

    # Units bought per (day, asset) = per-asset € / price, only where allocatable on
    # a contribution day; dividing by NaN elsewhere yields values we mask back to 0.
    # cumsum down the days turns these purchases into the holdings carried forward
    # to every later day (units of a NaN-priced asset are retained, not lost).
    safe_prices = np.where(allocatable, prices, np.nan)
    bought = np.where(allocatable & contribute_day[:, None], per_asset / safe_prices, 0.0)
    holdings = np.cumsum(np.nan_to_num(bought), axis=0)

    # Value the portfolio every day: Σ units × price over assets with a valid price
    # (a NaN price contributes 0 but keeps its units), mirroring _portfolio_value.
    # Cash is always 0 for DCA — every contribution is immediately invested.
    daily_value = (holdings * np.where(np.isnan(prices), 0.0, prices)).sum(axis=1)

    # Total deposited = the monthly amount once per contribution day with data.
    total_invested = float(monthly_investment * contribute_day.sum())

    return pd.Series(daily_value, index=price_df.index), total_invested


def compute_metrics(portfolio: pd.Series, total_invested: float) -> dict[str, str]:
    """Compute standard performance metrics from a *daily* portfolio value series.

    Parameters
    ----------
    portfolio      – daily portfolio value over time (DatetimeIndex).
    total_invested – total amount (reporting currency) deposited throughout the period.

    Returns
    -------
    dict of metric name → formatted string (9 keys), or {} if the input is
    empty, has no positive investment, or spans fewer than three calendar months.
    """
    # Cheap guards first: an empty/too-short series or a non-positive investment
    # cannot yield meaningful statistics.  The length check also protects the
    # DatetimeIndex access below from running on a degenerate (e.g. Range) index.
    if portfolio.empty or len(portfolio) < 3 or total_invested <= 0:
        return {}

    # Require at least three distinct calendar months of history (the daily
    # equivalent of the original "≥ 3 monthly points" rule): a one- or two-month
    # window is rejected as too short for annualised statistics.
    assert isinstance(portfolio.index, pd.DatetimeIndex)
    month_ids = portfolio.index.year * 12 + portfolio.index.month
    if month_ids.nunique() < 3:
        return {}

    # The last value in the portfolio series is the current total worth.
    final_value = portfolio.iloc[-1]

    # Total Return: how much money was made relative to what was put in.
    # Example: invested 12,000 €, now worth 15,000 € → (15k-12k)/12k = +25%.
    total_return = (final_value - total_invested) / total_invested

    # pct_change() computes day-over-day percentage returns:
    # return[i] = (value[i] - value[i-1]) / value[i-1]
    # The first row has no predecessor (NaN). DCA also sits at value 0 until its
    # first month-end contribution, so the jump off 0 yields ±inf; we map those
    # to NaN and drop them so they do not poison the volatility/Sharpe stats.
    daily_returns = portfolio.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Elapsed time in years from the calendar span of the index (robust to the
    # varying number of trading days per month/year); used as the CAGR exponent.
    span_days = (portfolio.index[-1] - portfolio.index[0]).days
    n_years = span_days / 365.25 if span_days > 0 else 0.0

    # CAGR = Compound Annual Growth Rate.
    # It answers: "at what constant yearly rate would the investment have
    # needed to grow to reach this end value from the total invested?"
    # Formula: (end_value / invested) ^ (1 / years) - 1
    cagr = (final_value / total_invested) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Annualised Volatility: the standard deviation of *daily* returns scaled up
    # to a yearly figure. Multiplying by √252 (≈ trading days per year) converts
    # daily std to annual std, assuming returns are independent day-to-day.
    vol = daily_returns.std() * np.sqrt(252)

    # Sharpe Ratio: return per unit of risk (volatility), annualised with √252.
    # Higher is better. A ratio > 1 is generally considered good. We assume a
    # risk-free rate of 0% (simplified). The std guard avoids division by zero.
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0 else 0.0
    )

    # Maximum Drawdown: the largest peak-to-trough decline ever experienced.
    # rolling_max tracks the highest portfolio value seen so far at each point.
    # The ratio (value - peak) / peak gives the percentage decline from peak.
    # .min() finds the worst such decline. The result is negative by convention
    # (e.g. -0.30 = the portfolio fell 30% from its all-time high at worst).
    # Before any money is invested the peak is 0; dividing by NaN there yields
    # NaN (skipped by .min()) instead of a 0/0 warning.
    rolling_max = portfolio.expanding().max()
    max_dd = ((portfolio - rolling_max) / rolling_max.replace(0, np.nan)).min()

    # Calmar Ratio: CAGR divided by the absolute maximum drawdown.
    # It measures how much annual return is earned per unit of drawdown risk.
    # Only defined when there was actually a drawdown (max_dd < 0); if the
    # portfolio rose monotonically it stays at 0.0 to avoid division by zero.
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # Return all metrics as a dict of pre-formatted strings so the UI layer
    # can display them directly without additional formatting logic.
    # The :+.1f format prefix forces a '+' sign on positive numbers.
    return {
        'Total Return': f"{total_return * 100:+.1f}%",
        'CAGR': f"{cagr * 100:.1f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max. Drawdown': f"{max_dd * 100:.1f}%",
        'Volatility (p.a.)': f"{vol * 100:.1f}%",
        'Calmar Ratio': f"{calmar:.2f}",
        'Invested': f"{total_invested:,.0f}",   # :, adds thousands separator
        'End Value': f"{final_value:,.0f}",
        'Profit/Loss': f"{final_value - total_invested:+,.0f}",
    }


# ---------------------------------------------------------------------------
# Order log – generic, strategy-agnostic builder
#
# Every strategy records what it *did* (the buy/sell trades it placed) as a list
# of raw OrderEvents; this module then turns those events into fully-populated
# OrderRows by deriving the columns that are computed the same way for *all*
# strategies (running net deposits, profit/loss, exposure, cash quota, period
# return).  Keeping only this generic step here means a new strategy adds its
# order log entirely within its own plugin file: it emits OrderEvents (the part
# that is genuinely strategy-specific) and calls build_order_log() — backtest.py
# never needs to change.
# ---------------------------------------------------------------------------


class OrderEvent(TypedDict):
    """One raw trade as recorded by a strategy, before derived columns are added.

    Fields
    ------
    date         – trading day on which the trade happened.
    side         – 'Buy' or 'Sell' (human-readable, rendered verbatim).
    value_before – total portfolio worth at this day's prices *before* the trade.
    inflow       – fresh external money added on this trade (a DCA contribution);
                   0.0 for pure re-allocations such as a Risk-Off rebalance.
    assets_after – worth of the held assets (excluding cash) *after* the trade.
    cash_after   – uninvested cash *after* the trade.
    asset_values – per-asset worth (units × price, reporting currency by symbol) *after* the trade;
                   one entry per basket asset (see ``_asset_values``).  Sums to
                   ``assets_after`` and feeds the order tables' per-asset columns.
    asset_prices – per-asset market price (exchange close, by symbol) on the trade
                   day, in the **reporting (base) currency** (see ``_asset_prices``);
                   the companion quote column shown next to each asset's value.
                   NaN prices map to None.
    asset_prices_local – per-asset close in the asset's own **trading** currency
                   (see ``_asset_prices_local``); optional, populated only when the
                   plugin passes FX context so the order table can show the local
                   quote next to the converted one.  NaN prices map to None.
    fx_rates     – the trade day's FX rate (base per local) for each affected
                   ``{LOCAL}{BASE}=X`` pair (see ``_fx_rate_values``); optional,
                   populated only with FX context so the order table can add one
                   column per currency pair used to convert the basket.
    """

    date: pd.Timestamp
    side: str
    value_before: float
    inflow: float
    assets_after: float
    cash_after: float
    asset_values: dict[str, float]
    asset_prices: dict[str, float | None]
    asset_prices_local: NotRequired[dict[str, float | None]]
    fx_rates: NotRequired[dict[str, float | None]]


class OrderRow(TypedDict):
    """A finalized order-log row: the raw OrderEvent plus all derived columns.

    The derived columns (everything from value_after down) are filled in by
    build_order_log and have an identical meaning for every strategy, so the UI
    can render one uniform table no matter which strategy produced it.  The
    ratio columns are Optional: they are None (rendered as '—') whenever their
    denominator would be zero.
    """

    date: pd.Timestamp
    side: str
    value_before: float
    inflow: float
    assets_after: float
    cash_after: float
    asset_values: dict[str, float]  # per-asset worth after the trade (base ccy by symbol)
    asset_prices: dict[str, float | None]  # per-asset exchange close (base ccy) on the trade day
    asset_prices_local: dict[str, float | None]  # per-asset close in the asset's trading currency
    fx_rates: dict[str, float | None]  # day's FX rate per {LOCAL}{BASE}=X pair used to convert
    value_after: float             # assets_after + cash_after
    bh_value: float | None         # net_deposits × normalised equal-weight B&H index (None when unavailable)
    net_deposits: float            # running sum of all external money put in
    pnl_abs: float                 # value_after − net_deposits (profit/loss, €)
    pnl_pct: float | None          # pnl_abs / net_deposits
    equity_exposure: float | None  # assets_after / value_after (invested share)
    cash_quote: float | None       # cash_after / value_after (= 1 − exposure)
    period_return: float | None    # value_before / previous value_after − 1


def build_order_log(
    events: list[OrderEvent],
    initial_capital: float,
    bh_index: pd.Series | None = None,
) -> list[OrderRow]:
    """Turn a strategy's raw OrderEvents into fully-populated OrderRows.

    This is the *only* order-log logic in backtest.py and is completely
    strategy-agnostic: it walks the events in chronological order and derives
    every column that is computed the same way for all strategies.  Strategies
    differ only in *how they fill the OrderEvents* (see each plugin), not in how
    those events become rows.

    Parameters
    ----------
    events          – chronological raw trades produced by a strategy.
    initial_capital – external money already present before the first event
                      (the Risk-Off lump sum; 0.0 for DCA, which adds all of its
                      money through per-event inflows).  Seeds the net-deposits
                      tally.
    bh_index        – optional equal-weight buy-and-hold index normalised so its
                      first value is 1.0; when provided, each row gets a
                      ``bh_value`` column (net_deposits × index[date]) that shows
                      what the same capital would be worth under a pure B&H.
                      None when the strategy cannot produce a meaningful benchmark
                      (e.g. the basket is empty).

    Returns
    -------
    One OrderRow per event, in the same order.  Empty when *events* is empty.
    """
    rows: list[OrderRow] = []

    # Running total of external money put in (lump sum + every inflow so far);
    # the basis against which absolute and relative P&L are measured.
    net_deposits = initial_capital

    # Worth of the portfolio at the *previous* event's close, used to express
    # each row's period return (market drift since the last trade).  None before
    # the first event, so that first row reports no period return.
    prev_value_after: float | None = None

    for ev in events:
        # Total worth after the trade, and the updated deposit basis.
        value_after = ev['assets_after'] + ev['cash_after']
        net_deposits += ev['inflow']

        # Profit/loss versus everything paid in (absolute € and relative %).
        pnl_abs = value_after - net_deposits
        pnl_pct = pnl_abs / net_deposits if net_deposits > 0 else None

        # How the portfolio is split between assets and cash after the trade.
        equity_exposure = ev['assets_after'] / value_after if value_after > 0 else None
        cash_quote = ev['cash_after'] / value_after if value_after > 0 else None

        # Market drift since the previous trade: this trade's pre-trade worth
        # relative to the previous trade's post-trade worth.
        period_return = (
            ev['value_before'] / prev_value_after - 1.0
            if prev_value_after is not None and prev_value_after > 0 else None
        )

        # Equal-weight B&H benchmark: what net_deposits would be worth if the
        # same total capital had been invested as a lump sum on day one.
        bh_val = bh_index.get(ev['date']) if bh_index is not None else None
        bh_value: float | None = net_deposits * float(bh_val) if bh_val is not None else None

        rows.append(OrderRow(
            date=ev['date'],
            side=ev['side'],
            value_before=ev['value_before'],
            inflow=ev['inflow'],
            assets_after=ev['assets_after'],
            cash_after=ev['cash_after'],
            # Carried through verbatim (defaulting to {} for events that predate
            # the per-asset breakdown) so the order tables can add a value and a
            # price column per asset; the generic builder stays strategy-agnostic.
            asset_values=ev.get('asset_values', {}),
            asset_prices=ev.get('asset_prices', {}),
            # The local-currency quote and per-pair FX rates are optional on the
            # event (only present when the plugin supplied FX context); default to
            # {} so the builder stays strategy-agnostic and back-compatible.
            asset_prices_local=ev.get('asset_prices_local', {}),
            fx_rates=ev.get('fx_rates', {}),
            value_after=value_after,
            bh_value=bh_value,
            net_deposits=net_deposits,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            equity_exposure=equity_exposure,
            cash_quote=cash_quote,
            period_return=period_return,
        ))

        prev_value_after = value_after

    return rows


# ---------------------------------------------------------------------------
# Risk-Off signal strategy engine
#
# A second, tactical strategy lives alongside DCA.  Instead of investing a
# fixed amount every month, it starts from a single lump sum held entirely in
# cash and shifts capital between the basket and cash based on three market
# signals evaluated on the basket as a whole:
#   1. 200-day trend  – basket index above its 200-trading-day moving average.
#   2. YTD return     – basket index above its value on the first trading day
#                       of the current calendar year.
#   3. January barometer – the first 10 trading days of the calendar year were
#                          positive (fixed for the whole year once known).
# The number of *positive* signals (0..3) maps directly to the invested
# fraction in thirds: 3 → 100 %, 2 → 66 %, 1 → 33 %, 0 → 0 % (all cash).
# All functions below are pure (no Dash dependency); the plugin in
# strategies/riskoff.py only orchestrates them.
# ---------------------------------------------------------------------------


# Currency values that carry no usable information (blank or the literal '0'
# placeholder some catalogue rows use). Treated like "unknown currency" → the
# asset is left unconverted (mostly Indices, which are unitless point values).
_BLANK_CURRENCIES = {'', '0', 'nan', 'None'}


@functools.lru_cache(maxsize=None)
def _read_close_series(base_url: str, filename: str) -> pd.Series:
    """Read a parquet file's UTC-normalised Close column once per process.

    This is the single choke point for *all* per-asset and FX-pair parquet reads
    (``load_daily_closes``, ``_fx_rate_series`` and ``_get_monthly_range`` all
    funnel through here), so each file is fetched from ``BASE_URL`` at most once
    for the lifetime of the process — mirroring the one-shot, always-resident
    ``master.parquet`` held in ``config.df``.  Because only the ``Close`` column
    is ever consumed anywhere, ``columns=['Close']`` keeps the transfer minimal.

    The cache spans separate Dash callbacks (e.g. the date-slider refresh and the
    later backtest run), which a per-call cache cannot: dragging the slider then
    running the backtest reads each asset's file only once in total, and an asset
    appearing in both baskets — or an FX pair needed by both
    ``load_daily_closes`` and ``build_fx_columns`` — is likewise read once.

    Callers MUST treat the returned Series as **read-only**: never mutate it in
    place; derive copies via slicing / arithmetic / dropna / reindex / resample
    (all of which the callers already do).

    Parameters
    ----------
    base_url – root URL/path where the parquet files are hosted.
    filename – parquet filename to read (asset OHLCV or FX-pair file).

    Returns
    -------
    A UTC-indexed Series of close prices (NaNs not yet dropped — callers decide).
    """
    ohlcv = pd.read_parquet(f"{base_url}/{filename}", columns=['Close'])
    close = ohlcv['Close']
    # Normalise to UTC so every series shares a common timezone regardless of how
    # the source file stored its index (tz-naive or tz-aware).
    assert isinstance(close.index, pd.DatetimeIndex)
    if close.index.tz is None:
        close.index = close.index.tz_localize('UTC')
    else:
        close.index = close.index.tz_convert('UTC')
    return close


def clear_parquet_cache() -> None:
    """Drop the process-wide Close cache.

    Used by the test-suite to isolate read counts between tests, and available as
    a data-refresh hook (the cache otherwise assumes the parquet data is static
    for the process lifetime, exactly like ``config.df``).
    """
    _read_close_series.cache_clear()


def _fx_rate_series(
    base_url: str,
    df_meta: pd.DataFrame,
    local_ccy: str,
    base_ccy: str,
    cache: dict[str, "pd.Series | None"],
) -> "pd.Series | None":
    """Return the daily FX rate that converts *local_ccy* prices into *base_ccy*.

    The returned series is the close of the ``{LOCAL}{BASE}=X`` pair, i.e. the
    number of *base* units per 1 *local* unit, so ``price_local × rate`` yields
    ``price_base``.

    Returns ``None`` (meaning "no conversion needed/possible") when:
      * *local_ccy* is blank/unknown (see _BLANK_CURRENCIES) or NaN, or
      * *local_ccy* already equals *base_ccy*, or
      * no matching FX pair exists in the catalogue (logged as a WARNING — a
        safety net only; the shipped data has a direct pair for every asset
        currency into EUR).

    Parameters
    ----------
    base_url   – root URL/path where the parquet files are hosted.
    df_meta    – master catalogue; its ``currency`` asset-class rows map a pair
                 name/symbol to the parquet filename to load.
    local_ccy  – the currency the asset's prices are quoted in.
    base_ccy   – the desired reporting currency.
    cache      – per-call dict keyed by *local_ccy* so several assets sharing a
                 currency trigger only one FX parquet read.

    Returns
    -------
    A UTC-indexed, chronologically sorted Series of FX closes, or ``None``.
    """
    # Normalise the input: NaN floats and stray whitespace both occur in the wild.
    local = '' if local_ccy is None else str(local_ccy).strip()
    if local in _BLANK_CURRENCIES or local == base_ccy:
        return None

    # Memoised within a single load_daily_closes call (the value may itself be
    # None when the pair is missing — caching that avoids repeated lookups/warns).
    if local in cache:
        return cache[local]

    # Locate the FX pair row: prefer an exact symbol match ({LOCAL}{BASE}=X),
    # fall back to the human pair name ('{LOCAL}/{BASE}'). Both identify the same
    # row; either is enough and robust to minor catalogue inconsistencies.
    fx_rows = df_meta[df_meta['asset_class'] == 'currency']
    pair_symbol = f'{local}{base_ccy}=X'
    pair_name = f'{local}/{base_ccy}'
    match = fx_rows[(fx_rows['symbol'] == pair_symbol) | (fx_rows['name'] == pair_name)]
    if match.empty:
        log.warning("No FX pair %s to convert %s prices to %s; leaving unconverted",
                    pair_symbol, local, base_ccy)
        cache[local] = None
        return None

    try:
        # Read (UTC-normalised, process-cached) Close of the FX pair; drop gaps
        # and sort so the rate aligns with the asset calendars in the callers.
        rate = _read_close_series(base_url, match.iloc[0]['filename']).dropna().sort_index()
        cache[local] = None if rate.empty else rate
    except Exception:
        log.exception("Failed to load FX pair for %s->%s", local, base_ccy)
        cache[local] = None
    return cache[local]


def load_daily_closes(
    base_url: str,
    filenames: list[str],
    df_meta: pd.DataFrame,
    base_currency: str = BASE_CURRENCY,
) -> pd.DataFrame:
    """Load and combine *daily* close prices for the given asset filenames.

    Keeps the full daily resolution and the full available history (no
    date-window restriction); callers window it by month via _window_by_month.
    The extra history is
    required so signals with long look-back windows (e.g. the 200-day moving
    average and the year-to-date anchor) have enough warm-up data before the
    backtest window begins.

    Every asset's prices are converted from the currency they are quoted in (the
    catalogue's ``currency`` column) into *base_currency* using the daily FX rate
    (``_fx_rate_series``), so a basket mixing currencies is valued consistently —
    the standard *unhedged* base-currency approach.  Assets already quoted in the
    base currency, or with no/unknown currency (e.g. Indices), pass through
    unchanged.  FX pairs added as basket assets take this same path, converted by
    their own quote currency.

    Parameters
    ----------
    base_url      – root URL/path where the parquet files are hosted.
    filenames     – list of parquet file names (e.g. ['aapl.parquet']).
    df_meta       – master metadata table mapping filenames to symbols/names and
                    (when present) currencies, plus the ``currency`` asset-class
                    FX-pair rows used for conversion.
    base_currency – reporting currency every series is converted into.

    Returns
    -------
    DataFrame with one column per successfully loaded asset (named by symbol)
    and one row per trading day. Missing days are NaN. Empty if nothing loaded.
    """
    # Accumulate individual daily price series here before combining them.
    series = {}
    # Per-call FX-rate cache: assets sharing a currency reuse one parquet read.
    fx_cache: dict[str, "pd.Series | None"] = {}
    # Conversion only runs when the catalogue actually carries currencies; the
    # test fixtures (and any legacy catalogue) omit the column → unchanged behaviour.
    has_currency = 'currency' in df_meta.columns
    t_total = time.perf_counter()

    for filename in filenames:
        try:
            # Look up the asset's metadata row by its filename; skip unknowns.
            meta = df_meta[df_meta['filename'] == filename]
            if meta.empty:
                continue

            symbol = meta.iloc[0]['symbol']

            # Time the read in isolation. The Close column comes from the
            # process-wide cache (_read_close_series), so the first request for a
            # file pays the network fetch and every later one is a cache hit —
            # the prime suspect for multi-second runs is thus paid at most once.
            t_read = time.perf_counter()
            close = _read_close_series(base_url, filename)
            read_ms = (time.perf_counter() - t_read) * 1000

            # Drop rows with no close (gaps before listing). dropna returns a
            # copy, so the cached series stays pristine for other callers. Keep
            # the file's native cadence (NOT resampled to daily — intraday files
            # stay intraday, which is why the row count below matters for speed).
            close = close.dropna()

            # Convert the local-currency close into the reporting currency.
            if has_currency:
                rate = _fx_rate_series(
                    base_url, df_meta, meta.iloc[0]['currency'], base_currency, fx_cache,
                )
                if rate is not None:
                    # Align the FX rate onto the asset's trading days: ffill carries
                    # the last known rate across weekend/holiday calendar mismatches;
                    # the trailing bfill back-fills the (small) span where an asset
                    # predates its FX pair's history using the earliest known rate,
                    # rather than silently dropping that pre-FX history.
                    aligned = rate.reindex(close.index, method='ffill').bfill()
                    close = (close * aligned).dropna()
                    log.debug('load_daily_closes converted %s: %s->%s',
                              symbol, meta.iloc[0]['currency'], base_currency)

            log.debug('[perf] load_daily_closes read %s: %d rows in %.1fms',
                      filename, len(close), read_ms)
            if not close.empty:
                series[symbol] = close

        except Exception:
            # Log the full traceback but keep processing remaining filenames.
            log.exception("Failed to load %s", filename)

    if not series:
        return pd.DataFrame()

    # Outer-join on the date index and sort chronologically so rolling windows
    # and as-of look-ups operate on a monotonically increasing index.
    combined = pd.DataFrame(series).sort_index()
    log.debug('[perf] load_daily_closes total: %d files -> %d rows x %d cols in %.1fms',
              len(filenames), combined.shape[0], combined.shape[1],
              (time.perf_counter() - t_total) * 1000)
    return combined


class FxColumns(NamedTuple):
    """The FX side-tables an order log needs to show each trade's conversion.

    ``load_daily_closes`` converts every price into the reporting currency and
    keeps only the result, so the order generators — which need to *also* show the
    untouched trading-currency quote and the rate they were converted at — get
    that information from here instead.  All three maps are built from the same
    catalogue rows and aligned with the same calendar logic as the conversion, so
    ``base_price / rate`` exactly reconstructs the original local price.

    Fields
    ------
    asset_local_ccy – symbol → the currency the asset trades in ('' when the
                      catalogue records none); labels the per-asset price columns.
    asset_rate      – symbol → the asset's daily FX rate (base units per local
                      unit) aligned onto the order index; present only for assets
                      actually converted (absent ⇒ local price = base price).
    pair_rate       – ``{LOCAL}{BASE}=X`` → that pair's daily rate aligned onto the
                      order index, one entry per distinct currency converted, so
                      the order table adds a column per FX pair used in the basket.
    """

    asset_local_ccy: dict[str, str]
    asset_rate: dict[str, pd.Series]
    pair_rate: dict[str, pd.Series]


def build_fx_columns(
    base_url: str,
    filenames: list[str],
    df_meta: pd.DataFrame,
    base_currency: str,
    index: pd.Index,
) -> FxColumns:
    """Build the per-asset and per-pair FX maps the order log renders (see FxColumns).

    For every basket asset this records the trading currency, and — when that
    currency is converted into *base_currency* — the daily FX rate aligned onto
    *index* (the windowed order index).  The alignment
    (``reindex(index, method='ffill').bfill()``) is identical to the one
    ``load_daily_closes`` applies during conversion, so dividing an asset's
    base-currency close by this rate exactly recovers the original local close.
    ``_fx_rate_series`` (and its per-call cache) is reused, so a currency shared by
    several assets is read only once and the same FX pair maps to a single column.

    Returns an empty FxColumns when the catalogue carries no ``currency`` column
    (legacy/test fixtures), so the order log simply omits the FX columns.
    """
    asset_local_ccy: dict[str, str] = {}
    asset_rate: dict[str, pd.Series] = {}
    pair_rate: dict[str, pd.Series] = {}
    if 'currency' not in df_meta.columns:
        return FxColumns(asset_local_ccy, asset_rate, pair_rate)

    # Per-call cache shared across assets (mirrors load_daily_closes), so a
    # currency used by several basket assets triggers a single FX parquet read.
    fx_cache: dict[str, "pd.Series | None"] = {}
    for filename in filenames:
        meta = df_meta[df_meta['filename'] == filename]
        if meta.empty:
            continue
        symbol = str(meta.iloc[0]['symbol'])
        local_raw = meta.iloc[0]['currency']
        local = '' if local_raw is None or pd.isna(local_raw) else str(local_raw).strip()
        asset_local_ccy[symbol] = local

        rate = _fx_rate_series(base_url, df_meta, local_raw, base_currency, fx_cache)
        if rate is not None:
            aligned = rate.reindex(index, method='ffill').bfill()
            asset_rate[symbol] = aligned
            # One column per distinct converted currency; setdefault dedupes the
            # pair when several assets share the same trading currency.
            pair_rate.setdefault(f'{local}{base_currency}=X', aligned)

    return FxColumns(asset_local_ccy, asset_rate, pair_rate)


def build_equal_weight_index(
    daily_df: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.Series:
    """Combine per-asset daily closes into one (optionally weighted) price index.

    Each day's basket return is the **weighted** mean of the individual assets'
    daily returns (averaging only over assets that have a price that day, so
    mixed start dates and single-asset baskets both work; the weights are
    renormalised over the assets present each day).  With no *weights* map the
    weights are all equal, so this reduces exactly to the original equal-weight
    index — hence the name is kept.  Compounding those returns yields a single
    index series (rebased to 100) on which the Risk-Off signals are evaluated and
    from which the Buy & Hold benchmark column is derived.

    Parameters
    ----------
    daily_df – DataFrame of daily closes, one column per asset.
    weights  – optional symbol → relative weight; None/empty means equal weight.

    Returns
    -------
    Series indexed by trading day with the (weighted) index value (base 100),
    or an empty Series when *daily_df* is empty.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)

    # Treat non-positive prices as missing so they never enter a return.
    prices = daily_df.where(daily_df > 0)

    # Per-asset day-over-day returns.
    returns = prices.pct_change()

    # Per-asset relative weights (all-ones = equal); broadcast as a column-aligned
    # Series so the weighted mean is computed only over the assets present each day.
    weight_vec = _weight_array([str(c) for c in daily_df.columns], weights)
    weight_ser = pd.Series(weight_vec, index=daily_df.columns)

    # Weighted cross-sectional mean per day = Σ(wᵢ·rᵢ) / Σ(wᵢ) over assets with a
    # return that day.  Both sums skip NaN, so the average spans only present
    # assets; an all-ones weight_ser makes this identical to the plain mean.
    present = returns.notna()
    weighted_sum = returns.mul(weight_ser, axis=1).sum(axis=1)
    weight_total = present.mul(weight_ser, axis=1).sum(axis=1)
    basket_return = weighted_sum / weight_total.replace(0, np.nan)

    # The first row (and any all-NaN row) has no return; fill with 0 so the
    # cumulative product is not poisoned to NaN. cumprod of (1+r) rebased to 100
    # reduces to price/price0*100 for a single asset.
    index = (1.0 + basket_return.fillna(0.0)).cumprod() * 100.0
    return index


def _sma_trend_signal(index: pd.Series, window: int = 200) -> pd.Series:
    """Signal 1 – index above its simple moving average over *window* days.

    During the warm-up period the rolling mean is NaN; a comparison against
    NaN evaluates to False in pandas, which is exactly the conservative
    "not invested" default we want (no explicit fillna needed).
    """
    sma = index.rolling(window=window).mean()
    return index > sma


def _ytd_return_signal(index: pd.Series) -> pd.Series:
    """Signal 2 – running year-to-date return positive.

    Compares each day's index value to the index value on the first trading
    day of the *same* calendar year.  The signal flips to False as soon as the
    basket falls back below its yearly starting value.  Calendar-year anchored,
    so it needs no window parameter and never produces NaN.
    """
    assert isinstance(index.index, pd.DatetimeIndex)
    # transform('first') broadcasts each year's first value back onto every
    # row of that year, so the comparison is elementwise.
    year_start = index.groupby(index.index.year).transform('first')
    return index > year_start


def _first_n_days_signal(index: pd.Series, n: int = 10) -> pd.Series:
    """Signal 3 – "January barometer": first *n* trading days of the year up.

    For each calendar year the close on the n-th trading day is compared to the
    close on the first trading day; the resulting boolean is broadcast to every
    day of that year (it is fixed once the first n days are known, ~mid-January,
    so it is always available at month-ends and introduces no look-ahead).
    A partial year without n trading days is treated as False (conservative).
    """
    assert isinstance(index.index, pd.DatetimeIndex)

    # Start all-False so partial years (and any gaps) default to "not invested".
    result = pd.Series(False, index=index.index)

    for _, group in index.groupby(index.index.year):
        if len(group) >= n:
            positive = bool(group.iloc[n - 1] > group.iloc[0])
        else:
            positive = False
        # Stamp the year's verdict onto all of that year's days.
        result.loc[group.index] = positive

    return result


def compute_riskoff_signals(index: pd.Series, sma_window: int = 200, first_n_days: int = 10) -> pd.Series:
    """Combine the three boolean signals into a positive-signal count (0..3).

    Parameters
    ----------
    index        – equal-weight basket index (from build_equal_weight_index).
    sma_window   – look-back window for the moving-average trend signal.
    first_n_days – number of January trading days for the barometer signal.

    Returns
    -------
    Integer Series in [0, 3]: the number of positive signals on each day.
    Dividing by 3 yields the target invested fraction.
    """
    sma_sig = _sma_trend_signal(index, sma_window)
    ytd_sig = _ytd_return_signal(index)
    jan_sig = _first_n_days_signal(index, first_n_days)
    return sma_sig.astype(int) + ytd_sig.astype(int) + jan_sig.astype(int)


def _rebalance_to_target(
    holdings: dict[str, float],
    cash: float,
    prices: pd.Series,
    inv_frac: float,
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], float, float]:
    """Buy/sell the whole portfolio to a target invested fraction on one day.

    The target invested value is *inv_frac* × total portfolio value, spread across
    all assets that have a valid price this day **in proportion to their per-asset
    weights** (equal weight when no *weights* map is given, renormalised over the
    priced assets otherwise); the remainder stays in cash.  This is a one-off
    adjustment: simulate_riskoff only calls it on the day a signal changes, then
    holds the resulting position (no daily maintenance), so the actual fraction
    drifts with the market in between.

    Parameters
    ----------
    holdings – units currently held per asset symbol.
    cash     – current uninvested cash.
    prices   – this day's close price per asset (may contain NaN/zero).
    inv_frac – target fraction of the portfolio to be invested (0.0..1.0).
    weights  – optional symbol → relative weight; None/empty means equal weight.

    Returns
    -------
    (new_holdings, new_cash, total_value) where total_value is the portfolio's
    worth this day (cash plus the value of all priced holdings).
    """
    # Assets that can actually be traded today (valid, positive price).
    priced = {str(c): float(p) for c, p in prices.items() if pd.notna(p) and p > 0}

    # Total worth = cash + value of currently priced holdings.
    total_value = _portfolio_value(holdings, cash, prices)

    if not priced:
        # Nothing tradable today: carry holdings and cash unchanged.
        return holdings, cash, total_value

    # Each priced asset's share of the invested amount (equal weight by default;
    # renormalised over the priced subset when explicit weights are supplied).
    shares = _normalised_weights(priced.keys(), weights)
    if sum(shares.values()) <= 0.0:
        # No positively-weighted asset is tradable today → stay fully in cash
        # (mirrors the vectorised engines, which make no purchase in this case).
        return holdings, cash, total_value

    # Buy the target invested amount, split across priced assets by their shares.
    target_invested = inv_frac * total_value

    new_holdings = dict(holdings)
    for c, price in priced.items():
        new_holdings[c] = (shares[c] * target_invested) / price

    new_cash = total_value - target_invested
    return new_holdings, new_cash, total_value


def _first_all_priced_pos(
    prices_mat: np.ndarray, weight_vec: np.ndarray | None = None
) -> int | None:
    """Position of the first row on which *every* asset has a valid, positive price.

    The lump-sum strategies (Buy & Hold and Risk-Off) deploy their initial
    investment only once the **whole basket** is priced, so the money is split (by
    the chosen weights) across the assets instead of over-weighting whichever
    happened to list first.  A valid price is non-NaN and strictly positive (a
    zero/negative price means suspended/delisted, a NaN a data gap).

    When a *weight_vec* is given, only the columns that actually receive an
    allocation (positive weight) need to be priced for a day to count as "all
    priced": a zero-weight asset gets nothing, so the deployment must not wait for
    it.  An all-ones vector (the equal-weight default) reproduces the original
    "every column priced" behaviour exactly.

    Parameters
    ----------
    prices_mat – (days × assets) price matrix; NaN marks a missing close.
    weight_vec – optional per-column relative weights (from ``_weight_array``);
                 only positive-weight columns are required to be priced.

    Returns
    -------
    The integer row position of the first all-priced day, or ``None`` when no
    such day exists (some required asset never gets a price within the window) —
    the caller then keeps the lump sum in cash.
    """
    buyable = ~np.isnan(prices_mat) & (prices_mat > 0)
    if weight_vec is not None:
        relevant = weight_vec > 0
        if relevant.any():
            # Only assets that will be bought must be priced before deploying.
            buyable = buyable[:, relevant]
    all_buyable = buyable.all(axis=1)
    pos = np.flatnonzero(all_buyable)
    return int(pos[0]) if pos.size else None


def gate_target_until_all_priced(
    price_df: pd.DataFrame,
    target_fraction: pd.Series,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Zero a daily target invested fraction until the whole basket is priced.

    The Risk-Off strategy deploys its lump sum only once **every** asset it will
    actually buy has a valid price, so the initial investment is split by the
    chosen weights rather than over-weighting whichever listed first.  This forces
    the daily target to 0.0 (fully in cash) on every day before the first day on
    which every (positively-weighted) asset has a valid, positive price (see
    ``_first_all_priced_pos``); from that day on the supplied target passes through
    unchanged.  When no such day exists the target is all-zero, leaving the lump
    sum in cash for the whole window — matching ``simulate_lumpsum``'s
    never-fully-priced behaviour.

    *target_fraction* is expected to already be aligned to *price_df*'s index
    (the plugin reindexes it before calling); a copy is returned so the caller's
    series is left untouched.  *weights* (None/empty = equal) decides which assets
    must be priced before the gate opens.
    """
    if price_df.empty:
        return target_fraction
    weight_vec = _weight_array([str(c) for c in price_df.columns], weights)
    first = _first_all_priced_pos(price_df.to_numpy(dtype=float), weight_vec)
    gated = target_fraction.copy()
    if first is None:
        gated.iloc[:] = 0.0
    else:
        gated.iloc[:first] = 0.0
    return gated


def simulate_riskoff(
    price_df: pd.DataFrame,
    target_fraction: pd.Series,
    initial_investment: float = INITIAL_INVESTMENT,
    weights: dict[str, float] | None = None,
) -> tuple[pd.Series, float]:
    """Simulate the lump-sum Risk-Off strategy with daily, change-driven trading.

    The full *initial_investment* starts as cash.  Signals are evaluated daily
    (by the caller) and supplied as a per-day *target_fraction*.  On every
    trading day this function compares that target to the fraction currently
    held: when it **changes** it buys/sells to the new target at that day's
    prices; when it is unchanged it **holds** (no trade — the basket drifts with
    the market and is *not* corrected back toward the target).  Cash earns 0 %.
    There are no additional contributions, so *total_invested* equals the lump sum.

    Parameters
    ----------
    price_df           – daily close prices (one column per asset).
    target_fraction    – per-day target invested fraction (0..1); reindexed onto
                         price_df's index, missing days defaulting to 0.0 (cash).
    initial_investment – one-off lump sum provided as cash at the start.
    weights            – optional symbol → relative weight; None/empty means equal
                         weight.  The invested portion is split across the day's
                         priced assets in proportion to these weights.

    Returns
    -------
    (portfolio_series, total_invested)
      portfolio_series – portfolio value (holdings + cash) on every trading day.
      total_invested   – the initial lump sum (constant; no contributions).

    Implemented as a hybrid: trades only happen on the (few) days the target
    changes, so the loop walks just those change-days — reusing the exact
    ``_rebalance_to_target`` math — to record the post-trade holdings/cash, then
    the dense daily valuation is done in one vectorised numpy pass.  This is
    numerically identical to the former day-by-day loop but ~60× faster.
    """
    if price_df.empty:
        return pd.Series(dtype=float), initial_investment

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # Column order shared by the holdings dict (string keys) and the price matrix.
    columns = [str(col) for col in price_df.columns]
    prices_mat = price_df.to_numpy(dtype=float)
    n_days, n_assets = prices_mat.shape

    # Align the daily target fractions to the price index; unknown days stay in
    # cash (0.0) as a conservative fallback.
    target = target_fraction.reindex(price_df.index).fillna(0.0).to_numpy()

    # Change-days: the target differs from the prior day's (the starting fraction
    # is 0.0 = all cash), i.e. exactly the days the former loop would have traded.
    prev = np.empty(n_days)
    prev[0] = 0.0
    prev[1:] = target[:-1]
    change_pos = np.flatnonzero(target != prev)

    # No change at all (e.g. a constant or never-positive signal) → never deployed,
    # so the portfolio is the lump sum held flat in cash for the whole window.
    if change_pos.size == 0:
        flat = pd.Series(np.full(n_days, float(initial_investment)), index=price_df.index)
        return flat, initial_investment

    # Walk only the change-days, rebalancing to the new target with the shared
    # primitive, and snapshot the resulting holdings (per asset) and cash.
    holdings: dict[str, float] = {col: 0.0 for col in columns}
    cash = float(initial_investment)
    snap_holdings = np.zeros((change_pos.size, n_assets))
    snap_cash = np.empty(change_pos.size)
    for j, i in enumerate(change_pos):
        holdings, cash, _ = _rebalance_to_target(
            holdings, cash, price_df.iloc[i], float(target[i]), weights
        )
        snap_holdings[j] = [holdings[col] for col in columns]
        snap_cash[j] = cash

    # Each day takes the holdings/cash of the most recent change at or before it;
    # days before the first change are still all cash (no holdings yet).
    seg = np.searchsorted(change_pos, np.arange(n_days), side='right') - 1
    has_position = seg >= 0
    seg_clip = np.clip(seg, 0, None)
    holdings_mat = np.where(has_position[:, None], snap_holdings[seg_clip], 0.0)
    cash_arr = np.where(has_position, snap_cash[seg_clip], float(initial_investment))

    # Value every day: Σ units × price over assets with a valid price (NaN priced
    # → 0 but units retained), plus cash — mirroring _portfolio_value exactly.
    daily_value = (holdings_mat * np.where(np.isnan(prices_mat), 0.0, prices_mat)).sum(axis=1) + cash_arr

    return pd.Series(daily_value, index=price_df.index), initial_investment


def simulate_rotation(
    price_df: pd.DataFrame,
    events: list[tuple[pd.Timestamp, float, dict[str, float] | None]],
    initial_investment: float = INITIAL_INVESTMENT,
) -> tuple[pd.Series, float]:
    """Simulate a lump sum rebalanced to explicit (date, fraction, weights) events.

    Generalises ``simulate_riskoff`` (whose ``weights`` is one dict closed over the
    whole run — only the aggregate invested *fraction* varies day to day) to
    strategies whose asset **selection** also changes over time: each event
    carries its own target invested fraction *and* its own per-asset weights, so a
    single rebalance can change both how much is invested and which assets it goes
    into — e.g. an annual "rotate into new positions" strategy. Trades happen only
    on the given event dates; the position is simply held (and drifts with the
    market) in between, exactly like ``simulate_riskoff``.

    Parameters
    ----------
    price_df           – daily close prices (one column per asset).
    events              – chronological ``(date, target_fraction, weights)`` triples.
                         *date* should be a row of ``price_df.index``; events whose date
                         is absent are skipped. *target_fraction* is 0..1. *weights* is
                         the symbol → relative weight ``_rebalance_to_target`` applies
                         for that rebalance (None/empty = equal weight over whatever is
                         priced that day).
    initial_investment – one-off lump sum provided as cash at the start.

    Returns
    -------
    (portfolio_series, total_invested)
      portfolio_series – portfolio value (holdings + cash) on every trading day.
      total_invested   – the initial lump sum (constant; no contributions).

    Implemented as a hybrid like ``simulate_riskoff``: the loop only walks the
    (few) event dates, reusing ``_rebalance_to_target``'s math to snapshot the
    post-trade holdings/cash, then the dense daily valuation is one vectorised
    numpy pass.
    """
    if price_df.empty:
        return pd.Series(dtype=float), initial_investment

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # Column order shared by the holdings dict (string keys) and the price matrix.
    columns = [str(col) for col in price_df.columns]
    prices_mat = price_df.to_numpy(dtype=float)
    n_days, n_assets = prices_mat.shape

    # Chronological order, and drop any event whose date isn't actually a row of
    # price_df (e.g. a rotation date outside the windowed range).
    ordered = sorted(events, key=lambda e: e[0])
    pos_arr = price_df.index.get_indexer([e[0] for e in ordered])
    valid = pos_arr >= 0
    ordered = [e for e, v in zip(ordered, valid) if v]
    event_pos = pos_arr[valid]

    # No usable event → never deployed, so the lump sum is held flat in cash.
    if not ordered:
        flat = pd.Series(np.full(n_days, float(initial_investment)), index=price_df.index)
        return flat, initial_investment

    # Walk only the event dates, rebalancing to each one's own target fraction AND
    # weights with the shared primitive, snapshotting the resulting holdings/cash.
    holdings: dict[str, float] = {col: 0.0 for col in columns}
    cash = float(initial_investment)
    snap_holdings = np.zeros((len(ordered), n_assets))
    snap_cash = np.empty(len(ordered))
    for j, (_, frac, weights) in enumerate(ordered):
        i = event_pos[j]
        holdings, cash, _ = _rebalance_to_target(
            holdings, cash, price_df.iloc[i], float(frac), weights
        )
        snap_holdings[j] = [holdings[col] for col in columns]
        snap_cash[j] = cash

    # Each day takes the holdings/cash of the most recent event at or before it;
    # days before the first event are still all cash (no holdings yet).
    seg = np.searchsorted(event_pos, np.arange(n_days), side='right') - 1
    has_position = seg >= 0
    seg_clip = np.clip(seg, 0, None)
    holdings_mat = np.where(has_position[:, None], snap_holdings[seg_clip], 0.0)
    cash_arr = np.where(has_position, snap_cash[seg_clip], float(initial_investment))

    # Value every day: Σ units × price over assets with a valid price (NaN priced
    # → 0 but units retained), plus cash — mirroring _portfolio_value exactly.
    daily_value = (holdings_mat * np.where(np.isnan(prices_mat), 0.0, prices_mat)).sum(axis=1) + cash_arr

    return pd.Series(daily_value, index=price_df.index), initial_investment


def _rotation_order_events(
    price_df: pd.DataFrame,
    events: list[tuple[pd.Timestamp, float, dict[str, float] | None]],
    initial_investment: float = INITIAL_INVESTMENT,
    fx: FxColumns | None = None,
) -> list[OrderEvent]:
    """Record one OrderEvent per rotation event (see ``simulate_rotation``).

    The events-list counterpart of ``_riskoff_order_events``: instead of deriving
    change-days from a per-day target-fraction Series and one static weights dict,
    it walks the (few) explicit ``(date, target_fraction, weights)`` triples
    directly — each one both a possible aggregate-fraction change *and* a possible
    re-selection of which assets are held — rebalancing at each one via
    ``_rebalance_to_target``. The side is 'Buy' when the fraction rose (more
    invested) or 'Sell' when it fell (more cash); there is no fresh money, so
    inflow is always 0.

    Parameters
    ----------
    price_df           – windowed daily closes, one column per asset.
    events              – chronological ``(date, target_fraction, weights)`` triples;
                         events whose date is absent from ``price_df.index`` are
                         skipped (mirrors ``simulate_rotation``).
    initial_investment – one-off lump sum held as cash at the start.
    fx                  – optional FX context (see backtest.FxColumns); when given,
                         each event also carries the trading-currency quote and the
                         per-pair FX rates so the order table can show conversions.

    Returns
    -------
    Chronological list of OrderEvents (empty when price_df or events is empty).
    """
    if price_df.empty or not events:
        return []

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # FX side-tables (empty when no conversion applies) used to add the
    # trading-currency price and per-pair rate columns to each order event.
    asset_rate = fx.asset_rate if fx else {}
    pair_rate = fx.pair_rate if fx else {}

    holdings: dict[str, float] = {str(col): 0.0 for col in price_df.columns}
    cash = initial_investment
    # Fraction currently allocated; starts at 0.0 (all cash) so the first event's
    # (presumably positive) fraction is always recorded as a Buy.
    current = 0.0

    ordered = sorted((e for e in events if e[0] in price_df.index), key=lambda e: e[0])

    out: list[OrderEvent] = []
    for date, frac, weights in ordered:
        prices = price_df.loc[date]
        value_before = _portfolio_value(holdings, cash, prices)
        side = 'Buy' if frac > current else 'Sell'

        holdings, cash, _ = _rebalance_to_target(holdings, cash, prices, frac, weights)
        current = frac

        out.append(OrderEvent(
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

    return out


def simulate_lumpsum(
    price_df: pd.DataFrame,
    initial_investment: float = INITIAL_INVESTMENT,
    weights: dict[str, float] | None = None,
) -> tuple[pd.Series, float]:
    """Simulate a single initial lump-sum investment held to the end (buy & hold).

    The whole *initial_investment* is deployed once, on the first trading day on
    which **every** (positively-weighted) basket asset is buyable, split across
    them **in proportion to their per-asset weights** (equal weight by default), so
    the initial investment follows the chosen allocation rather than
    over-weighting whichever assets listed first.  Those units are then held
    unchanged for the rest of the window —
    there are no further trades, no rebalancing and no contributions — so the
    daily value curve is simply the fixed holdings re-priced each day.  Because
    no fresh money ever enters after day one, *total_invested* is the lump sum.

    Mirrors ``simulate_riskoff``'s valuation exactly: the one-off purchase reuses
    ``_rebalance_to_target`` (target fraction 1.0 = fully invested), then every
    day is valued in a single vectorised numpy pass (Σ units × price over assets
    with a valid price — a NaN price contributes 0 but keeps its units — plus any
    residual cash).  Days before the first all-priced buy day hold the lump sum
    flat as cash.

    Parameters
    ----------
    price_df           – daily close prices (one column per asset).
    initial_investment – one-off lump sum invested on the first buyable day.
    weights            – optional symbol → relative weight; None/empty means equal
                         weight.  The lump sum is split across the basket in
                         proportion to these weights.

    Returns
    -------
    (portfolio_series, total_invested)
      portfolio_series – portfolio value (holdings + cash) on every trading day.
      total_invested   – the initial lump sum (constant; no contributions).
    """
    # Guard the empty case before touching the (then non-datetime) index.
    if price_df.empty:
        return pd.Series(dtype=float), initial_investment

    assert isinstance(price_df.index, pd.DatetimeIndex)

    # Column order shared by the holdings dict (string keys) and the price matrix.
    columns = [str(col) for col in price_df.columns]
    prices_mat = price_df.to_numpy(dtype=float)
    n_days, n_assets = prices_mat.shape

    # The lump sum is deployed on the first day on which *every* asset that will be
    # bought (positive weight) is buyable, so it follows the chosen allocation
    # rather than over-weighting whichever assets listed first.
    weight_vec = _weight_array(columns, weights)
    buy_pos = _first_all_priced_pos(prices_mat, weight_vec)

    # No day has the whole basket priced → the lump sum is held flat in cash.
    if buy_pos is None:
        flat = pd.Series(np.full(n_days, float(initial_investment)), index=price_df.index)
        return flat, initial_investment

    # Deploy 100% on that first all-priced day, weighted across the basket,
    # reusing the exact Risk-Off rebalance math (cash_after ≈ 0).
    holdings: dict[str, float] = {col: 0.0 for col in columns}
    holdings, cash, _ = _rebalance_to_target(
        holdings, float(initial_investment), price_df.iloc[buy_pos], 1.0, weights
    )
    holdings_vec = np.array([holdings[col] for col in columns])

    # The bought units are held from the buy day onward (0 before it); the cash
    # is the lump sum before the buy and the (tiny) residual after it.
    invested = np.arange(n_days) >= buy_pos
    holdings_mat = np.where(invested[:, None], holdings_vec[None, :], 0.0)
    cash_arr = np.where(invested, cash, float(initial_investment))

    # Value every day exactly like _portfolio_value: Σ units × price over assets
    # with a valid price (NaN priced → 0 but units retained), plus cash.
    daily_value = (holdings_mat * np.where(np.isnan(prices_mat), 0.0, prices_mat)).sum(axis=1) + cash_arr

    return pd.Series(daily_value, index=price_df.index), initial_investment


def _get_monthly_range(base_url: str, filename: str, df_meta: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the earliest and latest month-end dates for a single asset.

    Loads only the 'Close' column to minimise data transfer, then resamples
    to monthly because the date-range slider is month-granular.

    Parameters
    ----------
    base_url – root URL/path for the parquet files.
    filename – parquet filename for this asset.
    df_meta  – master metadata table.

    Returns
    -------
    (start, end) as pandas Timestamps, or (None, None) on any failure.
    """
    # Validate that the filename exists in the catalogue before loading.
    meta = df_meta[df_meta['filename'] == filename]
    if meta.empty:
        return None, None
    try:
        # The slider refreshes on every basket change; routing through the
        # process-wide cache (_read_close_series) means each asset's file is
        # fetched once across all those refreshes AND the later backtest run,
        # instead of once per refresh. The cached read already returns only the
        # UTC-normalised Close column, so we resample directly off it.
        t_read = time.perf_counter()
        close = _read_close_series(base_url, filename)
        log.debug('[perf] _get_monthly_range read %s: %.1fms', filename,
                  (time.perf_counter() - t_read) * 1000)
        # Resample to month-end so the reported range matches the month-granular
        # slider the user selects the backtest window with.
        monthly = close.resample('ME').last().dropna()
        if monthly.empty:
            return None, None
        return monthly.index.min(), monthly.index.max()
    except Exception:
        log.exception("Failed to get date range for %s", filename)
        return None, None


def get_common_date_range(
    base_url: str | None, filenames_a: list[str], filenames_b: list[str], df_meta: pd.DataFrame
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Find the monthly date range common to every asset in both baskets.

    The common (overlapping) range is [max(all_starts), min(all_ends)].
    If the two extremes do not overlap (max_start >= min_end), returns
    (None, None) to indicate no usable shared history.

    Parameters
    ----------
    base_url    – root URL/path for the parquet files.
    filenames_a – parquet filenames for basket A (may be empty).
    filenames_b – parquet filenames for basket B (may be empty).
    df_meta     – master metadata table.

    Returns
    -------
    (start, end) as UTC month-end Timestamps, or (None, None) if there is no
    overlap or no assets were provided.
    """
    # Combine both baskets' filenames into one list. 'or []' guards against
    # None (e.g. when a basket store has never been written).
    all_filenames = list(filenames_a or []) + list(filenames_b or [])
    if not all_filenames or not base_url:
        return None, None

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for filename in all_filenames:
        s, e = _get_monthly_range(base_url, filename, df_meta)
        if s is not None and e is not None:
            starts.append(s)
            ends.append(e)

    if not starts:
        return None, None

    # Intersection logic:
    #   common_start = the LATEST of all individual start dates (all assets
    #                  must be active → take the most-recently-launched one).
    #   common_end   = the EARLIEST of all individual end dates (all assets
    #                  must still be active → take the one that ended first).
    common_start = max(starts)
    common_end = min(ends)

    # If the latest start is not before the earliest end, there is no overlap.
    if common_start >= common_end:
        return None, None

    return common_start, common_end


def run_backtest(
    base_url: str | None,
    filenames: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    df_meta: pd.DataFrame,
    strategy: "BacktestStrategy | None" = None,
    strategy_params: dict[str, int | float | str] | None = None,
    base_currency: str = BASE_CURRENCY,
    weights: dict[str, float] | None = None,
) -> tuple[pd.Series | None, dict[str, str] | None, list[OrderRow] | None]:
    """Orchestrate a backtest for a single basket of assets.

    When *strategy* is provided the call is delegated entirely to that plugin,
    which is responsible for loading data, filtering dates, and computing
    metrics.  When *strategy* is None the built-in Buy & Hold logic below is
    used, preserving full backward compatibility for existing callers.

    Steps (strategy=None / built-in Buy & Hold path):
      1. Load daily close prices for every asset in the basket.
      2. Restrict to the calendar months of the [start_date, end_date] window.
      3. Forward-fill small price gaps.
      4. Run the lump-sum simulation (invest once on day one, hold to the end).
      5. Compute and return performance metrics.

    Parameters
    ----------
    base_url        – root URL/path for the parquet data files.
    filenames       – parquet filenames for every asset in the basket.
    start_date      – first month-end date to include (inclusive).
    end_date        – last month-end date to include (inclusive).
    df_meta         – master metadata table (symbol, name, filename …).
    strategy        – optional strategy plugin instance; when supplied the
                      remaining parameters are forwarded to strategy.run().
    strategy_params – config values for the strategy plugin (may be empty or
                      None; the plugin falls back to its declared defaults).
    base_currency   – reporting currency every asset is converted into before
                      simulation (forwarded to the plugin / load_daily_closes).
    weights         – optional symbol → relative weight controlling how capital is
                      split across the basket (None/empty = equal weight, the
                      historical behaviour); forwarded to the plugin / simulation.

    Returns
    -------
    (portfolio_series, metrics_dict, order_log) on success, or
    (None, None, None) on failure.  The built-in Buy & Hold path (strategy=None)
    is a legacy/back-compat entry point not used by the UI, so it returns None
    for the order log; the per-strategy order logs are produced by the plugins.
    """
    # Delegate to a strategy plugin when one is supplied.
    if strategy is not None:
        if not filenames or not base_url:
            return None, None, None
        return strategy.run(
            base_url, filenames, start_date, end_date, df_meta,
            strategy_params or {}, base_currency=base_currency, weights=weights,
        )

    # Bail out immediately if the caller provided nothing useful.
    if not filenames or not base_url:
        return None, None, None

    # Load all assets' daily close prices into a single aligned DataFrame.
    price_df = load_daily_closes(base_url, filenames, df_meta, base_currency)
    if price_df.empty:
        return None, None, None

    # Keep only the calendar months of the requested window (both month bounds
    # inclusive); empty asset columns within the window are dropped.
    price_df = _window_by_month(price_df, start_date, end_date)
    if price_df.empty:
        return None, None, None

    # Forward-fill fills a NaN price by carrying the previous valid price
    # forward. This handles short gaps such as weekends, exchange holidays or
    # delayed data without distorting the simulation. limit=5 (≈ one trading
    # week) keeps newly listed or suspended assets from being treated as priced
    # during a long absence.
    price_df = price_df.ffill(limit=5)

    portfolio, total_invested = simulate_lumpsum(price_df, weights=weights)
    return portfolio, compute_metrics(portfolio, total_invested), None
