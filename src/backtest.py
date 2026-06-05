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

# TYPE_CHECKING guard avoids a circular import at runtime: strategies/dca.py
# imports from backtest.py, so importing BacktestStrategy here unconditionally
# would create a cycle.  Under TYPE_CHECKING the import is only evaluated by
# static analysis tools (mypy), not at runtime.
from typing import TYPE_CHECKING
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


def simulate_dca(price_df: pd.DataFrame, monthly_investment: float = MONTHLY_INVESTMENT) -> tuple[pd.Series, float]:
    """Simulate monthly DCA on a *daily* price series.

    A fixed amount is contributed once per calendar month — on that month's last
    trading day — split equally across all assets with a valid price that day.
    The portfolio is then valued on *every* trading day, producing a dense daily
    value curve even though money only goes in monthly.

    For a monthly-cadence price_df every row is its month's only entry, so a
    contribution is made on every row and the result is identical to the
    original monthly engine (this keeps the monthly-input unit tests valid).

    Parameters
    ----------
    price_df           – daily (or monthly) close prices, one column per asset.
    monthly_investment – total EUR invested per month across the basket.

    Returns
    -------
    (portfolio_series, total_invested)
      portfolio_series – portfolio value on every row of price_df.
      total_invested   – cumulative EUR contributed (months with no data skipped).
    """
    # Guard the empty case first so the month-end logic below never runs on a
    # non-datetime index (e.g. the RangeIndex of an empty DataFrame).
    if price_df.empty:
        return pd.Series(dtype=float), 0.0

    # holdings maps each asset symbol to the number of units currently owned.
    holdings = {str(col): 0.0 for col in price_df.columns}

    # Portfolio value collected for every trading day, plus the running total
    # of money actually deposited (grows once per month with data).
    values: list[float] = []
    total_invested = 0.0

    # True on each month's last trading day → the single day we contribute.
    assert isinstance(price_df.index, pd.DatetimeIndex)
    contribute_day = _is_month_end_trading_day(price_df.index)

    for is_contribution, (_, prices) in zip(contribute_day, price_df.iterrows()):

        if is_contribution:
            # Assets that can actually be bought on this month-end day. A NaN
            # price means a data gap; a zero price means suspended/delisted –
            # both are skipped.
            available: dict[str, float] = {
                str(c): float(p) for c, p in prices.items() if pd.notna(p) and p > 0
            }
            if available:
                # Split the monthly investment equally among available assets
                # (e.g. 1,000 € across 4 assets → 250 € each) and record the deposit.
                per_asset = monthly_investment / len(available)
                total_invested += monthly_investment
                for col, price in available.items():
                    # units = € / price_per_unit; fractional units are fine.
                    holdings[col] += per_asset / price

        # Value the portfolio on every trading day (cash component is always 0
        # for DCA – all contributed money is immediately invested).
        values.append(_portfolio_value(holdings, 0.0, prices))

    # Wrap into a Series reusing the DataFrame's date index.
    return pd.Series(values, index=price_df.index), total_invested


def compute_metrics(portfolio: pd.Series, total_invested: float) -> dict[str, str]:
    """Compute standard performance metrics from a *daily* portfolio value series.

    Parameters
    ----------
    portfolio      – daily portfolio value over time (DatetimeIndex).
    total_invested – total EUR deposited throughout the period.

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


def load_daily_closes(base_url: str, filenames: list[str], df_meta: pd.DataFrame) -> pd.DataFrame:
    """Load and combine *daily* close prices for the given asset filenames.

    Keeps the full daily resolution and the full available history (no
    date-window restriction); callers window it by month via _window_by_month.
    The extra history is
    required so signals with long look-back windows (e.g. the 200-day moving
    average and the year-to-date anchor) have enough warm-up data before the
    backtest window begins.

    Parameters
    ----------
    base_url  – root URL/path where the parquet files are hosted.
    filenames – list of parquet file names (e.g. ['aapl.parquet']).
    df_meta   – master metadata table mapping filenames to symbols/names.

    Returns
    -------
    DataFrame with one column per successfully loaded asset (named by symbol)
    and one row per trading day. Missing days are NaN. Empty if nothing loaded.
    """
    # Accumulate individual daily price series here before combining them.
    series = {}

    for filename in filenames:
        try:
            # Look up the asset's metadata row by its filename; skip unknowns.
            meta = df_meta[df_meta['filename'] == filename]
            if meta.empty:
                continue

            symbol = meta.iloc[0]['symbol']

            ohlcv = pd.read_parquet(f"{base_url}/{filename}")
            close = ohlcv['Close']

            # Normalise to UTC so all series share a common timezone, matching
            # the timezone handling in _get_monthly_range.
            assert isinstance(close.index, pd.DatetimeIndex)
            if close.index.tz is None:
                close.index = close.index.tz_localize('UTC')
            else:
                close.index = close.index.tz_convert('UTC')

            # Drop rows with no close (gaps before listing). Keep daily cadence.
            close = close.dropna()
            if not close.empty:
                series[symbol] = close

        except Exception:
            # Log the full traceback but keep processing remaining filenames.
            log.exception("Failed to load %s", filename)

    if not series:
        return pd.DataFrame()

    # Outer-join on the date index and sort chronologically so rolling windows
    # and as-of look-ups operate on a monotonically increasing index.
    return pd.DataFrame(series).sort_index()


def build_equal_weight_index(daily_df: pd.DataFrame) -> pd.Series:
    """Combine per-asset daily closes into one equal-weight price index.

    Each day's basket return is the simple mean of the individual assets'
    daily returns (averaging only over assets that have a price that day, so
    mixed start dates and single-asset baskets both work).  Compounding those
    returns yields a single index series (rebased to 100) on which the three
    market signals are evaluated.

    Parameters
    ----------
    daily_df – DataFrame of daily closes, one column per asset.

    Returns
    -------
    Series indexed by trading day with the equal-weight index value (base 100),
    or an empty Series when *daily_df* is empty.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)

    # Treat non-positive prices as missing so they never enter a return.
    prices = daily_df.where(daily_df > 0)

    # Per-asset day-over-day returns, then the cross-sectional mean per day.
    # mean(axis=1) skips NaN, so the average spans only the assets present.
    basket_return = prices.pct_change().mean(axis=1)

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
    holdings: dict[str, float], cash: float, prices: pd.Series, inv_frac: float
) -> tuple[dict[str, float], float, float]:
    """Buy/sell the whole portfolio to a target invested fraction on one day.

    The target invested value is *inv_frac* × total portfolio value, spread
    equally across all assets that have a valid price this day; the remainder
    stays in cash.  This is a one-off adjustment: simulate_riskoff only calls it
    on the day a signal changes, then holds the resulting position (no daily
    maintenance), so the actual fraction drifts with the market in between.

    Parameters
    ----------
    holdings – units currently held per asset symbol.
    cash     – current uninvested cash.
    prices   – this day's close price per asset (may contain NaN/zero).
    inv_frac – target fraction of the portfolio to be invested (0.0..1.0).

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

    # Buy the target invested amount, split equally across priced assets.
    target_invested = inv_frac * total_value
    per_asset = target_invested / len(priced)

    new_holdings = dict(holdings)
    for c in priced:
        new_holdings[c] = per_asset / priced[c]

    new_cash = total_value - target_invested
    return new_holdings, new_cash, total_value


def simulate_riskoff(
    price_df: pd.DataFrame,
    target_fraction: pd.Series,
    initial_investment: float = INITIAL_INVESTMENT,
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

    Returns
    -------
    (portfolio_series, total_invested)
      portfolio_series – portfolio value (holdings + cash) on every trading day.
      total_invested   – the initial lump sum (constant; no contributions).
    """
    if price_df.empty:
        return pd.Series(dtype=float), initial_investment

    holdings: dict[str, float] = {str(col): 0.0 for col in price_df.columns}
    cash = initial_investment

    # Align the daily target fractions to the price index; unknown days stay in
    # cash (0.0) as a conservative fallback.
    target = target_fraction.reindex(price_df.index).fillna(0.0)

    # Fraction we are currently allocated to. Starts at 0.0 (all cash), so the
    # first day with a non-zero target triggers the initial deployment.
    current = 0.0

    values: list[float] = []
    for i, (_, prices) in enumerate(price_df.iterrows()):
        frac = float(target.iloc[i])
        if frac != current:
            # Signal changed → trade to the new target and remember it.
            holdings, cash, _ = _rebalance_to_target(holdings, cash, prices, frac)
            current = frac
        # Value every day, whether or not we traded.
        values.append(_portfolio_value(holdings, cash, prices))

    return pd.Series(values, index=price_df.index), initial_investment


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
        # columns=['Close'] tells pyarrow to read only the Close column from
        # the Parquet file, skipping Open/High/Low/Volume. This is much faster
        # than loading the full OHLCV dataset when we only need date bounds.
        ohlcv = pd.read_parquet(f"{base_url}/{filename}", columns=['Close'])
        close = ohlcv['Close']
        assert isinstance(close.index, pd.DatetimeIndex)
        if close.index.tz is None:
            close.index = close.index.tz_localize('UTC')
        else:
            close.index = close.index.tz_convert('UTC')
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
) -> tuple[pd.Series | None, dict[str, str] | None]:
    """Orchestrate a backtest for a single basket of assets.

    When *strategy* is provided the call is delegated entirely to that plugin,
    which is responsible for loading data, filtering dates, and computing
    metrics.  When *strategy* is None the built-in DCA logic below is used,
    preserving full backward compatibility for existing callers.

    Steps (strategy=None / built-in DCA path):
      1. Load daily close prices for every asset in the basket.
      2. Restrict to the calendar months of the [start_date, end_date] window.
      3. Forward-fill small price gaps.
      4. Run the DCA simulation (monthly contributions, daily valuation).
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

    Returns
    -------
    (portfolio_series, metrics_dict) on success, or (None, None) on failure.
    """
    # Delegate to a strategy plugin when one is supplied.
    if strategy is not None:
        if not filenames or not base_url:
            return None, None
        return strategy.run(
            base_url, filenames, start_date, end_date, df_meta,
            strategy_params or {},
        )

    # Bail out immediately if the caller provided nothing useful.
    if not filenames or not base_url:
        return None, None

    # Load all assets' daily close prices into a single aligned DataFrame.
    price_df = load_daily_closes(base_url, filenames, df_meta)
    if price_df.empty:
        return None, None

    # Keep only the calendar months of the requested window (both month bounds
    # inclusive); empty asset columns within the window are dropped.
    price_df = _window_by_month(price_df, start_date, end_date)
    if price_df.empty:
        return None, None

    # Forward-fill fills a NaN price by carrying the previous valid price
    # forward. This handles short gaps such as weekends, exchange holidays or
    # delayed data without distorting the simulation. limit=5 (≈ one trading
    # week) keeps newly listed or suspended assets from being treated as priced
    # during a long absence.
    price_df = price_df.ffill(limit=5)

    portfolio, total_invested = simulate_dca(price_df)
    return portfolio, compute_metrics(portfolio, total_invested)
