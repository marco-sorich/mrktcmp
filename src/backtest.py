# ---------------------------------------------------------------------------
# backtest.py – DCA (Dollar-Cost Averaging) simulation engine
#
# Dollar-Cost Averaging means investing a fixed amount of money at regular
# intervals (here: monthly) regardless of the current price. Over time this
# automatically buys more units when prices are low and fewer when prices
# are high, smoothing out the effect of volatility.
# ---------------------------------------------------------------------------

# logging: Python's standard library for recording messages at different
# severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). Using a named
# logger (instead of print) lets callers control output format and level.
import logging

# dataclass: lightweight, typed container used for BacktestRun (see below).
from dataclasses import dataclass

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

# A tiny threshold below which a share delta is treated as "no trade".  Risk-Off
# rebalances every month, so floating-point noise can produce vanishingly small
# deltas; ignoring them keeps the transaction ledger to genuine buys/sells.
_TRADE_EPS = 1e-9


@dataclass
class BacktestRun:
    """One comparable backtest result: a (basket, strategy) combination.

    This is the generic unit the result UI iterates over.  Today the app
    produces two runs (basket A and B), but modelling a *list* of runs keeps
    the chart and transaction tables agnostic of how many comparisons exist or
    how baskets and strategies are paired, so future N-way comparisons (more
    baskets, or one basket compared under several strategies) need no change to
    the rendering code.

    Fields
    ------
    run_id    – stable key used in component IDs and the event store
                (e.g. 'a' / 'b'; any unique string in the future).
    label     – human-readable name shown on the tab, e.g. 'Basket A · DCA'.
    color     – trace/tab colour, assigned by position from a palette.
    portfolio – monthly portfolio value series, or None if the run failed.
    metrics   – pre-formatted performance metrics, or None.
    events    – transaction/event ledger (see simulate_dca/simulate_riskoff),
                or None when the strategy does not produce one.
    """

    run_id: str
    label: str
    color: str
    portfolio: pd.Series | None
    metrics: dict[str, str] | None
    events: list[dict] | None


def _value_at(holdings: dict[str, float], prices: pd.Series, columns: "pd.Index | list[str]") -> float:
    """Total value of *holdings* priced at *prices*, ignoring assets without a price.

    Shared by the simulation engines so the "before" and "after" valuations of
    an event use exactly the same logic.
    """
    return float(sum(
        holdings[c] * prices[c]
        for c in columns
        if pd.notna(prices.get(c, np.nan))  # prices.get returns NaN if key missing
    ))


def _enrich_events(events: list[dict] | None) -> list[dict] | None:
    """Add derived per-event KPIs to a raw event ledger, in place.

    The simulation engines emit only the base event fields (date, pre/post-trade
    value, cash, per-asset legs and the external_flow contributed at the event).
    This single, strategy-agnostic pass derives the running cost basis and the
    performance KPIs shown in the transaction table, so every strategy gets the
    same columns computed the same way.

    Added keys
    ----------
    cum_invested      – running sum of external_flow (cost basis to date).
    pnl, pnl_pct      – profit/loss vs cost basis, absolute and relative.
    equity_pct        – fraction of the post-trade value held in assets.
    cash_pct          – fraction of the post-trade value held in cash.
    period_return_pct – return of the post-trade value vs the previous event.
    """
    if not events:
        return events

    cum_invested = 0.0
    prev_value: float | None = None
    for ev in events:
        # cum_invested accumulates the external capital put in.  For DCA this is
        # the monthly contribution; for Risk-Off the one-off lump sum carried by
        # the first event.  Defining it via an explicit external_flow (rather
        # than post-minus-pre value) is what makes the lump-sum cost basis come
        # out correctly even though Risk-Off rebalancing is value-neutral.
        cum_invested += ev['external_flow']
        value = ev['value_post_trade']
        ev['cum_invested'] = cum_invested
        ev['pnl'] = value - cum_invested
        ev['pnl_pct'] = (ev['pnl'] / cum_invested) if cum_invested else 0.0
        ev['equity_pct'] = ((value - ev['cash']) / value) if value else 0.0
        ev['cash_pct'] = (ev['cash'] / value) if value else 0.0
        ev['period_return_pct'] = (value / prev_value - 1.0) if prev_value else 0.0
        prev_value = value
    return events


def load_monthly_closes(base_url: str, filenames: list[str], df_meta: pd.DataFrame) -> pd.DataFrame:
    """Load and combine monthly close prices for the given asset filenames.

    Parameters
    ----------
    base_url  – root URL/path where the parquet files are hosted.
    filenames – list of parquet file names (e.g. ['aapl.parquet']).
    df_meta   – the master metadata table that maps filenames to
                human-readable symbols and names.

    Returns
    -------
    DataFrame with one column per successfully loaded asset (named by symbol)
    and one row per month-end date. Missing months are NaN.
    """
    # Accumulate individual monthly price series here before combining them.
    series = {}

    for filename in filenames:
        try:
            # Look up the asset's metadata row by its filename.
            # The result is a DataFrame subset, possibly empty if the
            # filename is not in the master table.
            meta = df_meta[df_meta['filename'] == filename]
            if meta.empty:
                # Unknown file – skip it silently (no crash, no empty chart).
                continue

            # .iloc[0] picks the first (and normally only) matching row,
            # returning it as a Series so we can access columns by name.
            symbol = meta.iloc[0]['symbol']

            # Parquet is a columnar binary file format. It is much faster to
            # read and smaller than CSV for tabular numerical data.
            ohlcv = pd.read_parquet(f"{base_url}/{filename}")

            # OHLCV = Open / High / Low / Close / Volume.
            # We only need the Close price for backtesting.
            close = ohlcv['Close']

            # Normalise to UTC so all series share a common timezone for
            # alignment. Without a timezone, pandas cannot safely compare
            # timestamps from different series.
            assert isinstance(close.index, pd.DatetimeIndex)
            if close.index.tz is None:
                close.index = close.index.tz_localize('UTC')

            # resample('ME') groups all daily rows within each calendar month
            # into a single bucket labelled at the month's last day ('ME' =
            # Month End). .last() picks the closing price on the last trading
            # day of that month. .dropna() removes buckets where no data
            # existed (e.g. before the asset was listed).
            monthly = close.resample('ME').last().dropna()

            if not monthly.empty:
                # Use the ticker symbol (e.g. 'AAPL') as the column name so
                # the combined DataFrame is human-readable.
                series[symbol] = monthly

        except Exception:
            # Log the full traceback but continue processing the remaining
            # filenames; one bad file should not break the whole backtest.
            log.exception("Failed to load %s", filename)

    if not series:
        # Return an empty DataFrame (not None) so callers can check .empty
        # instead of testing for None.
        return pd.DataFrame()

    # pd.DataFrame(dict_of_series) performs an outer join on the index:
    # every month that appears in *any* series gets a row; assets that have
    # no price for a given month get NaN in that cell.
    return pd.DataFrame(series)


def simulate_dca(
    price_df: pd.DataFrame, monthly_investment: float = MONTHLY_INVESTMENT
) -> tuple[pd.Series, float, list[dict]]:
    """Simulate monthly DCA: invest a fixed amount each month, split equally
    across all assets that have a valid price that month.

    Parameters
    ----------
    price_df           – monthly close prices (one column per asset).
    monthly_investment – total EUR invested per month across the basket.

    Returns
    -------
    (portfolio_series, total_invested, events)
      portfolio_series – Series of portfolio value at each month-end.
      total_invested   – cumulative EUR put in (excludes months with no data).
      events           – per-event transaction ledger; one entry per month in
                         which a purchase occurred (see module docs for the
                         dict shape).  KPIs are added later by _enrich_events.
    """
    # holdings maps each asset symbol to the number of units (shares/coins)
    # currently owned. Starts at zero for every asset.
    holdings = {col: 0.0 for col in price_df.columns}

    # Collect the portfolio's total value at the end of each month.
    values = []

    # Raw transaction ledger (one entry per investing month).
    events: list[dict] = []

    # Running total of money actually deposited. This grows by
    # monthly_investment every month where at least one asset has a price.
    total_invested = 0.0

    # Iterate over every month in chronological order.
    # price_df.iterrows() yields (date, Series_of_prices) pairs.
    for date, prices in price_df.iterrows():

        # Build a dict of assets that can actually be bought this month.
        # A NaN price means the data feed had a gap; a zero price means the
        # asset was suspended or delisted – both should be skipped.
        available: dict[str, float] = {str(c): float(p) for c, p in prices.items() if pd.notna(p) and p > 0}

        # Value of existing holdings at this month's prices, BEFORE buying.
        value_pre_trade = _value_at(holdings, prices, price_df.columns)

        legs: dict[str, dict] = {}
        if available:
            # Split the monthly investment equally among all available assets.
            # Example: 1,000 € split across 4 assets → 250 € per asset.
            per_asset = monthly_investment / len(available)

            # Record that this month's deposit has been made.
            total_invested += monthly_investment

            for col, price in available.items():
                # Convert euros to units: units = € / price_per_unit.
                # Example: 250 € / 50 €per share = 5 shares.
                # The decimal result is fine; fractional units model a fund
                # or broker that allows partial shares.
                shares = per_asset / price
                holdings[col] += shares
                # DCA only ever buys, so shares/amount are always positive.
                legs[col] = {'shares': shares, 'amount': per_asset, 'price': price}

        # Portfolio value AFTER this month's purchase (same logic as before).
        value = _value_at(holdings, prices, price_df.columns)
        values.append(value)

        # Record an event only for months where a purchase actually happened.
        if legs:
            events.append({
                'date': date,
                'value_pre_trade': value_pre_trade,
                'value_post_trade': value,
                'cash': 0.0,                       # DCA is always fully invested
                'external_flow': monthly_investment,  # the deposit made this month
                'legs': legs,
            })

    # Wrap the list into a pandas Series, reusing the DataFrame's date index
    # so the result is properly time-indexed.
    return pd.Series(values, index=price_df.index), total_invested, events


def compute_metrics(portfolio: pd.Series, total_invested: float) -> dict[str, str]:
    """Compute standard performance metrics from a DCA portfolio value series.

    Parameters
    ----------
    portfolio      – monthly portfolio value over time.
    total_invested – total EUR deposited throughout the period.

    Returns
    -------
    dict of metric name → formatted string, or {} if input is too short.
    """
    # Require at least 3 months of data for meaningful statistics, and a
    # positive investment (avoid division by zero).
    if portfolio.empty or len(portfolio) < 3 or total_invested <= 0:
        return {}

    # The last value in the portfolio series is the current total worth.
    final_value = portfolio.iloc[-1]

    # Total Return: how much money was made relative to what was put in.
    # Example: invested 12,000 €, now worth 15,000 € → (15k-12k)/12k = +25%.
    total_return = (final_value - total_invested) / total_invested

    # pct_change() computes month-over-month percentage returns:
    # return[i] = (value[i] - value[i-1]) / value[i-1]
    # The very first row has no predecessor so it becomes NaN; dropna() removes it.
    monthly_returns = portfolio.pct_change().dropna()

    # Convert the number of monthly data points to years.
    # This is used as the exponent in the CAGR formula below.
    n_years = len(portfolio) / 12

    # CAGR = Compound Annual Growth Rate.
    # It answers: "at what constant yearly rate would the investment have
    # needed to grow to reach this end value from the total invested?"
    # Formula: (end_value / invested) ^ (1 / years) - 1
    cagr = (final_value / total_invested) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Annualised Volatility: the standard deviation of monthly returns scaled
    # up to a yearly figure. Multiplying by √12 converts monthly std to annual
    # std, assuming returns are independent month-to-month.
    vol = monthly_returns.std() * np.sqrt(12)

    # Sharpe Ratio: return per unit of risk (volatility), annualised.
    # Higher is better. A ratio > 1 is generally considered good.
    # We assume a risk-free rate of 0% here (simplified).
    # The guard against zero std prevents division by zero on a flat portfolio.
    sharpe = (
        (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
        if monthly_returns.std() > 0 else 0.0
    )

    # Maximum Drawdown: the largest peak-to-trough decline ever experienced.
    # rolling_max tracks the highest portfolio value seen so far at each point.
    # The ratio (value - peak) / peak gives the percentage decline from peak.
    # .min() finds the worst such decline. The result is negative by convention
    # (e.g. -0.30 = the portfolio fell 30% from its all-time high at worst).
    rolling_max = portfolio.expanding().max()
    max_dd = ((portfolio - rolling_max) / rolling_max).min()

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
        'Best Month': f"{monthly_returns.max() * 100:+.1f}%",
        'Worst Month': f"{monthly_returns.min() * 100:+.1f}%",
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

    Mirrors load_monthly_closes but keeps the full daily resolution and the
    full available history (no date-window restriction).  The extra history is
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
            # the timezone handling in load_monthly_closes / _get_monthly_range.
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
    """Rebalance the whole portfolio to a target invested fraction for one month.

    The target invested value is *inv_frac* × total portfolio value, spread
    equally across all assets that have a valid price this month; the remainder
    stays in cash.  Extracted into its own helper to keep simulate_riskoff
    below the flake8 complexity threshold.

    Parameters
    ----------
    holdings – units currently held per asset symbol.
    cash     – current uninvested cash.
    prices   – this month's close price per asset (may contain NaN/zero).
    inv_frac – target fraction of the portfolio to be invested (0.0..1.0).

    Returns
    -------
    (new_holdings, new_cash, total_value) where total_value is the portfolio's
    worth this month (cash plus the value of all priced holdings).
    """
    # Assets that can actually be traded this month (valid, positive price).
    priced = {str(c): float(p) for c, p in prices.items() if pd.notna(p) and p > 0}

    # Total worth = cash + value of currently priced holdings.
    invested_value = sum(holdings[c] * priced[c] for c in priced)
    total_value = cash + invested_value

    if not priced:
        # Nothing tradable this month: carry holdings and cash unchanged.
        return holdings, cash, total_value

    # Buy the target invested amount, split equally across priced assets.
    target_invested = inv_frac * total_value
    per_asset = target_invested / len(priced)

    new_holdings = dict(holdings)
    for c in priced:
        new_holdings[c] = per_asset / priced[c]

    new_cash = total_value - target_invested
    return new_holdings, new_cash, total_value


def _rebalance_legs(
    old_holdings: dict[str, float], new_holdings: dict[str, float], prices: pd.Series
) -> dict[str, dict]:
    """Build the per-asset transaction legs of one rebalancing step.

    A leg is recorded for every priced asset whose unit count changed; the
    share delta is signed (positive = buy, negative = sell).  Tiny deltas from
    floating-point noise are ignored via _TRADE_EPS.
    """
    legs: dict[str, dict] = {}
    for c, p in prices.items():
        if pd.isna(p) or p <= 0:
            continue
        col = str(c)
        delta = new_holdings[col] - old_holdings[col]
        if abs(delta) > _TRADE_EPS:
            price = float(p)
            legs[col] = {'shares': delta, 'amount': delta * price, 'price': price}
    return legs


def simulate_riskoff(
    price_df: pd.DataFrame,
    invested_fraction_at_month: pd.Series,
    initial_investment: float = INITIAL_INVESTMENT,
) -> tuple[pd.Series, float, list[dict]]:
    """Simulate the lump-sum Risk-Off strategy month by month.

    The full *initial_investment* starts as cash.  Each month the portfolio is
    rebalanced to the invested fraction supplied for that month (selling when
    the fraction drops, buying when it rises); cash earns 0 %.  There are no
    additional contributions, so *total_invested* equals the initial lump sum.

    Parameters
    ----------
    price_df                    – monthly close prices (one column per asset).
    invested_fraction_at_month  – Series of target invested fractions (0..1);
                                  reindexed onto price_df's index, with missing
                                  months defaulting to 0.0 (fully in cash).
    initial_investment          – one-off lump sum provided as cash at the start.

    Returns
    -------
    (portfolio_series, total_invested, events)
      portfolio_series – Series of portfolio value (holdings + cash) per month.
      total_invested   – the initial lump sum (constant; no contributions).
      events           – per-event rebalancing ledger; one entry per month in
                         which at least one asset was traded.  Rebalancing is
                         value-neutral, so value_pre_trade == value_post_trade.
    """
    holdings: dict[str, float] = {str(col): 0.0 for col in price_df.columns}
    cash = initial_investment

    # Align the per-month fractions to the price index; unknown months stay in
    # cash (0.0) as a conservative fallback.
    inv = invested_fraction_at_month.reindex(price_df.index).fillna(0.0)

    values: list[float] = []
    events: list[dict] = []
    # The lump sum is external capital injected once; it is attached to the
    # first recorded event so _enrich_events derives the cost basis correctly.
    flow_pending = initial_investment

    # Event cadence is monthly by design: this loop steps once per month, so at
    # most one event is ever recorded per month – never several.  Because the
    # portfolio is rebalanced back to equal weight every month, prices drifting
    # apart mean some asset almost always needs a trade, so an event is produced
    # in essentially every *invested* month.  Event-free months therefore occur
    # only during sustained fully-in-cash stretches (target fraction 0, holdings
    # already 0 → no trade → no leg → no event).  This is intentional: the table
    # mirrors the strategy's monthly rebalancing, it does not trade only on
    # signal changes.
    for i, (date, prices) in enumerate(price_df.iterrows()):
        inv_frac = float(inv.iloc[i])
        old_holdings = holdings
        holdings, cash, value = _rebalance_to_target(old_holdings, cash, prices, inv_frac)
        values.append(value)

        legs = _rebalance_legs(old_holdings, holdings, prices)
        if legs:
            events.append({
                'date': date,
                'value_pre_trade': value,   # rebalancing is value-neutral, so
                'value_post_trade': value,  # before == after at current prices
                'cash': cash,
                'external_flow': flow_pending,  # lump sum on first event, else 0
                'legs': legs,
            })
            flow_pending = 0.0

    return pd.Series(values, index=price_df.index), initial_investment, events


def _get_monthly_range(base_url: str, filename: str, df_meta: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the earliest and latest month-end dates for a single asset.

    Loads only the 'Close' column to minimise data transfer, then resamples
    to monthly to match the cadence used by load_monthly_closes.

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
        # Resample to month-end, same logic as load_monthly_closes, so that
        # the range we report matches the rows the simulation will actually use.
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
) -> tuple[pd.Series | None, dict[str, str] | None, list[dict] | None]:
    """Orchestrate a backtest for a single basket of assets.

    When *strategy* is provided the call is delegated entirely to that plugin,
    which is responsible for loading data, filtering dates, and computing
    metrics.  When *strategy* is None the built-in DCA logic below is used,
    preserving full backward compatibility for existing callers.

    Steps (strategy=None / built-in DCA path):
      1. Load monthly close prices for every asset in the basket.
      2. Restrict to the caller-specified [start_date, end_date] window.
      3. Forward-fill small price gaps.
      4. Run the DCA simulation.
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
    (portfolio_series, metrics_dict, events) on success, or (None, None, None)
    on failure.  *events* is the per-event transaction ledger (with derived
    KPIs added by _enrich_events), or None when the strategy produces no ledger.
    """
    # Delegate to a strategy plugin when one is supplied.
    if strategy is not None:
        if not filenames or not base_url:
            return None, None, None
        portfolio, metrics, events = strategy.run(
            base_url, filenames, start_date, end_date, df_meta,
            strategy_params or {},
        )
        # Derive the per-event KPIs centrally so every strategy's ledger gets
        # the same columns computed identically.
        return portfolio, metrics, _enrich_events(events)

    # Bail out immediately if the caller provided nothing useful.
    if not filenames or not base_url:
        return None, None, None

    # Load all assets' monthly close prices into a single aligned DataFrame.
    price_df = load_monthly_closes(base_url, filenames, df_meta)
    if price_df.empty:
        return None, None, None

    # Keep only rows within the requested date window [start_date, end_date].
    # Both bounds are inclusive (>=, <=) so the user's chosen start/end months
    # are always included in the simulation.
    # dropna(how='all', axis=1) removes any asset column that is entirely NaN
    # within the window (the asset did not exist during this period at all).
    mask = (price_df.index >= start_date) & (price_df.index <= end_date)
    price_df = price_df[mask].dropna(how='all', axis=1)

    if price_df.empty:
        return None, None, None

    # Forward-fill fills a NaN price by carrying the previous valid price
    # forward. This handles short gaps such as exchange holidays or delayed
    # data without distorting the simulation.
    # limit=3 means at most 3 consecutive months can be filled; longer gaps
    # remain NaN so newly listed or temporarily suspended assets are not
    # incorrectly treated as having a price during their absence.
    price_df = price_df.ffill(limit=3)

    portfolio, total_invested, events = simulate_dca(price_df)
    return portfolio, compute_metrics(portfolio, total_invested), _enrich_events(events)
