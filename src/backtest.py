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


def load_monthly_closes(base_url, filenames, df_meta):
    """Load and combine monthly close prices for the given asset filenames.

    Parameters
    ----------
    base_url  : str  – root URL/path where the parquet files are hosted.
    filenames : list – list of parquet file names (e.g. ['aapl.parquet']).
    df_meta   : DataFrame – the master metadata table that maps filenames to
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


def simulate_dca(price_df, monthly_investment=MONTHLY_INVESTMENT):
    """Simulate monthly DCA: invest a fixed amount each month, split equally
    across all assets that have a valid price that month.

    Parameters
    ----------
    price_df          : DataFrame – monthly close prices (one column per asset).
    monthly_investment: float     – total EUR invested per month across the basket.

    Returns
    -------
    (portfolio_series, total_invested)
      portfolio_series : Series of portfolio value at each month-end.
      total_invested   : float – cumulative EUR put in (excludes months with no data).
    """
    # holdings maps each asset symbol to the number of units (shares/coins)
    # currently owned. Starts at zero for every asset.
    holdings = {col: 0.0 for col in price_df.columns}

    # Collect the portfolio's total value at the end of each month.
    values = []

    # Running total of money actually deposited. This grows by
    # monthly_investment every month where at least one asset has a price.
    total_invested = 0.0

    # Iterate over every month in chronological order.
    # price_df.iterrows() yields (index_value, Series_of_prices) pairs.
    # The underscore '_' discards the date index since we only need prices here.
    for _, prices in price_df.iterrows():

        # Build a dict of assets that can actually be bought this month.
        # A NaN price means the data feed had a gap; a zero price means the
        # asset was suspended or delisted – both should be skipped.
        available = {c: p for c, p in prices.items() if pd.notna(p) and p > 0}

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
                holdings[col] += per_asset / price

        # Calculate total portfolio value: sum up (units × current price)
        # for every asset that has a price this month. Assets without a
        # current price are excluded from the valuation but their holdings
        # are kept intact; they will contribute again when prices reappear.
        value = sum(
            holdings[c] * prices[c]
            for c in price_df.columns
            if pd.notna(prices.get(c, np.nan))  # prices.get returns NaN if key missing
        )
        values.append(value)

    # Wrap the list into a pandas Series, reusing the DataFrame's date index
    # so the result is properly time-indexed.
    return pd.Series(values, index=price_df.index), total_invested


def compute_metrics(portfolio, total_invested):
    """Compute standard performance metrics from a DCA portfolio value series.

    Parameters
    ----------
    portfolio      : Series – monthly portfolio value over time.
    total_invested : float  – total EUR deposited throughout the period.

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


def run_backtest(base_url, filenames, years, df_meta):
    """Orchestrate a full DCA backtest for a single basket of assets.

    Steps:
      1. Load monthly close prices for every asset in the basket.
      2. Cut the price history to the requested number of years, measured
         backwards from the last available data point (not from today).
      3. Forward-fill small price gaps.
      4. Run the DCA simulation.
      5. Compute and return performance metrics.

    Parameters
    ----------
    base_url  : str      – root URL/path for the parquet data files.
    filenames : list     – parquet filenames for every asset in the basket.
    years     : int      – how many years of history to simulate.
    df_meta   : DataFrame– master metadata table (symbol, name, filename …).

    Returns
    -------
    (portfolio_series, metrics_dict) on success, or (None, None) on failure.
    """
    # Bail out immediately if the caller provided nothing useful.
    if not filenames or not base_url:
        return None, None

    # Load all assets' monthly close prices into a single aligned DataFrame.
    price_df = load_monthly_closes(base_url, filenames, df_meta)
    if price_df.empty:
        return None, None

    # Anchor the cutoff to the last available data point rather than to the
    # current clock time. If the data feed is stale (e.g. updated monthly),
    # using 'now' would silently shorten the simulated window by however long
    # the feed lag is. Using data_end keeps the full requested period intact.
    cutoff = price_df.index.max() - pd.DateOffset(years=years)

    # Keep only rows at or after the cutoff date.
    # dropna(how='all', axis=1) removes any asset column whose entire window
    # is NaN (i.e. the asset did not exist during the requested period at all).
    price_df = price_df[price_df.index >= cutoff].dropna(how='all', axis=1)

    if price_df.empty:
        return None, None

    # Forward-fill fills a NaN price by carrying the previous valid price
    # forward. This handles short gaps such as exchange holidays or delayed
    # data without distorting the simulation.
    # limit=3 means at most 3 consecutive months can be filled; longer gaps
    # remain NaN so newly listed or temporarily suspended assets are not
    # incorrectly treated as having a price during their absence.
    price_df = price_df.ffill(limit=3)

    portfolio, total_invested = simulate_dca(price_df)
    return portfolio, compute_metrics(portfolio, total_invested)
