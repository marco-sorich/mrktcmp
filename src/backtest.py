import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MONTHLY_INVESTMENT = 1000.0


def load_monthly_closes(base_url, filenames, df_meta):
    """Load and combine monthly close prices for the given asset filenames."""
    series = {}
    for filename in filenames:
        try:
            meta = df_meta[df_meta['filename'] == filename]
            if meta.empty:
                continue
            symbol = meta.iloc[0]['symbol']
            ohlcv = pd.read_parquet(f"{base_url}/{filename}")
            close = ohlcv['Close']
            if close.index.tz is None:
                close.index = close.index.tz_localize('UTC')
            monthly = close.resample('ME').last().dropna()
            if not monthly.empty:
                series[symbol] = monthly
        except Exception:
            log.exception("Failed to load %s", filename)

    if not series:
        return pd.DataFrame()

    return pd.DataFrame(series)


def simulate_dca(price_df, monthly_investment=MONTHLY_INVESTMENT):
    """
    Simulate monthly DCA: invest a fixed amount each month, split equally
    across all assets that have a valid price that month.
    Returns (portfolio_series, total_invested).
    """
    holdings = {col: 0.0 for col in price_df.columns}
    values = []
    total_invested = 0.0

    for _, prices in price_df.iterrows():
        available = {c: p for c, p in prices.items() if pd.notna(p) and p > 0}
        if available:
            per_asset = monthly_investment / len(available)
            total_invested += monthly_investment
            for col, price in available.items():
                holdings[col] += per_asset / price

        value = sum(
            holdings[c] * prices[c]
            for c in price_df.columns
            if pd.notna(prices.get(c, np.nan))
        )
        values.append(value)

    return pd.Series(values, index=price_df.index), total_invested


def compute_metrics(portfolio, total_invested):
    """Compute performance metrics from a DCA portfolio value series."""
    if portfolio.empty or len(portfolio) < 3 or total_invested <= 0:
        return {}

    final_value = portfolio.iloc[-1]
    total_return = (final_value - total_invested) / total_invested
    monthly_returns = portfolio.pct_change().dropna()
    n_years = len(portfolio) / 12

    cagr = (final_value / total_invested) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    vol = monthly_returns.std() * np.sqrt(12)
    sharpe = (
        (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
        if monthly_returns.std() > 0 else 0.0
    )
    rolling_max = portfolio.expanding().max()
    max_dd = ((portfolio - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    return {
        'Total Return': f"{total_return * 100:+.1f}%",
        'CAGR': f"{cagr * 100:.1f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max. Drawdown': f"{max_dd * 100:.1f}%",
        'Volatility (p.a.)': f"{vol * 100:.1f}%",
        'Calmar Ratio': f"{calmar:.2f}",
        'Invested': f"{total_invested:,.0f}",
        'End Value': f"{final_value:,.0f}",
        'Profit/Loss': f"{final_value - total_invested:+,.0f}",
        'Best Month': f"{monthly_returns.max() * 100:+.1f}%",
        'Worst Month': f"{monthly_returns.min() * 100:+.1f}%",
    }


def run_backtest(base_url, filenames, years, df_meta):
    """
    Load price data, restrict to the requested window (capped by the
    shortest available asset history), run DCA simulation.
    Returns (portfolio_series, metrics_dict) or (None, None) on failure.
    """
    if not filenames or not base_url:
        return None, None

    price_df = load_monthly_closes(base_url, filenames, df_meta)
    if price_df.empty:
        return None, None

    tz = price_df.index.tz
    cutoff = pd.Timestamp.now(tz=tz) - pd.DateOffset(years=years)
    price_df = price_df[price_df.index >= cutoff].dropna(how='all', axis=1)

    if price_df.empty:
        return None, None

    price_df = price_df.ffill(limit=3)
    portfolio, total_invested = simulate_dca(price_df)
    return portfolio, compute_metrics(portfolio, total_invested)
