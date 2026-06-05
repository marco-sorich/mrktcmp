import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

os.environ.pop("BASE_URL", None)

from src.backtest import (  # noqa: E402
    simulate_dca,
    compute_metrics,
    run_backtest,
    get_common_date_range,
    _get_monthly_range,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_META = pd.DataFrame({
    'asset_class': ['stocks', 'crypto'],
    'symbol': ['AAPL', 'BTC'],
    'name': ['Apple Inc', 'Bitcoin'],
    'filename': ['aapl.parquet', 'btc.parquet'],
})

BASE_URL = "http://example.com"

_MONTHLY_IDX = pd.date_range('2020-01-31', periods=24, freq='ME', tz='UTC')


def _daily_ohlcv(price, n_days=400, tz: str | None = 'UTC'):
    """Build a minimal daily OHLCV DataFrame with a constant close price."""
    idx = pd.date_range('2020-01-01', periods=n_days, freq='D', tz=tz)
    return pd.DataFrame(
        {'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': 1000},
        index=idx,
    )


def _monthly_portfolio(values):
    return pd.Series(values, index=_MONTHLY_IDX[:len(values)])


# ---------------------------------------------------------------------------
# simulate_dca
# ---------------------------------------------------------------------------

class TestSimulateDca:
    def test_empty_dataframe_returns_empty_series_and_zero_invested(self):
        portfolio, total = simulate_dca(pd.DataFrame())
        assert portfolio.empty
        assert total == 0.0

    def test_single_asset_total_invested_equals_monthly_times_periods(self):
        df = pd.DataFrame({'AAPL': [100.0] * 12}, index=_MONTHLY_IDX[:12])
        _, total = simulate_dca(df, monthly_investment=1000.0)
        assert total == pytest.approx(12 * 1000.0)

    def test_single_asset_flat_price_first_month_value(self):
        df = pd.DataFrame({'AAPL': [100.0] * 3}, index=_MONTHLY_IDX[:3])
        portfolio, _ = simulate_dca(df, monthly_investment=1000.0)
        assert portfolio.iloc[0] == pytest.approx(1000.0)

    def test_single_asset_flat_price_value_grows_linearly(self):
        df = pd.DataFrame({'AAPL': [100.0] * 4}, index=_MONTHLY_IDX[:4])
        portfolio, _ = simulate_dca(df, monthly_investment=1000.0)
        # After n months at price 100 with 1000/month: n * 10 shares * 100 = n * 1000
        for i, expected in enumerate([1000.0, 2000.0, 3000.0, 4000.0]):
            assert portfolio.iloc[i] == pytest.approx(expected)

    def test_two_assets_investment_split_equally_single_month(self):
        df = pd.DataFrame(
            {'AAPL': [100.0], 'BTC': [200.0]},
            index=_MONTHLY_IDX[:1],
        )
        portfolio, total = simulate_dca(df, monthly_investment=1000.0)
        # 500 in AAPL (5 shares) + 500 in BTC (2.5 shares) = 500 + 500 = 1000
        assert total == pytest.approx(1000.0)
        assert portfolio.iloc[0] == pytest.approx(1000.0)

    def test_nan_asset_excluded_from_monthly_buy(self):
        df = pd.DataFrame(
            {'AAPL': [100.0, 100.0], 'BTC': [np.nan, 200.0]},
            index=_MONTHLY_IDX[:2],
        )
        portfolio, total = simulate_dca(df, monthly_investment=1000.0)
        # Month 1: only AAPL → invest full 1000 in AAPL
        # Month 2: both → invest 500 each; AAPL holdings = 10 + 5 = 15 shares
        assert total == pytest.approx(2000.0)
        assert portfolio.iloc[0] == pytest.approx(1000.0)

    def test_output_index_matches_input_index(self):
        idx = _MONTHLY_IDX[:6]
        df = pd.DataFrame({'AAPL': [100.0] * 6}, index=idx)
        portfolio, _ = simulate_dca(df)
        assert list(portfolio.index) == list(idx)

    def test_rising_prices_produce_profit(self):
        prices = [100.0 * (1.05 ** i) for i in range(12)]
        df = pd.DataFrame({'AAPL': prices}, index=_MONTHLY_IDX[:12])
        portfolio, total = simulate_dca(df, monthly_investment=1000.0)
        assert portfolio.iloc[-1] > total

    def test_daily_data_invests_monthly_values_daily(self):
        # 3 calendar months of *daily* constant-price data.
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        df = pd.DataFrame({'AAPL': 100.0}, index=idx)
        portfolio, total = simulate_dca(df, monthly_investment=1000.0)
        # One value per trading day (daily valuation).
        assert len(portfolio) == len(idx)
        # Exactly one contribution per calendar month (Jan, Feb, Mar) → 3 × 1000.
        assert total == pytest.approx(3000.0)
        # Before the first month-end nothing is invested yet → value 0 on day 1.
        assert portfolio.iloc[0] == pytest.approx(0.0)
        # On the final day (31 Mar, a month-end) all three contributions are in:
        # 30 shares × 100 = 3000.
        assert portfolio.iloc[-1] == pytest.approx(3000.0)
        # The value steps up only on the three month-end contribution days.
        increases = int((portfolio.diff().fillna(0.0) > 0).sum())
        assert increases == 3


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    _EXPECTED_KEYS = {
        'Total Return', 'CAGR', 'Sharpe Ratio', 'Max. Drawdown',
        'Volatility (p.a.)', 'Calmar Ratio', 'Invested', 'End Value',
        'Profit/Loss',
    }

    def test_empty_series_returns_empty_dict(self):
        assert compute_metrics(pd.Series([], dtype=float), 1000.0) == {}

    def test_series_shorter_than_three_returns_empty_dict(self):
        assert compute_metrics(pd.Series([1000.0, 2000.0]), 1000.0) == {}

    def test_zero_invested_returns_empty_dict(self):
        assert compute_metrics(_monthly_portfolio([1000.0] * 12), 0.0) == {}

    def test_returns_exactly_the_expected_keys(self):
        metrics = compute_metrics(_monthly_portfolio([float(i * 1000) for i in range(1, 25)]), 12000.0)
        assert set(metrics.keys()) == self._EXPECTED_KEYS

    def test_all_values_are_strings(self):
        metrics = compute_metrics(_monthly_portfolio([float(i * 1000) for i in range(1, 25)]), 12000.0)
        assert all(isinstance(v, str) for v in metrics.values())

    def test_profit_shows_plus_sign_in_gesamtertrag(self):
        # Final value >> invested
        portfolio = _monthly_portfolio([1000.0 * (i + 10) for i in range(12)])
        metrics = compute_metrics(portfolio, 6000.0)
        assert metrics['Total Return'].startswith('+')

    def test_loss_shows_minus_sign_in_gesamtertrag(self):
        portfolio = _monthly_portfolio([max(500.0 - i * 40, 1.0) for i in range(12)])
        metrics = compute_metrics(portfolio, 6000.0)
        assert metrics['Total Return'].startswith('-')

    def test_monotonically_rising_portfolio_has_zero_drawdown(self):
        portfolio = _monthly_portfolio([float(i + 1) * 1000 for i in range(24)])
        metrics = compute_metrics(portfolio, 12000.0)
        assert metrics['Max. Drawdown'] == '0.0%'

    def test_investiert_is_parseable_as_number(self):
        portfolio = _monthly_portfolio([float(i + 1) * 1000 for i in range(12)])
        metrics = compute_metrics(portfolio, 6000.0)
        parsed = float(metrics['Invested'].replace(',', ''))
        assert parsed == pytest.approx(6000.0)


# ---------------------------------------------------------------------------
# run_backtest (integration)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    # Shared date bounds used across tests – cover 3 years of monthly data.
    _START = pd.Timestamp('2021-01-31', tz='UTC')
    _END = pd.Timestamp('2023-12-31', tz='UTC')

    def test_empty_filenames_returns_none_none(self):
        p, m = run_backtest(BASE_URL, [], self._START, self._END, SAMPLE_META)
        assert p is None and m is None

    def test_no_base_url_returns_none_none(self):
        p, m = run_backtest(None, ['aapl.parquet'], self._START, self._END, SAMPLE_META)
        assert p is None and m is None

    def test_successful_run_returns_portfolio_and_metrics_dict(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=2000)):
            p, m = run_backtest(BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META)
        assert p is not None
        assert isinstance(m, dict)
        assert 'CAGR' in m

    def test_date_window_filters_portfolio_length(self):
        # 5 years of daily data; request only 2 years. The curve is now daily,
        # so ~2 years of trading days are returned (one value per day).
        now = pd.Timestamp.now(tz='UTC')
        idx = pd.date_range(now - pd.DateOffset(years=5), now, freq='D')
        ohlcv = pd.DataFrame(
            {'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1},
            index=idx,
        )
        start = now - pd.DateOffset(years=2)
        with patch('src.backtest.pd.read_parquet', return_value=ohlcv):
            p, _ = run_backtest(BASE_URL, ['aapl.parquet'], start, now, SAMPLE_META)
        assert p is not None
        # ~2 years of daily rows (month-windowed): clearly daily, well over the
        # ~24 a monthly curve would have, and below ~3 years' worth.
        assert 700 <= len(p) <= 800

    def test_start_after_end_returns_none_none(self):
        # Invalid range: start > end → empty DataFrame → (None, None).
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=2000)):
            p, m = run_backtest(
                BASE_URL, ['aapl.parquet'],
                self._END, self._START,   # intentionally reversed
                SAMPLE_META,
            )
        assert p is None and m is None

    def test_both_bounds_respected(self):
        # Data spans 5 years; request the middle year → only ~12 months.
        now = pd.Timestamp.now(tz='UTC')
        idx = pd.date_range(now - pd.DateOffset(years=5), now, freq='D')
        ohlcv = pd.DataFrame(
            {'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1},
            index=idx,
        )
        start = now - pd.DateOffset(years=3)
        end = now - pd.DateOffset(years=2)
        with patch('src.backtest.pd.read_parquet', return_value=ohlcv):
            p, _ = run_backtest(BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META)
        assert p is not None
        # Middle year spans ~13 calendar months of daily rows (month-windowed).
        assert 360 <= len(p) <= 420


# ---------------------------------------------------------------------------
# _get_monthly_range
# ---------------------------------------------------------------------------

class TestGetMonthlyRange:
    def test_unknown_filename_returns_none_none(self):
        s, e = _get_monthly_range(BASE_URL, 'unknown.parquet', SAMPLE_META)
        assert s is None and e is None

    def test_returns_start_and_end_timestamps(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=400)):
            s, e = _get_monthly_range(BASE_URL, 'aapl.parquet', SAMPLE_META)
        assert s is not None
        assert e is not None
        assert s <= e

    def test_parquet_error_returns_none_none(self):
        with patch('src.backtest.pd.read_parquet', side_effect=OSError('fail')):
            s, e = _get_monthly_range(BASE_URL, 'aapl.parquet', SAMPLE_META)
        assert s is None and e is None


# ---------------------------------------------------------------------------
# get_common_date_range
# ---------------------------------------------------------------------------

class TestGetCommonDateRange:
    def test_both_empty_returns_none_none(self):
        s, e = get_common_date_range(BASE_URL, [], [], SAMPLE_META)
        assert s is None and e is None

    def test_no_base_url_returns_none_none(self):
        s, e = get_common_date_range(None, ['aapl.parquet'], [], SAMPLE_META)
        assert s is None and e is None

    def test_single_basket_returns_that_assets_range(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=730)):
            s, e = get_common_date_range(BASE_URL, ['aapl.parquet'], [], SAMPLE_META)
        assert s is not None and e is not None
        assert s <= e

    def test_non_overlapping_ranges_return_none_none(self):
        # Asset A: Jan 2020 – Dec 2020 (only).
        # Asset B: Jan 2022 – Dec 2022 (only).
        # They have no shared months → (None, None).
        idx_a = pd.date_range('2020-01-01', '2020-12-31', freq='D', tz='UTC')
        idx_b = pd.date_range('2022-01-01', '2022-12-31', freq='D', tz='UTC')
        ohlcv_a = pd.DataFrame({'Close': 1.0}, index=idx_a)
        ohlcv_b = pd.DataFrame({'Close': 1.0}, index=idx_b)

        def _mock_read(path, columns=None):
            return ohlcv_a if 'aapl' in path else ohlcv_b

        with patch('src.backtest.pd.read_parquet', side_effect=_mock_read):
            s, e = get_common_date_range(
                BASE_URL, ['aapl.parquet'], ['btc.parquet'], SAMPLE_META
            )
        assert s is None and e is None

    def test_overlapping_ranges_return_intersection(self):
        # Asset A starts Jan 2020, Asset B starts Jul 2020.
        # Both end Dec 2021. Intersection = Jul 2020 – Dec 2021.
        idx_a = pd.date_range('2020-01-01', '2021-12-31', freq='D', tz='UTC')
        idx_b = pd.date_range('2020-07-01', '2021-12-31', freq='D', tz='UTC')
        ohlcv_a = pd.DataFrame({'Close': 1.0}, index=idx_a)
        ohlcv_b = pd.DataFrame({'Close': 1.0}, index=idx_b)

        def _mock_read(path, columns=None):
            return ohlcv_a if 'aapl' in path else ohlcv_b

        with patch('src.backtest.pd.read_parquet', side_effect=_mock_read):
            s, e = get_common_date_range(
                BASE_URL, ['aapl.parquet'], ['btc.parquet'], SAMPLE_META
            )
        assert s is not None
        # Common start must be >= Jul 2020
        assert s >= pd.Timestamp('2020-07-01', tz='UTC')
