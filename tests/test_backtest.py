import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

os.environ.pop("BASE_URL", None)

from src.backtest import (  # noqa: E402
    load_monthly_closes,
    simulate_dca,
    compute_metrics,
    run_backtest,
    get_common_date_range,
    _get_monthly_range,
    _enrich_events,
    BacktestRun,
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
# load_monthly_closes
# ---------------------------------------------------------------------------

class TestLoadMonthlyCloses:
    def test_empty_filenames_returns_empty_dataframe(self):
        result = load_monthly_closes(BASE_URL, [], SAMPLE_META)
        assert result.empty

    def test_unknown_filename_skipped_returns_empty(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            result = load_monthly_closes(BASE_URL, ['unknown.parquet'], SAMPLE_META)
        assert result.empty

    def test_single_asset_column_named_by_symbol(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            result = load_monthly_closes(BASE_URL, ['aapl.parquet'], SAMPLE_META)
        assert 'AAPL' in result.columns

    def test_daily_data_resampled_to_monthly(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=365)):
            result = load_monthly_closes(BASE_URL, ['aapl.parquet'], SAMPLE_META)
        assert len(result) <= 13

    def test_timezone_naive_index_gets_utc_localization(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, tz=None)):
            result = load_monthly_closes(BASE_URL, ['aapl.parquet'], SAMPLE_META)
        assert isinstance(result.index, pd.DatetimeIndex) and result.index.tz is not None

    def test_timezone_aware_index_preserved(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, tz='America/New_York')):
            result = load_monthly_closes(BASE_URL, ['aapl.parquet'], SAMPLE_META)
        assert isinstance(result.index, pd.DatetimeIndex) and result.index.tz is not None

    def test_parquet_error_is_swallowed_returns_empty(self):
        with patch('src.backtest.pd.read_parquet', side_effect=OSError("not found")):
            result = load_monthly_closes(BASE_URL, ['aapl.parquet'], SAMPLE_META)
        assert result.empty

    def test_multiple_assets_combined_into_one_dataframe(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            result = load_monthly_closes(BASE_URL, ['aapl.parquet', 'btc.parquet'], SAMPLE_META)
        assert 'AAPL' in result.columns
        assert 'BTC' in result.columns

    def test_monthly_close_is_last_price_of_month(self):
        idx = pd.date_range('2022-01-01', '2022-01-31', freq='D', tz='UTC')
        prices = [100.0] * 30 + [999.0]
        ohlcv = pd.DataFrame(
            {'Open': prices, 'High': prices, 'Low': prices, 'Close': prices, 'Volume': 1},
            index=idx,
        )
        with patch('src.backtest.pd.read_parquet', return_value=ohlcv):
            result = load_monthly_closes(BASE_URL, ['aapl.parquet'], SAMPLE_META)
        assert result['AAPL'].iloc[0] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# simulate_dca
# ---------------------------------------------------------------------------

class TestSimulateDca:
    def test_empty_dataframe_returns_empty_series_and_zero_invested(self):
        portfolio, total, events = simulate_dca(pd.DataFrame())
        assert portfolio.empty
        assert total == 0.0
        assert events == []

    def test_single_asset_total_invested_equals_monthly_times_periods(self):
        df = pd.DataFrame({'AAPL': [100.0] * 12}, index=_MONTHLY_IDX[:12])
        _, total, _ = simulate_dca(df, monthly_investment=1000.0)
        assert total == pytest.approx(12 * 1000.0)

    def test_single_asset_flat_price_first_month_value(self):
        df = pd.DataFrame({'AAPL': [100.0] * 3}, index=_MONTHLY_IDX[:3])
        portfolio, _, _ = simulate_dca(df, monthly_investment=1000.0)
        assert portfolio.iloc[0] == pytest.approx(1000.0)

    def test_single_asset_flat_price_value_grows_linearly(self):
        df = pd.DataFrame({'AAPL': [100.0] * 4}, index=_MONTHLY_IDX[:4])
        portfolio, _, _ = simulate_dca(df, monthly_investment=1000.0)
        # After n months at price 100 with 1000/month: n * 10 shares * 100 = n * 1000
        for i, expected in enumerate([1000.0, 2000.0, 3000.0, 4000.0]):
            assert portfolio.iloc[i] == pytest.approx(expected)

    def test_two_assets_investment_split_equally_single_month(self):
        df = pd.DataFrame(
            {'AAPL': [100.0], 'BTC': [200.0]},
            index=_MONTHLY_IDX[:1],
        )
        portfolio, total, _ = simulate_dca(df, monthly_investment=1000.0)
        # 500 in AAPL (5 shares) + 500 in BTC (2.5 shares) = 500 + 500 = 1000
        assert total == pytest.approx(1000.0)
        assert portfolio.iloc[0] == pytest.approx(1000.0)

    def test_nan_asset_excluded_from_monthly_buy(self):
        df = pd.DataFrame(
            {'AAPL': [100.0, 100.0], 'BTC': [np.nan, 200.0]},
            index=_MONTHLY_IDX[:2],
        )
        portfolio, total, _ = simulate_dca(df, monthly_investment=1000.0)
        # Month 1: only AAPL → invest full 1000 in AAPL
        # Month 2: both → invest 500 each; AAPL holdings = 10 + 5 = 15 shares
        assert total == pytest.approx(2000.0)
        assert portfolio.iloc[0] == pytest.approx(1000.0)

    def test_output_index_matches_input_index(self):
        idx = _MONTHLY_IDX[:6]
        df = pd.DataFrame({'AAPL': [100.0] * 6}, index=idx)
        portfolio, _, _ = simulate_dca(df)
        assert list(portfolio.index) == list(idx)

    def test_rising_prices_produce_profit(self):
        prices = [100.0 * (1.05 ** i) for i in range(12)]
        df = pd.DataFrame({'AAPL': prices}, index=_MONTHLY_IDX[:12])
        portfolio, total, _ = simulate_dca(df, monthly_investment=1000.0)
        assert portfolio.iloc[-1] > total


# ---------------------------------------------------------------------------
# simulate_dca – event ledger
# ---------------------------------------------------------------------------

class TestSimulateDcaEvents:
    def test_one_event_per_investing_month(self):
        df = pd.DataFrame({'AAPL': [100.0] * 4}, index=_MONTHLY_IDX[:4])
        _, _, events = simulate_dca(df, monthly_investment=1000.0)
        assert len(events) == 4

    def test_event_records_pre_and_post_trade_value(self):
        df = pd.DataFrame({'AAPL': [100.0] * 3}, index=_MONTHLY_IDX[:3])
        _, _, events = simulate_dca(df, monthly_investment=1000.0)
        # First month: nothing held before, 1000 invested after.
        assert events[0]['value_pre_trade'] == pytest.approx(0.0)
        assert events[0]['value_post_trade'] == pytest.approx(1000.0)
        # post - pre always equals the monthly contribution (the external flow).
        for ev in events:
            assert ev['value_post_trade'] - ev['value_pre_trade'] == pytest.approx(1000.0)
            assert ev['external_flow'] == pytest.approx(1000.0)

    def test_legs_are_positive_buys_split_equally(self):
        df = pd.DataFrame({'AAPL': [100.0], 'BTC': [200.0]}, index=_MONTHLY_IDX[:1])
        _, _, events = simulate_dca(df, monthly_investment=1000.0)
        legs = events[0]['legs']
        assert set(legs) == {'AAPL', 'BTC'}
        assert legs['AAPL']['amount'] == pytest.approx(500.0)
        assert legs['AAPL']['shares'] == pytest.approx(5.0)
        assert legs['BTC']['shares'] == pytest.approx(2.5)
        assert all(leg['shares'] > 0 for leg in legs.values())

    def test_month_without_priced_assets_yields_no_event(self):
        df = pd.DataFrame({'AAPL': [np.nan, 100.0]}, index=_MONTHLY_IDX[:2])
        _, _, events = simulate_dca(df, monthly_investment=1000.0)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# _enrich_events
# ---------------------------------------------------------------------------

class TestEnrichEvents:
    def _dca_events(self):
        df = pd.DataFrame({'AAPL': [100.0, 110.0, 121.0]}, index=_MONTHLY_IDX[:3])
        _, _, events = simulate_dca(df, monthly_investment=1000.0)
        return _enrich_events(events)

    def test_none_and_empty_pass_through(self):
        assert _enrich_events(None) is None
        assert _enrich_events([]) == []

    def test_cum_invested_accumulates_external_flow(self):
        events = self._dca_events()
        assert [ev['cum_invested'] for ev in events] == pytest.approx([1000.0, 2000.0, 3000.0])

    def test_pnl_equals_value_minus_cost_basis(self):
        events = self._dca_events()
        for ev in events:
            assert ev['pnl'] == pytest.approx(ev['value_post_trade'] - ev['cum_invested'])

    def test_equity_and_cash_fractions_sum_to_one(self):
        events = self._dca_events()
        for ev in events:
            assert ev['equity_pct'] + ev['cash_pct'] == pytest.approx(1.0)
            assert ev['cash_pct'] == pytest.approx(0.0)  # DCA holds no cash

    def test_first_period_return_is_zero(self):
        events = self._dca_events()
        assert events[0]['period_return_pct'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BacktestRun
# ---------------------------------------------------------------------------

class TestBacktestRun:
    def test_holds_provided_fields(self):
        run = BacktestRun(
            run_id='a', label='Basket A · DCA', color='#1a56db',
            portfolio=None, metrics=None, events=None,
        )
        assert run.run_id == 'a'
        assert run.label == 'Basket A · DCA'
        assert run.color == '#1a56db'


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    _EXPECTED_KEYS = {
        'Total Return', 'CAGR', 'Sharpe Ratio', 'Max. Drawdown',
        'Volatility (p.a.)', 'Calmar Ratio', 'Invested', 'End Value',
        'Profit/Loss', 'Best Month', 'Worst Month',
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
        p, m, ev = run_backtest(BASE_URL, [], self._START, self._END, SAMPLE_META)
        assert p is None and m is None and ev is None

    def test_no_base_url_returns_none_none(self):
        p, m, ev = run_backtest(None, ['aapl.parquet'], self._START, self._END, SAMPLE_META)
        assert p is None and m is None and ev is None

    def test_successful_run_returns_portfolio_and_metrics_dict(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=2000)):
            p, m, ev = run_backtest(BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META)
        assert p is not None
        assert isinstance(m, dict)
        assert 'CAGR' in m
        # The built-in DCA path returns an enriched event ledger.
        assert isinstance(ev, list) and ev
        assert 'cum_invested' in ev[0]

    def test_date_window_filters_portfolio_length(self):
        # 5 years of daily data; request only 2 years → ~24 months returned.
        now = pd.Timestamp.now(tz='UTC')
        idx = pd.date_range(now - pd.DateOffset(years=5), now, freq='D')
        ohlcv = pd.DataFrame(
            {'Open': 100.0, 'High': 100.0, 'Low': 100.0, 'Close': 100.0, 'Volume': 1},
            index=idx,
        )
        start = now - pd.DateOffset(years=2)
        with patch('src.backtest.pd.read_parquet', return_value=ohlcv):
            p, _, _ = run_backtest(BASE_URL, ['aapl.parquet'], start, now, SAMPLE_META)
        assert p is not None
        assert len(p) <= 26  # 2 years ≈ 24 months, allow 2 for edge rounding

    def test_start_after_end_returns_none_none(self):
        # Invalid range: start > end → empty DataFrame → (None, None).
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=2000)):
            p, m, ev = run_backtest(
                BASE_URL, ['aapl.parquet'],
                self._END, self._START,   # intentionally reversed
                SAMPLE_META,
            )
        assert p is None and m is None and ev is None

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
            p, _, _ = run_backtest(BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META)
        assert p is not None
        assert 10 <= len(p) <= 14   # roughly 12 months


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
