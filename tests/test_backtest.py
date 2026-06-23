import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock

os.environ.pop("BASE_URL", None)

from src.backtest import (  # noqa: E402
    INITIAL_INVESTMENT,
    OrderEvent,
    build_order_log,
    simulate_dca,
    simulate_lumpsum,
    compute_metrics,
    run_backtest,
    load_daily_closes,
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

    def test_nan_price_gap_retains_units_and_revalues_when_price_returns(self):
        # BTC is priced, then has a NaN gap, then is priced again. Its units must
        # be retained through the gap (contributing 0 while NaN) and revalued once
        # the price returns — the key edge case of the vectorised cumsum engine.
        df = pd.DataFrame(
            {'AAPL': [100.0, 100.0, 100.0], 'BTC': [200.0, np.nan, 200.0]},
            index=_MONTHLY_IDX[:3],
        )
        portfolio, total = simulate_dca(df, monthly_investment=1000.0)
        # Month 1: 500 AAPL (5 sh) + 500 BTC (2.5 sh) → 1,000.
        assert portfolio.iloc[0] == pytest.approx(1000.0)
        # Month 2: BTC NaN → full 1,000 into AAPL (now 15 sh). BTC's 2.5 units are
        # retained but not valued (NaN) → 15 × 100 = 1,500.
        assert portfolio.iloc[1] == pytest.approx(1500.0)
        # Month 3: BTC priced again → its retained units revalue; this month adds
        # 5 AAPL (→20) + 2.5 BTC (→5): 20 × 100 + 5 × 200 = 3,000.
        assert portfolio.iloc[2] == pytest.approx(3000.0)
        assert total == pytest.approx(3000.0)


# ---------------------------------------------------------------------------
# simulate_lumpsum
# ---------------------------------------------------------------------------

class TestSimulateLumpsum:
    def test_empty_dataframe_returns_empty_series_and_default_invested(self):
        portfolio, total = simulate_lumpsum(pd.DataFrame())
        assert portfolio.empty
        assert total == pytest.approx(INITIAL_INVESTMENT)

    def test_total_invested_equals_initial_investment(self):
        df = pd.DataFrame({'AAPL': [100.0] * 12}, index=_MONTHLY_IDX[:12])
        _, total = simulate_lumpsum(df, initial_investment=5000.0)
        assert total == pytest.approx(5000.0)

    def test_flat_price_stays_flat_at_initial_investment(self):
        # Buy once on day one and hold: a constant price means the value never
        # moves off the lump sum on any day.
        df = pd.DataFrame({'AAPL': [100.0] * 6}, index=_MONTHLY_IDX[:6])
        portfolio, _ = simulate_lumpsum(df, initial_investment=10_000.0)
        assert portfolio.tolist() == pytest.approx([10_000.0] * 6)

    def test_rising_price_grows_proportionally(self):
        # Price doubles from day one to the end → so does the held position.
        df = pd.DataFrame({'AAPL': [100.0, 150.0, 200.0]}, index=_MONTHLY_IDX[:3])
        portfolio, _ = simulate_lumpsum(df, initial_investment=10_000.0)
        assert portfolio.iloc[0] == pytest.approx(10_000.0)
        assert portfolio.iloc[-1] == pytest.approx(20_000.0)

    def test_two_assets_split_equally_on_day_one(self):
        # 5,000 into AAPL (50 sh) + 5,000 into BTC (25 sh) = 10,000 on day one.
        df = pd.DataFrame(
            {'AAPL': [100.0, 100.0], 'BTC': [200.0, 200.0]},
            index=_MONTHLY_IDX[:2],
        )
        portfolio, _ = simulate_lumpsum(df, initial_investment=10_000.0)
        assert portfolio.iloc[0] == pytest.approx(10_000.0)

    def test_holdings_held_constant_no_rebalance(self):
        # AAPL doubles while BTC halves: a true buy-and-hold keeps the day-one
        # units, so the end value reflects the drifted (un-rebalanced) position.
        df = pd.DataFrame(
            {'AAPL': [100.0, 200.0], 'BTC': [100.0, 50.0]},
            index=_MONTHLY_IDX[:2],
        )
        portfolio, _ = simulate_lumpsum(df, initial_investment=10_000.0)
        # 50 AAPL sh × 200 + 50 BTC sh × 50 = 10,000 + 2,500 = 12,500.
        assert portfolio.iloc[-1] == pytest.approx(12_500.0)

    def test_lump_sum_held_as_cash_before_first_buyable_day(self):
        # No buyable price on day one → the lump sum waits in cash until BTC
        # becomes priced, then is fully deployed.
        df = pd.DataFrame(
            {'AAPL': [np.nan, np.nan], 'BTC': [np.nan, 200.0]},
            index=_MONTHLY_IDX[:2],
        )
        portfolio, _ = simulate_lumpsum(df, initial_investment=10_000.0)
        assert portfolio.iloc[0] == pytest.approx(10_000.0)   # cash
        assert portfolio.iloc[1] == pytest.approx(10_000.0)   # just invested

    def test_output_index_matches_input_index(self):
        idx = _MONTHLY_IDX[:6]
        df = pd.DataFrame({'AAPL': [100.0] * 6}, index=idx)
        portfolio, _ = simulate_lumpsum(df)
        assert list(portfolio.index) == list(idx)


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
        p, m, o = run_backtest(BASE_URL, [], self._START, self._END, SAMPLE_META)
        assert p is None and m is None and o is None

    def test_no_base_url_returns_none_none(self):
        p, m, o = run_backtest(None, ['aapl.parquet'], self._START, self._END, SAMPLE_META)
        assert p is None and m is None and o is None

    def test_successful_run_returns_portfolio_and_metrics_dict(self):
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=2000)):
            p, m, o = run_backtest(BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META)
        assert p is not None
        assert isinstance(m, dict)
        assert 'CAGR' in m
        # The built-in DCA path (strategy=None) is a back-compat entry point not
        # used by the UI, so it produces no order log.
        assert o is None

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
            p, _, _ = run_backtest(BASE_URL, ['aapl.parquet'], start, now, SAMPLE_META)
        assert p is not None
        # ~2 years of daily rows (month-windowed): clearly daily, well over the
        # ~24 a monthly curve would have, and below ~3 years' worth.
        assert 700 <= len(p) <= 800

    def test_start_after_end_returns_none_none(self):
        # Invalid range: start > end → empty DataFrame → (None, None).
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0, n_days=2000)):
            p, m, o = run_backtest(
                BASE_URL, ['aapl.parquet'],
                self._END, self._START,   # intentionally reversed
                SAMPLE_META,
            )
        assert p is None and m is None and o is None

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
        # Middle year spans ~13 calendar months of daily rows (month-windowed).
        assert 360 <= len(p) <= 420


# ---------------------------------------------------------------------------
# build_order_log (generic, strategy-agnostic builder)
# ---------------------------------------------------------------------------

class TestBuildOrderLog:
    """Feeds synthetic OrderEvents to the generic builder and checks the derived
    columns, which are computed identically for every strategy."""

    def _events(self):
        # Two fully-invested buys (each with a fresh inflow), then a pure
        # rebalance to all-cash (no inflow). value_before on row 1 is 1,200 so a
        # non-trivial period return (drift) can be asserted.
        ts = pd.Timestamp
        # asset_values split each assets_after across two assets so it can be
        # asserted to carry through and to sum back to assets_after; asset_prices
        # carries each asset's quote on the trade day (independent of holdings).
        return [
            OrderEvent(date=ts('2020-01-31', tz='UTC'), side='Buy',
                       value_before=0.0, inflow=1000.0, assets_after=1000.0, cash_after=0.0,
                       asset_values={'AAA': 600.0, 'BBB': 400.0},
                       asset_prices={'AAA': 10.0, 'BBB': 20.0}),
            OrderEvent(date=ts('2020-02-29', tz='UTC'), side='Buy',
                       value_before=1200.0, inflow=1000.0, assets_after=2200.0, cash_after=0.0,
                       asset_values={'AAA': 1300.0, 'BBB': 900.0},
                       asset_prices={'AAA': 11.0, 'BBB': 22.0}),
            OrderEvent(date=ts('2020-03-31', tz='UTC'), side='Sell',
                       value_before=2200.0, inflow=0.0, assets_after=0.0, cash_after=2200.0,
                       asset_values={'AAA': 0.0, 'BBB': 0.0},
                       asset_prices={'AAA': 12.0, 'BBB': 24.0}),
        ]

    def test_empty_events_return_empty_list(self):
        assert build_order_log([], initial_capital=0.0) == []

    def test_value_after_is_assets_plus_cash(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        for r in rows:
            assert r['value_after'] == pytest.approx(r['assets_after'] + r['cash_after'])

    def test_asset_values_carried_through_and_sum_to_assets_after(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        # The per-asset breakdown is passed through verbatim …
        assert rows[0]['asset_values'] == {'AAA': 600.0, 'BBB': 400.0}
        # … and its values sum back to assets_after on every row.
        for r in rows:
            assert sum(r['asset_values'].values()) == pytest.approx(r['assets_after'])

    def test_asset_prices_carried_through(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        # The per-asset quotes are passed through verbatim, independent of value
        # (e.g. the final Sell row holds nothing yet still reports both prices).
        assert rows[0]['asset_prices'] == {'AAA': 10.0, 'BBB': 20.0}
        assert rows[2]['asset_prices'] == {'AAA': 12.0, 'BBB': 24.0}

    def test_asset_values_and_prices_default_empty_for_events_without_breakdown(self):
        # Events that predate the per-asset breakdown still build valid rows.
        ev = [OrderEvent(date=pd.Timestamp('2020-01-31', tz='UTC'), side='Buy',
                         value_before=0.0, inflow=1000.0, assets_after=1000.0,
                         cash_after=0.0)]  # type: ignore[typeddict-item]
        rows = build_order_log(ev, initial_capital=0.0)
        assert rows[0]['asset_values'] == {}
        assert rows[0]['asset_prices'] == {}

    def test_net_deposits_accumulates_inflows_from_initial_capital(self):
        rows = build_order_log(self._events(), initial_capital=500.0)
        # seed 500 + inflows 1000, 1000, 0
        assert [r['net_deposits'] for r in rows] == pytest.approx([1500.0, 2500.0, 2500.0])

    def test_pnl_abs_is_value_after_minus_net_deposits(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        # row0: 1000-1000=0; row1: 2200-2000=200; row2: 2200-2000=200
        assert [r['pnl_abs'] for r in rows] == pytest.approx([0.0, 200.0, 200.0])

    def test_pnl_pct_is_pnl_over_net_deposits(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        assert rows[1]['pnl_pct'] == pytest.approx(200.0 / 2000.0)

    def test_equity_exposure_and_cash_quote_sum_to_one(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        for r in rows:
            assert r['equity_exposure'] + r['cash_quote'] == pytest.approx(1.0)
        # Fully invested on the buys, fully in cash after the final sell.
        assert rows[0]['equity_exposure'] == pytest.approx(1.0)
        assert rows[2]['cash_quote'] == pytest.approx(1.0)

    def test_period_return_none_on_first_row_then_reflects_drift(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        assert rows[0]['period_return'] is None
        # row1: value_before 1,200 / prev value_after 1,000 − 1 = +0.20
        assert rows[1]['period_return'] == pytest.approx(0.2)

    def test_zero_denominators_guard_ratio_columns_to_none(self):
        # All-zero seed/inflow/value → every ratio's denominator is 0 → None.
        ev = [OrderEvent(date=pd.Timestamp('2020-01-31', tz='UTC'), side='Buy',
                         value_before=0.0, inflow=0.0, assets_after=0.0, cash_after=0.0,
                         asset_values={}, asset_prices={})]
        rows = build_order_log(ev, initial_capital=0.0)
        assert rows[0]['pnl_pct'] is None
        assert rows[0]['equity_exposure'] is None
        assert rows[0]['cash_quote'] is None

    def test_bh_value_none_when_no_bh_index(self):
        rows = build_order_log(self._events(), initial_capital=0.0)
        for r in rows:
            assert r['bh_value'] is None

    def test_bh_value_computed_from_bh_index(self):
        # bh_index is 1.0 on row-0, 1.5 on row-1, 0.8 on row-2.
        ts = pd.Timestamp
        dates = [ts('2020-01-31', tz='UTC'), ts('2020-02-29', tz='UTC'), ts('2020-03-31', tz='UTC')]
        bh_index = pd.Series([1.0, 1.5, 0.8], index=dates)
        rows = build_order_log(self._events(), initial_capital=0.0, bh_index=bh_index)
        # net_deposits after row-0: 1000; row-1: 2000; row-2: 2000
        assert rows[0]['bh_value'] == pytest.approx(1000.0 * 1.0)
        assert rows[1]['bh_value'] == pytest.approx(2000.0 * 1.5)
        assert rows[2]['bh_value'] == pytest.approx(2000.0 * 0.8)

    def test_bh_value_none_when_date_missing_from_index(self):
        # bh_index only covers row-0; rows 1 and 2 fall back to None.
        ts = pd.Timestamp
        bh_index = pd.Series([1.0], index=[ts('2020-01-31', tz='UTC')])
        rows = build_order_log(self._events(), initial_capital=0.0, bh_index=bh_index)
        assert rows[0]['bh_value'] == pytest.approx(1000.0)
        assert rows[1]['bh_value'] is None
        assert rows[2]['bh_value'] is None


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


# ---------------------------------------------------------------------------
# load_daily_closes – FX (trading-currency) conversion
# ---------------------------------------------------------------------------

# Catalogue with currencies + the FX-pair rows needed to convert into EUR/USD.
# AAPL/VWRL are quoted in USD, SAP in EUR, IDX carries the '0' placeholder
# (unknown currency → left unconverted). USDEUR=X is itself addable as an asset.
FX_META = pd.DataFrame({
    'asset_class': ['Stocks', 'Stocks', 'ETFs', 'Indices', 'currency', 'currency'],
    'symbol':      ['AAPL', 'SAP', 'VWRL', 'IDX', 'USDEUR=X', 'EURUSD=X'],
    'name':        ['Apple', 'SAP SE', 'Vanguard All-World', 'Some Index',
                    'USD/EUR', 'EUR/USD'],
    'filename':    ['aapl.parquet', 'sap.parquet', 'vwrl.parquet', 'idx.parquet',
                    'USDEUR_X.parquet', 'EURUSD_X.parquet'],
    'currency':    ['USD', 'EUR', 'USD', '0', 'EUR', 'USD'],
})

# Constant FX rates used by the mock loader.
_USDEUR = 0.90   # 1 USD = 0.90 EUR
_EURUSD = 1.20   # 1 EUR = 1.20 USD


def _fx_ohlcv(rate, n_days=400, start='2020-01-01'):
    """Daily OHLCV with a constant FX close (same shape as an asset file)."""
    idx = pd.date_range(start, periods=n_days, freq='D', tz='UTC')
    return pd.DataFrame({'Close': rate}, index=idx)


def _fx_mock_read(asset_price=100.0):
    """Build a read_parquet side-effect mapping each filename to its OHLCV."""
    def _read(path, columns=None):
        if 'USDEUR' in path:
            return _fx_ohlcv(_USDEUR)
        if 'EURUSD' in path:
            return _fx_ohlcv(_EURUSD)
        # Any non-FX (asset) file: constant close.
        return _daily_ohlcv(asset_price)
    return _read


class TestLoadDailyClosesFx:
    def test_usd_asset_converted_to_eur(self):
        # AAPL (USD, close 100) × USDEUR (0.90) → 90 EUR on every day.
        with patch('src.backtest.pd.read_parquet', side_effect=_fx_mock_read()):
            df = load_daily_closes(BASE_URL, ['aapl.parquet'], FX_META, 'EUR')
        assert not df.empty
        assert np.allclose(df['AAPL'].dropna(), 100.0 * _USDEUR)

    def test_base_currency_asset_unchanged(self):
        # SAP already in EUR → no conversion, raw close preserved.
        with patch('src.backtest.pd.read_parquet', side_effect=_fx_mock_read()):
            df = load_daily_closes(BASE_URL, ['sap.parquet'], FX_META, 'EUR')
        assert np.allclose(df['SAP'].dropna(), 100.0)

    def test_placeholder_currency_left_unconverted(self):
        # IDX carries the '0' placeholder → treated as unknown, no FX applied.
        with patch('src.backtest.pd.read_parquet', side_effect=_fx_mock_read()):
            df = load_daily_closes(BASE_URL, ['idx.parquet'], FX_META, 'EUR')
        assert np.allclose(df['IDX'].dropna(), 100.0)

    def test_missing_currency_column_is_backward_compatible(self):
        # SAMPLE_META has no 'currency' column → conversion is skipped entirely.
        with patch('src.backtest.pd.read_parquet', side_effect=_fx_mock_read()):
            df = load_daily_closes(BASE_URL, ['aapl.parquet'], SAMPLE_META, 'EUR')
        assert np.allclose(df['AAPL'].dropna(), 100.0)

    def test_selectable_base_currency(self):
        # Same USD asset: unchanged when base=USD, converted when base=EUR.
        with patch('src.backtest.pd.read_parquet', side_effect=_fx_mock_read()):
            usd = load_daily_closes(BASE_URL, ['aapl.parquet'], FX_META, 'USD')
            eur = load_daily_closes(BASE_URL, ['aapl.parquet'], FX_META, 'EUR')
        assert np.allclose(usd['AAPL'].dropna(), 100.0)
        assert np.allclose(eur['AAPL'].dropna(), 100.0 * _USDEUR)

    def test_fx_rate_read_is_cached_across_assets(self):
        # AAPL and VWRL are both USD → the USDEUR pair must be read only once
        # (assets + one shared FX read = 3 reads total).
        mock = Mock(side_effect=_fx_mock_read())
        with patch('src.backtest.pd.read_parquet', mock):
            load_daily_closes(BASE_URL, ['aapl.parquet', 'vwrl.parquet'], FX_META, 'EUR')
        fx_reads = [c for c in mock.call_args_list if 'USDEUR' in c.args[0]]
        assert len(fx_reads) == 1

    def test_calendar_misalignment_ffilled_no_nan(self):
        # FX series covers fewer / offset days than the asset; ffill+bfill must
        # leave no NaN across the overlap so the converted series is complete.
        def _read(path, columns=None):
            if 'USDEUR' in path:                       # short, offset FX calendar
                return _fx_ohlcv(_USDEUR, n_days=50, start='2020-02-01')
            return _daily_ohlcv(100.0, n_days=400)
        with patch('src.backtest.pd.read_parquet', side_effect=_read):
            df = load_daily_closes(BASE_URL, ['aapl.parquet'], FX_META, 'EUR')
        assert not df['AAPL'].isna().any()
        assert np.allclose(df['AAPL'], 100.0 * _USDEUR)

    def test_currency_pair_as_basket_asset(self):
        # An FX pair added as an asset is converted by its own quote currency:
        # USDEUR=X is quoted in EUR, so with base=EUR it passes through unchanged
        # (its close is the rate itself).
        with patch('src.backtest.pd.read_parquet', side_effect=_fx_mock_read()):
            df = load_daily_closes(BASE_URL, ['USDEUR_X.parquet'], FX_META, 'EUR')
        assert np.allclose(df['USDEUR=X'].dropna(), _USDEUR)

    def test_currency_pair_asset_converted_when_base_differs(self):
        # Same pair (quoted in EUR) with base=USD → converted EUR→USD via EURUSD.
        with patch('src.backtest.pd.read_parquet', side_effect=_fx_mock_read()):
            df = load_daily_closes(BASE_URL, ['USDEUR_X.parquet'], FX_META, 'USD')
        assert np.allclose(df['USDEUR=X'].dropna(), _USDEUR * _EURUSD)
