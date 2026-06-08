import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

os.environ.pop("BASE_URL", None)

import src.strategies.registry as _registry_module  # noqa: E402
from src.strategies.base import BacktestStrategy, ConfigParam  # noqa: E402
from src.strategies.registry import get_strategy, list_strategies, register  # noqa: E402
from src.strategies.dca import DCAStrategy, _dca_order_events  # noqa: E402
from src.strategies.riskoff import RiskOffStrategy, _riskoff_order_events  # noqa: E402
from src.backtest import (  # noqa: E402
    INITIAL_INVESTMENT,
    build_equal_weight_index,
    compute_riskoff_signals,
    run_backtest,
    simulate_riskoff,
)

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_backtest.py conventions)
# ---------------------------------------------------------------------------

SAMPLE_META = pd.DataFrame({
    'asset_class': ['stocks', 'crypto'],
    'symbol': ['AAPL', 'BTC'],
    'name': ['Apple Inc', 'Bitcoin'],
    'filename': ['aapl.parquet', 'btc.parquet'],
})

BASE_URL = "http://example.com"

_MONTHLY_IDX = pd.date_range('2020-01-31', periods=24, freq='ME', tz='UTC')
_START = pd.Timestamp('2020-01-31', tz='UTC')
_END = pd.Timestamp('2021-12-31', tz='UTC')

_EXPECTED_METRIC_KEYS = {
    'Total Return', 'CAGR', 'Sharpe Ratio', 'Max. Drawdown',
    'Volatility (p.a.)', 'Calmar Ratio', 'Invested', 'End Value',
    'Profit/Loss',
}


def _daily_ohlcv(price: float, n_days: int = 1200) -> pd.DataFrame:
    """Constant-price daily OHLCV DataFrame for mocking pd.read_parquet.

    Default of 1200 days (~3.3 years from 2019-01-01) ensures coverage through
    the test window end of 2021-12-31 with margin to spare.
    """
    idx = pd.date_range('2019-01-01', periods=n_days, freq='D', tz='UTC')
    return pd.DataFrame(
        {'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': 1000},
        index=idx,
    )


def _daily_rising(low: float = 100.0, high: float = 400.0, n_days: int = 1200) -> pd.DataFrame:
    """Monotonically rising daily OHLCV DataFrame for mocking pd.read_parquet.

    A steady uptrend keeps all three Risk-Off signals positive, so the strategy
    stays fully invested (behaves like buy-and-hold of the lump sum).
    """
    idx = pd.date_range('2019-01-01', periods=n_days, freq='D', tz='UTC')
    close = np.linspace(low, high, n_days)
    return pd.DataFrame(
        {'Open': close, 'High': close, 'Low': close, 'Close': close, 'Volume': 1000},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Layer 1 – ConfigParam structure
# ---------------------------------------------------------------------------

class TestConfigParam:
    def test_float_param_has_min_and_max(self):
        p = ConfigParam(
            key='amount', label='Amount', type='float',
            default=100.0, min_value=1.0, max_value=10_000.0,
        )
        assert p.min_value is not None
        assert p.max_value is not None
        assert p.min_value < p.default <= p.max_value

    def test_int_param_stores_correct_values(self):
        p = ConfigParam(
            key='months', label='Months', type='int',
            default=12, min_value=1, max_value=120,
        )
        assert p.default == 12
        assert p.min_value == 1
        assert p.max_value == 120

    def test_select_param_has_non_empty_options(self):
        p = ConfigParam(
            key='mode', label='Mode', type='select',
            default='equal', options=['equal', 'weighted'],
        )
        assert isinstance(p.options, list)
        assert len(p.options) > 0
        assert p.default in p.options

    def test_options_list_is_not_shared_between_instances(self):
        # Verify that field(default_factory=list) creates a fresh list per
        # instance so mutating one instance doesn't affect another.
        p1 = ConfigParam(key='a', label='A', type='select', default='x', options=['x'])
        p2 = ConfigParam(key='b', label='B', type='select', default='y', options=['y'])
        p1.options.append('z')
        assert 'z' not in p2.options

    # --- validation tests ---------------------------------------------------

    def test_int_param_without_min_max_raises(self):
        with pytest.raises(ValueError, match="min_value and max_value"):
            ConfigParam(key='x', label='X', type='int', default=5)

    def test_float_param_without_max_raises(self):
        with pytest.raises(ValueError, match="min_value and max_value"):
            ConfigParam(key='x', label='X', type='float', default=1.0, min_value=0.0)

    def test_select_param_empty_options_raises(self):
        with pytest.raises(ValueError, match="options must be non-empty"):
            ConfigParam(key='x', label='X', type='select', default='a')

    def test_select_param_default_not_in_options_raises(self):
        with pytest.raises(ValueError, match="not in options"):
            ConfigParam(key='x', label='X', type='select', default='z', options=['a', 'b'])


# ---------------------------------------------------------------------------
# Layer 2 – Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Uses a local dummy strategy so tests are independent of DCAStrategy."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Snapshot _registry before each test and restore it afterwards.

        Prevents dummy registrations from leaking into subsequent tests or
        into the live registry used by the running app.
        """
        monkeypatch.setattr(_registry_module, '_registry', dict(_registry_module._registry))

    def _make_dummy(self, name: str) -> type[BacktestStrategy]:
        """Create and register a minimal BacktestStrategy subclass on the fly."""
        class _Dummy(BacktestStrategy):
            @classmethod
            def get_name(cls) -> str:
                return name

            @classmethod
            def get_description(cls) -> str:
                return "dummy"

            @classmethod
            def get_icon(cls) -> str:
                return "bi-question"

            @classmethod
            def get_config_schema(cls) -> list[ConfigParam]:
                return []

            def run(self, base_url, filenames, start_date, end_date, df_meta, params):
                return None, None

        register(_Dummy)
        return _Dummy

    def test_register_adds_to_list(self):
        self._make_dummy("_test_dummy_a")
        assert "_test_dummy_a" in list_strategies()

    def test_get_strategy_returns_registered_class(self):
        dummy_cls = self._make_dummy("_test_dummy_b")
        assert get_strategy("_test_dummy_b") is dummy_cls

    def test_get_strategy_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown strategy"):
            get_strategy("__nonexistent__")

    def test_re_register_overwrites_silently(self):
        self._make_dummy("_test_dummy_c")
        cls2 = self._make_dummy("_test_dummy_c")  # same name, different class
        assert get_strategy("_test_dummy_c") is cls2

    def test_registry_isolated_between_tests(self):
        # Confirm that dummy registrations from other tests are not visible here.
        assert "_test_dummy_a" not in list_strategies()
        assert "_test_dummy_b" not in list_strategies()


# ---------------------------------------------------------------------------
# Layer 3 – DCAStrategy unit tests
# ---------------------------------------------------------------------------

class TestDCAStrategy:
    def test_get_name_returns_dca(self):
        assert DCAStrategy.get_name() == 'DCA'

    def test_get_icon_returns_non_empty_string(self):
        icon = DCAStrategy.get_icon()
        assert isinstance(icon, str) and icon, "get_icon() must return a non-empty string"

    def test_get_description_returns_non_empty_string(self):
        assert isinstance(DCAStrategy.get_description(), str)
        assert DCAStrategy.get_description()

    def test_get_config_schema_contains_monthly_investment(self):
        schema = DCAStrategy.get_config_schema()
        keys = [p.key for p in schema]
        assert 'monthly_investment' in keys

    def test_all_params_have_non_none_default(self):
        # Every param must provide a default so the GUI can pre-fill it.
        for param in DCAStrategy.get_config_schema():
            assert param.default is not None, f"Param '{param.key}' has no default"

    def test_run_with_empty_params_uses_defaults(self):
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert portfolio is not None
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS
        # A successful run also produces a (non-empty) order log.
        assert orders is not None and len(orders) > 0

    def test_run_returns_9_metric_keys(self):
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, metrics, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert metrics is not None
        assert len(metrics) == 9

    def test_run_with_empty_filenames_returns_none_none(self):
        strategy = DCAStrategy()
        portfolio, metrics, orders = strategy.run(
            BASE_URL, [], _START, _END, SAMPLE_META, params={}
        )
        assert portfolio is None
        assert metrics is None
        assert orders is None

    def test_run_with_too_short_date_range_returns_none_none(self):
        # Only 1 month of data → compute_metrics needs ≥3 → run() returns 3×None.
        strategy = DCAStrategy()
        start = pd.Timestamp('2020-06-30', tz='UTC')
        end = pd.Timestamp('2020-06-30', tz='UTC')
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META, params={}
            )
        assert portfolio is None
        assert metrics is None
        assert orders is None

    def test_custom_monthly_investment_halves_invested_total(self):
        # At the same constant price, halving the monthly investment halves the
        # invested total.  Use that relationship to verify params are respected.
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, m_default, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
            _, m_half, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META,
                params={'monthly_investment': 500.0},
            )

        assert m_default is not None and m_half is not None
        invested_default = float(m_default['Invested'].replace(',', ''))
        invested_half = float(m_half['Invested'].replace(',', ''))
        assert invested_default == pytest.approx(invested_half * 2, rel=1e-3)

    def test_dca_is_registered_in_registry(self):
        assert "DCA" in list_strategies()
        assert get_strategy("DCA") is DCAStrategy


# ---------------------------------------------------------------------------
# Layer 4 – Backward-compatibility regression
# ---------------------------------------------------------------------------

class TestRunBacktestBackwardCompat:
    _START = pd.Timestamp('2020-01-31', tz='UTC')
    _END = pd.Timestamp('2021-12-31', tz='UTC')

    def test_five_positional_args_still_work(self):
        # Existing callers pass exactly 5 positional args; must not break.
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            p, m, o = run_backtest(
                BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META
            )
        assert p is not None
        assert isinstance(m, dict)
        assert set(m.keys()) == _EXPECTED_METRIC_KEYS
        # Built-in DCA path (no strategy) produces no order log.
        assert o is None

    def test_with_strategy_plugin_returns_correct_structure(self):
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            p, m, o = run_backtest(
                BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META,
                strategy=strategy,
                strategy_params={'monthly_investment': 500.0},
            )
        assert p is not None
        assert isinstance(m, dict)
        assert set(m.keys()) == _EXPECTED_METRIC_KEYS
        # Routing through a plugin DOES produce an order log.
        assert o is not None and len(o) > 0

    def test_strategy_params_honoured_in_run_backtest(self):
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, m_default, _ = run_backtest(
                BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META,
                strategy=strategy,
                strategy_params={},
            )
            _, m_half, _ = run_backtest(
                BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META,
                strategy=strategy,
                strategy_params={'monthly_investment': 500.0},
            )

        assert m_default is not None and m_half is not None
        invested_default = float(m_default['Invested'].replace(',', ''))
        invested_half = float(m_half['Invested'].replace(',', ''))
        assert invested_default == pytest.approx(invested_half * 2, rel=1e-3)

    def test_strategy_with_no_filenames_returns_none_none(self):
        strategy = DCAStrategy()
        p, m, o = run_backtest(
            BASE_URL, [], self._START, self._END, SAMPLE_META,
            strategy=strategy,
        )
        assert p is None and m is None and o is None


# ---------------------------------------------------------------------------
# Layer 5 – RiskOffStrategy plugin
# ---------------------------------------------------------------------------

class TestRiskOffStrategy:
    def test_is_registered_in_registry(self):
        assert "Risk-Off Signale" in list_strategies()
        assert get_strategy("Risk-Off Signale") is RiskOffStrategy

    def test_get_name_and_icon(self):
        assert RiskOffStrategy.get_name() == 'Risk-Off Signale'
        icon = RiskOffStrategy.get_icon()
        assert isinstance(icon, str) and icon

    def test_get_description_is_non_empty_string(self):
        assert isinstance(RiskOffStrategy.get_description(), str)
        assert RiskOffStrategy.get_description()

    def test_config_schema_keys_and_defaults(self):
        schema = RiskOffStrategy.get_config_schema()
        by_key = {p.key: p for p in schema}
        assert set(by_key) == {'initial_investment', 'sma_window', 'first_n_days'}
        # Every param must provide a default so the GUI can pre-fill it.
        for param in schema:
            assert param.default is not None, f"Param '{param.key}' has no default"
        assert by_key['initial_investment'].default == float(INITIAL_INVESTMENT)
        assert by_key['sma_window'].default == 200
        assert by_key['first_n_days'].default == 10

    def test_run_returns_exactly_9_metric_keys(self):
        strategy = RiskOffStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert portfolio is not None
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS
        assert len(metrics) == 9
        # A rising market deploys the lump sum → at least one (Buy) order.
        assert orders is not None and len(orders) >= 1

    def test_run_with_empty_filenames_returns_none_none(self):
        strategy = RiskOffStrategy()
        portfolio, metrics, orders = strategy.run(
            BASE_URL, [], _START, _END, SAMPLE_META, params={}
        )
        assert portfolio is None and metrics is None and orders is None

    def test_run_with_too_short_date_range_returns_none_none(self):
        # Only 1 month in the window → compute_metrics needs ≥3 → 3×None.
        strategy = RiskOffStrategy()
        start = pd.Timestamp('2020-06-30', tz='UTC')
        end = pd.Timestamp('2020-06-30', tz='UTC')
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META, params={}
            )
        assert portfolio is None and metrics is None and orders is None

    def test_constant_price_stays_in_cash(self):
        # Flat market → every signal negative → 0 % invested → value never moves
        # from the initial lump sum, so Total Return is ~0 % and End == Invested.
        strategy = RiskOffStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert metrics is not None
        assert metrics['Total Return'] == '+0.0%'
        invested = float(metrics['Invested'].replace(',', ''))
        end_value = float(metrics['End Value'].replace(',', ''))
        assert invested == pytest.approx(end_value)
        # The lump sum used should be the configured default.
        assert invested == pytest.approx(float(INITIAL_INVESTMENT))
        # Never invested → the target fraction never changes → no orders at all.
        assert orders == []

    def test_rising_market_is_fully_invested(self):
        # Steady uptrend → all signals positive → behaves like buy-and-hold:
        # the portfolio grows well above the invested lump sum.
        strategy = RiskOffStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert metrics is not None
        invested = float(metrics['Invested'].replace(',', ''))
        end_value = float(metrics['End Value'].replace(',', ''))
        assert end_value > invested
        assert metrics['Total Return'].startswith('+')
        # Deploying into the basket is recorded as a Buy in the order log.
        assert orders is not None and orders[0]['side'] == 'Buy'

    def test_custom_initial_investment_scales_invested(self):
        # The reported 'Invested' equals the configured lump sum.
        strategy = RiskOffStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            _, metrics, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META,
                params={'initial_investment': 25_000.0},
            )
        assert metrics is not None
        invested = float(metrics['Invested'].replace(',', ''))
        assert invested == pytest.approx(25_000.0)


# ---------------------------------------------------------------------------
# Layer 6 – Risk-Off pure functions (no Dash / no I/O)
# ---------------------------------------------------------------------------

class TestRiskOffPureFunctions:
    def test_equal_weight_index_single_asset(self):
        idx = pd.date_range('2020-01-01', periods=5, freq='D', tz='UTC')
        df = pd.DataFrame({'A': [10.0, 11.0, 12.0, 11.0, 13.0]}, index=idx)
        result = build_equal_weight_index(df)
        expected = df['A'] / 10.0 * 100.0
        assert result.round(6).tolist() == expected.round(6).tolist()

    def test_equal_weight_index_handles_mixed_start_dates(self):
        # Asset B starts a day later (NaN on day 0); the index must not be NaN.
        idx = pd.date_range('2020-01-01', periods=4, freq='D', tz='UTC')
        df = pd.DataFrame({'A': [10.0, 11.0, 12.0, 12.0], 'B': [np.nan, 20.0, 22.0, 21.0]}, index=idx)
        result = build_equal_weight_index(df)
        assert not result.isna().any()
        assert result.iloc[0] == pytest.approx(100.0)

    def test_sma_trend_signal_warmup_is_negative(self):
        # Within the warm-up window the rolling mean is NaN → signal False.
        from src.backtest import _sma_trend_signal
        idx = pd.date_range('2020-01-01', periods=10, freq='D', tz='UTC')
        s = pd.Series(np.linspace(100, 110, 10), index=idx)
        assert not _sma_trend_signal(s, window=200).any()

    def test_ytd_signal_flips_below_year_start(self):
        from src.backtest import _ytd_return_signal
        dates = pd.to_datetime(
            ['2020-01-02', '2020-06-01', '2020-12-01', '2021-01-04', '2021-06-01']
        ).tz_localize('UTC')
        s = pd.Series([100.0, 110.0, 120.0, 90.0, 95.0], index=dates)
        # 2020 starts at 100 (False at start, then above → True); 2021 starts at
        # 90 (False at start, then above 90 → True).
        assert _ytd_return_signal(s).tolist() == [False, True, True, False, True]

    def test_first_n_days_partial_year_is_negative(self):
        from src.backtest import _first_n_days_signal
        dates = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']).tz_localize('UTC')
        s = pd.Series([100.0, 101.0, 102.0], index=dates)
        # Fewer than 10 trading days → conservative False everywhere.
        assert not _first_n_days_signal(s, n=10).any()
        # With n=3 the barometer is satisfied (102 > 100) → True everywhere.
        assert _first_n_days_signal(s, n=3).all()

    def test_compute_riskoff_signals_counts_positive_signals(self):
        idx = pd.date_range('2019-01-01', periods=400, freq='D', tz='UTC')
        # Steady uptrend → all three signals positive → count saturates at 3.
        rising = pd.Series(np.linspace(100, 300, 400), index=idx)
        rising_count = compute_riskoff_signals(rising, sma_window=50, first_n_days=10)
        assert int(rising_count.max()) == 3
        assert rising_count.iloc[-1] == 3
        # Flat market → no signal positive → count is always 0.
        flat = pd.Series(100.0, index=idx)
        flat_count = compute_riskoff_signals(flat, sma_window=50, first_n_days=10)
        assert sorted(flat_count.unique().tolist()) == [0]

    def test_simulate_riskoff_full_invested_equals_buy_and_hold(self):
        idx = pd.date_range('2020-01-31', periods=6, freq='ME', tz='UTC')
        prices = pd.DataFrame({'A': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]}, index=idx)
        frac = pd.Series(1.0, index=idx)
        portfolio, invested = simulate_riskoff(prices, frac, initial_investment=10_000.0)
        assert invested == 10_000.0
        # Lump sum grows 100→150 → 10,000 → 15,000.
        assert portfolio.iloc[0] == pytest.approx(10_000.0)
        assert portfolio.iloc[-1] == pytest.approx(15_000.0)

    def test_simulate_riskoff_trades_on_the_day_the_target_changes(self):
        # Flat price until the last day; the target turns on at index 2. The
        # trade must execute on that day, so when the price jumps on the final
        # day the (now invested) lump sum doubles in value.
        idx = pd.date_range('2020-01-31', periods=4, freq='ME', tz='UTC')
        prices = pd.DataFrame({'A': [100.0, 100.0, 100.0, 200.0]}, index=idx)
        target = pd.Series([0.0, 0.0, 1.0, 1.0], index=idx)
        portfolio, _ = simulate_riskoff(prices, target, initial_investment=10_000.0)
        # Still all cash while the target is 0 (no deployment before the change).
        assert portfolio.iloc[0] == pytest.approx(10_000.0)
        # Fully invested by index 2, so the price doubling on the last day
        # doubles the portfolio (100 shares × 200).
        assert portfolio.iloc[-1] == pytest.approx(20_000.0)

    def test_simulate_riskoff_holds_and_drifts_between_changes(self):
        # The target never changes after the initial deployment (constant 0.5),
        # so the basket is bought once and then HELD: as the price rises and
        # falls back the portfolio drifts and returns exactly to its start. A
        # daily "maintain-the-fraction" strategy would instead end at 11,250.
        idx = pd.date_range('2020-01-31', periods=3, freq='ME', tz='UTC')
        prices = pd.DataFrame({'A': [100.0, 200.0, 100.0]}, index=idx)
        target = pd.Series([0.5, 0.5, 0.5], index=idx)
        portfolio, _ = simulate_riskoff(prices, target, initial_investment=10_000.0)
        # Bought 50 shares (5,000) + 5,000 cash at index 0, then held.
        assert portfolio.iloc[0] == pytest.approx(10_000.0)
        assert portfolio.iloc[1] == pytest.approx(15_000.0)   # 50×200 + 5,000
        assert portfolio.iloc[-1] == pytest.approx(10_000.0)  # 50×100 + 5,000
        # Confirm we did NOT rebalance daily (which would have ended at 11,250).
        assert portfolio.iloc[-1] != pytest.approx(11_250.0)

    def test_simulate_riskoff_zero_fraction_stays_constant(self):
        idx = pd.date_range('2020-01-31', periods=6, freq='ME', tz='UTC')
        prices = pd.DataFrame({'A': [100.0, 110.0, 90.0, 130.0, 70.0, 150.0]}, index=idx)
        frac = pd.Series(0.0, index=idx)
        portfolio, _ = simulate_riskoff(prices, frac, initial_investment=10_000.0)
        assert (portfolio == 10_000.0).all()

    def test_simulate_riskoff_cash_caps_drawdown(self):
        # A crash after the peak: going to cash before the worst of the decline
        # produces a strictly smaller (less negative) max drawdown than holding.
        idx = pd.date_range('2020-01-31', periods=6, freq='ME', tz='UTC')
        prices = pd.DataFrame({'A': [100.0, 120.0, 150.0, 90.0, 80.0, 75.0]}, index=idx)

        def _max_drawdown(pf: pd.Series) -> float:
            peak = pf.expanding().max()
            return ((pf - peak) / peak).min()

        hold, _ = simulate_riskoff(prices, pd.Series(1.0, index=idx), initial_investment=10_000.0)
        protect, _ = simulate_riskoff(
            prices, pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=idx), initial_investment=10_000.0
        )
        assert _max_drawdown(protect) > _max_drawdown(hold)

    def test_simulate_riskoff_cash_out_then_redeploy(self):
        # Deploy → sell to cash → hold cash → redeploy: multiple change segments,
        # exercising the vectorised engine's per-segment holdings/cash fill.
        idx = pd.date_range('2020-01-31', periods=6, freq='ME', tz='UTC')
        prices = pd.DataFrame({'A': [100.0, 120.0, 150.0, 90.0, 80.0, 200.0]}, index=idx)
        target = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], index=idx)
        portfolio, _ = simulate_riskoff(prices, target, initial_investment=10_000.0)
        # Deploy 100 sh @100; hold to 120 → 12,000; sell all @150 → 15,000 cash;
        # hold cash through 90; redeploy 15,000 @80 = 187.5 sh; hold to 200 → 37,500.
        assert portfolio.tolist() == pytest.approx(
            [10_000.0, 12_000.0, 15_000.0, 15_000.0, 15_000.0, 37_500.0]
        )


# ---------------------------------------------------------------------------
# Layer 7 – per-plugin order-event generators (the strategy-specific halves
# of the order log; the generic finalize step is tested in test_backtest.py)
# ---------------------------------------------------------------------------

class TestDcaOrderEvents:
    def test_empty_frame_returns_no_events(self):
        assert _dca_order_events(pd.DataFrame(), 1000.0) == []

    def test_one_buy_event_per_month_end(self):
        # 3 calendar months of daily constant-price data → 3 contribution days.
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        price = pd.DataFrame({'AAPL': 100.0}, index=idx)
        events = _dca_order_events(price, 1000.0)
        assert len(events) == 3
        # Every DCA contribution is a Buy with the fixed inflow and no cash
        # (DCA is always fully invested).
        assert all(e['side'] == 'Buy' for e in events)
        assert all(e['inflow'] == pytest.approx(1000.0) for e in events)
        assert all(e['cash_after'] == pytest.approx(0.0) for e in events)

    def test_first_event_has_zero_value_before(self):
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        price = pd.DataFrame({'AAPL': 100.0}, index=idx)
        events = _dca_order_events(price, 1000.0)
        # Nothing held before the first contribution.
        assert events[0]['value_before'] == pytest.approx(0.0)
        # After it, one month's money is invested at the constant price.
        assert events[0]['assets_after'] == pytest.approx(1000.0)

    def test_event_dates_are_month_ends(self):
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        price = pd.DataFrame({'AAPL': 100.0}, index=idx)
        events = _dca_order_events(price, 1000.0)
        assert [e['date'].month for e in events] == [1, 2, 3]
        assert [e['date'].day for e in events] == [31, 29, 31]  # 2020 is a leap year


class TestRiskOffOrderEvents:
    _IDX = pd.date_range('2020-01-31', periods=4, freq='ME', tz='UTC')

    def test_empty_frame_returns_no_events(self):
        assert _riskoff_order_events(pd.DataFrame(), pd.Series(dtype=float), 10_000.0) == []

    def test_constant_zero_target_never_trades(self):
        price = pd.DataFrame({'A': [100.0, 110.0, 120.0, 130.0]}, index=self._IDX)
        target = pd.Series(0.0, index=self._IDX)
        assert _riskoff_order_events(price, target, 10_000.0) == []

    def test_buy_then_sell_on_target_changes(self):
        # Target 0 → 1 (buy) → 1 → 0 (sell). Price doubles before the sell.
        price = pd.DataFrame({'A': [100.0, 100.0, 200.0, 200.0]}, index=self._IDX)
        target = pd.Series([0.0, 1.0, 1.0, 0.0], index=self._IDX)
        events = _riskoff_order_events(price, target, 10_000.0)
        assert len(events) == 2
        buy, sell = events
        assert buy['side'] == 'Buy'
        assert sell['side'] == 'Sell'
        # Fully invested after the buy (no cash); fully in cash after the sell.
        assert buy['cash_after'] == pytest.approx(0.0)
        assert buy['assets_after'] == pytest.approx(10_000.0)
        assert sell['assets_after'] == pytest.approx(0.0)
        assert sell['cash_after'] == pytest.approx(20_000.0)  # holdings doubled
        # No fresh money ever enters a Risk-Off rebalance.
        assert all(e['inflow'] == pytest.approx(0.0) for e in events)

    def test_partial_target_splits_between_assets_and_cash(self):
        # A 0 → 0.5 change invests half the lump sum, leaving half in cash.
        price = pd.DataFrame({'A': [100.0, 100.0, 100.0, 100.0]}, index=self._IDX)
        target = pd.Series([0.0, 0.5, 0.5, 0.5], index=self._IDX)
        events = _riskoff_order_events(price, target, 10_000.0)
        assert len(events) == 1
        (ev,) = events
        assert ev['side'] == 'Buy'
        assert ev['assets_after'] == pytest.approx(5_000.0)
        assert ev['cash_after'] == pytest.approx(5_000.0)
