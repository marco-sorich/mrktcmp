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
from src.strategies.lumpsum import BuyHoldStrategy, _lumpsum_order_events  # noqa: E402
from src.strategies.riskoff import RiskOffStrategy, _riskoff_order_events  # noqa: E402
from src.strategies.summergap import SummerGapStrategy, _seasonal_target  # noqa: E402
from src.strategies.loserrotation import (  # noqa: E402
    LoserRotationStrategy,
    _build_rotation_events,
    _first_trading_day_of_year,
    _first_trading_day_on_or_after,
    _select_losers,
)
from src.backtest import (  # noqa: E402
    INITIAL_INVESTMENT,
    FxColumns,
    build_equal_weight_index,
    compute_riskoff_signals,
    gate_target_until_all_priced,
    run_backtest,
    simulate_riskoff,
    simulate_rotation,
    _rotation_order_events,
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

            def run(self, base_url, filenames, start_date, end_date, df_meta,
                    params, base_currency='EUR'):
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

    def test_get_long_description_defaults_to_description(self):
        # _Dummy does not override get_long_description, so the base-class default
        # must fall back to the one-sentence get_description().
        dummy_cls = self._make_dummy("_test_dummy_long")
        assert dummy_cls.get_long_description() == dummy_cls.get_description()


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

    def test_get_long_description_returns_rich_markdown(self):
        long = DCAStrategy.get_long_description()
        assert isinstance(long, str) and long
        # The override must be a fuller write-up, not just the one-liner.
        assert len(long) > len(DCAStrategy.get_description())

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
# Layer 3b – BuyHoldStrategy unit tests (one single initial investment)
# ---------------------------------------------------------------------------

class TestBuyHoldStrategy:
    def test_get_name_returns_buy_and_hold(self):
        assert BuyHoldStrategy.get_name() == 'Buy & Hold'

    def test_get_icon_returns_non_empty_string(self):
        icon = BuyHoldStrategy.get_icon()
        assert isinstance(icon, str) and icon, "get_icon() must return a non-empty string"

    def test_get_description_returns_non_empty_string(self):
        assert isinstance(BuyHoldStrategy.get_description(), str)
        assert BuyHoldStrategy.get_description()

    def test_get_long_description_returns_rich_markdown(self):
        long = BuyHoldStrategy.get_long_description()
        assert isinstance(long, str) and long
        assert len(long) > len(BuyHoldStrategy.get_description())

    def test_get_config_schema_contains_initial_investment(self):
        schema = BuyHoldStrategy.get_config_schema()
        keys = [p.key for p in schema]
        assert 'initial_investment' in keys

    def test_all_params_have_non_none_default(self):
        for param in BuyHoldStrategy.get_config_schema():
            assert param.default is not None, f"Param '{param.key}' has no default"

    def test_run_with_empty_params_uses_defaults(self):
        strategy = BuyHoldStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert portfolio is not None
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS
        # Buy & Hold makes exactly one trade, so the order log has a single row.
        assert orders is not None and len(orders) == 1

    def test_run_returns_9_metric_keys(self):
        strategy = BuyHoldStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, metrics, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert metrics is not None
        assert len(metrics) == 9

    def test_run_with_empty_filenames_returns_none_none(self):
        strategy = BuyHoldStrategy()
        portfolio, metrics, orders = strategy.run(
            BASE_URL, [], _START, _END, SAMPLE_META, params={}
        )
        assert portfolio is None
        assert metrics is None
        assert orders is None

    def test_run_with_too_short_date_range_returns_none_none(self):
        # Only 1 month of data → compute_metrics needs ≥3 → run() returns 3×None.
        strategy = BuyHoldStrategy()
        start = pd.Timestamp('2020-06-30', tz='UTC')
        end = pd.Timestamp('2020-06-30', tz='UTC')
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META, params={}
            )
        assert portfolio is None
        assert metrics is None
        assert orders is None

    def test_custom_initial_investment_scales_invested_total(self):
        # The whole lump sum is the invested total, so halving the param halves
        # the reported 'Invested' figure.
        strategy = BuyHoldStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, m_default, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
            _, m_half, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META,
                params={'initial_investment': float(INITIAL_INVESTMENT) / 2},
            )

        assert m_default is not None and m_half is not None
        invested_default = float(m_default['Invested'].replace(',', ''))
        invested_half = float(m_half['Invested'].replace(',', ''))
        assert invested_default == pytest.approx(invested_half * 2, rel=1e-3)

    def test_buy_and_hold_is_registered_in_registry(self):
        assert "Buy & Hold" in list_strategies()
        assert get_strategy("Buy & Hold") is BuyHoldStrategy

    def test_buy_and_hold_is_the_default_first_strategy(self):
        # Imported first in strategies/__init__.py so the GUI defaults to it.
        assert list_strategies()[0] == "Buy & Hold"


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
        assert "Risk-Off" in list_strategies()
        assert get_strategy("Risk-Off") is RiskOffStrategy

    def test_get_name_and_icon(self):
        assert RiskOffStrategy.get_name() == 'Risk-Off'
        icon = RiskOffStrategy.get_icon()
        assert isinstance(icon, str) and icon

    def test_get_description_is_non_empty_string(self):
        assert isinstance(RiskOffStrategy.get_description(), str)
        assert RiskOffStrategy.get_description()

    def test_get_long_description_returns_rich_markdown(self):
        long = RiskOffStrategy.get_long_description()
        assert isinstance(long, str) and long
        assert len(long) > len(RiskOffStrategy.get_description())

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


class TestGateTargetUntilAllPriced:
    _IDX = pd.date_range('2020-01-31', periods=4, freq='ME', tz='UTC')

    def test_zeroes_target_until_whole_basket_is_priced(self):
        # B lists only on day three → the target is forced to cash on days 1–2 and
        # passes through unchanged from the first all-priced day on.
        price = pd.DataFrame(
            {'A': [100.0, 100.0, 100.0, 100.0], 'B': [np.nan, np.nan, 200.0, 200.0]},
            index=self._IDX,
        )
        target = pd.Series([1.0, 1.0, 1.0, 1.0], index=self._IDX)
        gated = gate_target_until_all_priced(price, target)
        assert gated.tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_single_asset_passes_through_unchanged(self):
        # One asset priced from day one → nothing to gate.
        price = pd.DataFrame({'A': [100.0, 100.0, 100.0, 100.0]}, index=self._IDX)
        target = pd.Series([0.0, 0.33, 0.67, 1.0], index=self._IDX)
        gated = gate_target_until_all_priced(price, target)
        assert gated.tolist() == pytest.approx([0.0, 0.33, 0.67, 1.0])

    def test_all_cash_when_an_asset_is_never_priced(self):
        # B never gets a price → the target is zeroed for the whole window.
        price = pd.DataFrame(
            {'A': [100.0, 100.0], 'B': [np.nan, np.nan]},
            index=self._IDX[:2],
        )
        target = pd.Series([1.0, 1.0], index=self._IDX[:2])
        gated = gate_target_until_all_priced(price, target)
        assert gated.tolist() == pytest.approx([0.0, 0.0])

    def test_does_not_mutate_input(self):
        price = pd.DataFrame(
            {'A': [100.0, 100.0], 'B': [np.nan, 200.0]},
            index=self._IDX[:2],
        )
        target = pd.Series([1.0, 1.0], index=self._IDX[:2])
        gate_target_until_all_priced(price, target)
        assert target.tolist() == pytest.approx([1.0, 1.0])


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

    def test_asset_values_break_down_assets_after(self):
        # Two assets → each contribution is split equally; the per-asset values
        # carry the asset symbols and sum back to assets_after on every event.
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        price = pd.DataFrame({'AAPL': 100.0, 'MSFT': 200.0}, index=idx)
        events = _dca_order_events(price, 1000.0)
        first = events[0]['asset_values']
        assert set(first) == {'AAPL', 'MSFT'}
        # Equal € split (500 each) on the first contribution.
        assert first['AAPL'] == pytest.approx(500.0)
        assert first['MSFT'] == pytest.approx(500.0)
        for e in events:
            assert sum(e['asset_values'].values()) == pytest.approx(e['assets_after'])

    def test_asset_prices_carry_each_close(self):
        # Each event also records the assets' close prices on its contribution day.
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        price = pd.DataFrame({'AAPL': 100.0, 'MSFT': 200.0}, index=idx)
        events = _dca_order_events(price, 1000.0)
        assert events[0]['asset_prices'] == pytest.approx({'AAPL': 100.0, 'MSFT': 200.0})

    def test_no_fx_context_local_equals_base_and_no_fx_pairs(self):
        # Without FX context the local price passes through (= the base close) and
        # no FX-pair rates are recorded, so the order table shows no extra columns.
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        price = pd.DataFrame({'AAPL': 100.0}, index=idx)
        ev = _dca_order_events(price, 1000.0)[0]
        assert ev['asset_prices_local'] == ev['asset_prices']
        assert ev['fx_rates'] == {}


class TestOrderEventsFxContext:
    """An FX context (FxColumns) makes the generators record the trading-currency
    quote and the per-pair rate on every event, for all three strategies."""

    def _price(self):
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        # AAPL holds a base-currency (e.g. EUR) close of 90 = 100 USD × 0.90.
        return pd.DataFrame({'AAPL': 90.0}, index=idx)

    def _fx(self, index):
        # AAPL trades in USD; the rate (0.90 base per USD) is constant over `index`.
        rate = pd.Series(0.90, index=index)
        return FxColumns(
            asset_local_ccy={'AAPL': 'USD'},
            asset_rate={'AAPL': rate},
            pair_rate={'USDEUR=X': rate},
        )

    def test_dca_records_local_price_and_fx_rate(self):
        price = self._price()
        fx = self._fx(price.index)
        ev = _dca_order_events(price, 1000.0, fx)[0]
        # 90 base / 0.90 rate → 100 in the trading currency.
        assert ev['asset_prices_local']['AAPL'] == pytest.approx(100.0)
        assert ev['fx_rates'] == {'USDEUR=X': pytest.approx(0.90)}

    def test_lumpsum_records_local_price_and_fx_rate(self):
        price = self._price()
        fx = self._fx(price.index)
        ev = _lumpsum_order_events(price, 10_000.0, fx)[0]
        assert ev['asset_prices_local']['AAPL'] == pytest.approx(100.0)
        assert ev['fx_rates'] == {'USDEUR=X': pytest.approx(0.90)}

    def test_riskoff_records_local_price_and_fx_rate(self):
        price = self._price()
        fx = self._fx(price.index)
        # A target that turns on once so exactly one rebalance event is produced.
        target = pd.Series(0.0, index=price.index)
        target.iloc[10:] = 1.0
        ev = _riskoff_order_events(price, target, 10_000.0, fx)[0]
        assert ev['asset_prices_local']['AAPL'] == pytest.approx(100.0)
        assert ev['fx_rates'] == {'USDEUR=X': pytest.approx(0.90)}


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
        # The single asset holds the whole invested half; cash is excluded.
        assert ev['asset_values'] == pytest.approx({'A': 5_000.0})

    def test_asset_values_track_holdings_after_each_rebalance(self):
        price = pd.DataFrame({'A': [100.0, 100.0, 200.0, 200.0]}, index=self._IDX)
        target = pd.Series([0.0, 1.0, 1.0, 0.0], index=self._IDX)
        buy, sell = _riskoff_order_events(price, target, 10_000.0)
        # Fully invested after the buy, fully in cash (asset value 0) after sell.
        assert buy['asset_values'] == pytest.approx({'A': 10_000.0})
        assert sell['asset_values'] == pytest.approx({'A': 0.0})
        # The quote is recorded on both days — still meaningful after the sell,
        # where the position is 0 but the asset has doubled to 200.
        assert buy['asset_prices'] == pytest.approx({'A': 100.0})
        assert sell['asset_prices'] == pytest.approx({'A': 200.0})


class TestAssetWeighting:
    """Per-asset weights flow from run()/order events into the allocation."""

    def _two_asset_read(self):
        """read_parquet side-effect: AAPL flat at 100, BTC rising 100→400."""
        flat = _daily_ohlcv(100.0)
        rising = _daily_rising(low=100.0, high=400.0)

        def _read(path, columns=None):
            return rising if 'btc' in path else flat
        return _read

    def test_buyhold_weights_route_capital(self):
        strategy = BuyHoldStrategy()
        with patch('src.backtest.pd.read_parquet', side_effect=self._two_asset_read()):
            _, m_into_flat, _ = strategy.run(
                BASE_URL, ['aapl.parquet', 'btc.parquet'], _START, _END, SAMPLE_META,
                params={}, weights={'AAPL': 1.0, 'BTC': 0.0},
            )
            _, m_into_rising, _ = strategy.run(
                BASE_URL, ['aapl.parquet', 'btc.parquet'], _START, _END, SAMPLE_META,
                params={}, weights={'AAPL': 0.0, 'BTC': 1.0},
            )
        end_flat = float(m_into_flat['End Value'].replace(',', ''))
        end_rising = float(m_into_rising['End Value'].replace(',', ''))
        # All capital in the flat asset barely moves; all in the rising asset grows.
        assert end_rising > end_flat

    def test_run_backtest_forwards_weights(self):
        # run_backtest must thread weights through to the plugin; weighting only
        # the rising asset beats weighting only the flat one.
        strategy = BuyHoldStrategy()
        with patch('src.backtest.pd.read_parquet', side_effect=self._two_asset_read()):
            _, m_flat, _ = run_backtest(
                BASE_URL, ['aapl.parquet', 'btc.parquet'], _START, _END, SAMPLE_META,
                strategy=strategy, weights={'AAPL': 1.0, 'BTC': 0.0},
            )
            _, m_rising, _ = run_backtest(
                BASE_URL, ['aapl.parquet', 'btc.parquet'], _START, _END, SAMPLE_META,
                strategy=strategy, weights={'AAPL': 0.0, 'BTC': 1.0},
            )
        assert float(m_rising['End Value'].replace(',', '')) > float(m_flat['End Value'].replace(',', ''))

    def test_dca_order_events_split_by_weight(self):
        # First contribution split 3:1 across two equally-priced assets.
        idx = pd.date_range('2020-01-01', '2020-03-31', freq='D', tz='UTC')
        price = pd.DataFrame({'AAPL': 100.0, 'MSFT': 100.0}, index=idx)
        ev = _dca_order_events(price, 1000.0, None, {'AAPL': 3.0, 'MSFT': 1.0})[0]
        assert ev['asset_values']['AAPL'] == pytest.approx(750.0)
        assert ev['asset_values']['MSFT'] == pytest.approx(250.0)

    def test_lumpsum_order_event_splits_by_weight(self):
        idx = pd.date_range('2020-01-31', periods=4, freq='ME', tz='UTC')
        price = pd.DataFrame({'A': [100.0] * 4, 'B': [100.0] * 4}, index=idx)
        ev = _lumpsum_order_events(price, 10_000.0, None, {'A': 3.0, 'B': 1.0})[0]
        assert ev['asset_values'] == pytest.approx({'A': 7_500.0, 'B': 2_500.0})


class TestLumpsumOrderEvents:
    _IDX = pd.date_range('2020-01-31', periods=4, freq='ME', tz='UTC')

    def test_empty_frame_returns_no_events(self):
        assert _lumpsum_order_events(pd.DataFrame(), 10_000.0) == []

    def test_single_buy_on_first_day_fully_invested(self):
        # One trade only: the whole lump sum invested on day one, no cash left.
        price = pd.DataFrame({'A': [100.0, 110.0, 120.0, 130.0]}, index=self._IDX)
        events = _lumpsum_order_events(price, 10_000.0)
        assert len(events) == 1
        (ev,) = events
        assert ev['side'] == 'Buy'
        assert ev['date'] == self._IDX[0]
        assert ev['value_before'] == pytest.approx(0.0)
        assert ev['inflow'] == pytest.approx(10_000.0)
        assert ev['assets_after'] == pytest.approx(10_000.0)
        assert ev['cash_after'] == pytest.approx(0.0)

    def test_buy_split_equally_across_assets(self):
        price = pd.DataFrame(
            {'A': [100.0, 100.0], 'B': [200.0, 200.0]},
            index=self._IDX[:2],
        )
        events = _lumpsum_order_events(price, 10_000.0)
        assert len(events) == 1
        # 5,000 into each asset on day one → fully invested.
        assert events[0]['assets_after'] == pytest.approx(10_000.0)
        # The per-asset breakdown carries both symbols with the equal split …
        assert events[0]['asset_values'] == pytest.approx({'A': 5_000.0, 'B': 5_000.0})
        # … and both assets' day-one close prices.
        assert events[0]['asset_prices'] == pytest.approx({'A': 100.0, 'B': 200.0})

    def test_buy_waits_for_first_buyable_day(self):
        # No price on day one → the buy lands on the first day with a valid price.
        price = pd.DataFrame({'A': [np.nan, 100.0, 100.0, 100.0]}, index=self._IDX)
        events = _lumpsum_order_events(price, 10_000.0)
        assert len(events) == 1
        assert events[0]['date'] == self._IDX[1]

    def test_buy_waits_until_all_assets_priced(self):
        # B lists only on day three → the buy waits until both A and B are priced,
        # so the lump sum is split equally across the whole basket.
        price = pd.DataFrame(
            {'A': [100.0, 100.0, 100.0, 100.0], 'B': [np.nan, np.nan, 200.0, 200.0]},
            index=self._IDX,
        )
        events = _lumpsum_order_events(price, 10_000.0)
        assert len(events) == 1
        assert events[0]['date'] == self._IDX[2]
        assert events[0]['asset_values'] == pytest.approx({'A': 5_000.0, 'B': 5_000.0})

    def test_no_event_when_an_asset_is_never_priced(self):
        # B never gets a price → the basket is never fully priced, so nothing is
        # deployed and no order is recorded.
        price = pd.DataFrame(
            {'A': [100.0, 100.0], 'B': [np.nan, np.nan]},
            index=self._IDX[:2],
        )
        assert _lumpsum_order_events(price, 10_000.0) == []


# ---------------------------------------------------------------------------
# Layer 7 – SummerGapStrategy plugin
# ---------------------------------------------------------------------------

class TestSummerGapStrategy:
    def test_is_registered_in_registry(self):
        assert "Summer Gap" in list_strategies()
        assert get_strategy("Summer Gap") is SummerGapStrategy

    def test_get_name_and_icon(self):
        assert SummerGapStrategy.get_name() == 'Summer Gap'
        icon = SummerGapStrategy.get_icon()
        assert isinstance(icon, str) and icon

    def test_get_description_is_non_empty_string(self):
        assert isinstance(SummerGapStrategy.get_description(), str)
        assert SummerGapStrategy.get_description()

    def test_get_long_description_returns_rich_markdown(self):
        long = SummerGapStrategy.get_long_description()
        assert isinstance(long, str) and long
        assert len(long) > len(SummerGapStrategy.get_description())

    def test_config_schema_keys_and_defaults(self):
        schema = SummerGapStrategy.get_config_schema()
        by_key = {p.key: p for p in schema}
        assert set(by_key) == {
            'initial_investment', 'sell_month', 'sell_day', 'buy_month', 'buy_day',
        }
        # Every param must provide a default so the GUI can pre-fill it.
        for param in schema:
            assert param.default is not None, f"Param '{param.key}' has no default"
        assert by_key['initial_investment'].default == float(INITIAL_INVESTMENT)
        # The seasonal defaults reproduce the classic Aug -> Oct summer gap.
        assert by_key['sell_month'].default == 'August'
        assert by_key['sell_day'].default == 1
        assert by_key['buy_month'].default == 'October'
        assert by_key['buy_day'].default == 1
        # Select defaults must be members of their options (also enforced by
        # ConfigParam, asserted here as a guard against typos).
        assert by_key['sell_month'].default in by_key['sell_month'].options
        assert by_key['buy_month'].default in by_key['buy_month'].options

    def test_run_returns_exactly_9_metric_keys(self):
        strategy = SummerGapStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert portfolio is not None
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS
        assert len(metrics) == 9
        assert orders is not None and len(orders) >= 1

    def test_run_with_empty_filenames_returns_none_none(self):
        strategy = SummerGapStrategy()
        portfolio, metrics, orders = strategy.run(
            BASE_URL, [], _START, _END, SAMPLE_META, params={}
        )
        assert portfolio is None and metrics is None and orders is None

    def test_run_with_too_short_date_range_returns_none_none(self):
        # Only 1 month in the window → compute_metrics needs ≥3 → 3×None.
        strategy = SummerGapStrategy()
        start = pd.Timestamp('2020-06-30', tz='UTC')
        end = pd.Timestamp('2020-06-30', tz='UTC')
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META, params={}
            )
        assert portfolio is None and metrics is None and orders is None

    def test_flat_market_still_swaps_in_and_out(self):
        # Unlike Risk-Off (which stays all-cash in a flat market), the seasonal
        # target flips regardless of price, so a multi-year window over flat
        # prices still records Sell (start of Aug) and Buy (start of Oct) trades.
        strategy = SummerGapStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert metrics is not None
        assert orders is not None and orders
        sides = [o['side'] for o in orders]
        assert 'Sell' in sides and 'Buy' in sides
        # Two full years in the window → two summer gaps → two Sell exits.
        assert sides.count('Sell') == 2
        # Sells land in August, buy-backs in October.
        sell_months = {o['date'].month for o in orders if o['side'] == 'Sell'}
        buy_back_months = {o['date'].month for o in orders if o['side'] == 'Buy'}
        assert sell_months == {8}
        assert buy_back_months <= {1, 10}  # initial deployment (Jan) + Oct re-entries

    def test_custom_initial_investment_scales_invested(self):
        strategy = SummerGapStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            _, metrics, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META,
                params={'initial_investment': 25_000.0},
            )
        assert metrics is not None
        invested = float(metrics['Invested'].replace(',', ''))
        assert invested == pytest.approx(25_000.0)


# ---------------------------------------------------------------------------
# Layer 8 – Summer Gap pure functions (no Dash / no I/O)
# ---------------------------------------------------------------------------

class TestSeasonalTarget:
    # A full year of daily dates to test month-level membership.
    _YEAR = pd.date_range('2021-01-01', '2021-12-31', freq='D', tz='UTC')

    def test_default_window_is_out_in_august_and_september(self):
        # Default Aug 1 -> Oct 1: cash through August and September, invested else.
        target = _seasonal_target(self._YEAR, 8, 1, 10, 1)
        out_months = {d.month for d, v in target.items() if v == 0.0}
        in_months = {d.month for d, v in target.items() if v == 1.0}
        assert out_months == {8, 9}
        assert in_months == {1, 2, 3, 4, 5, 6, 7, 10, 11, 12}

    def test_boundaries_are_half_open(self):
        # Sell date is "out", buy date is back "in" (half-open [sell, buy)).
        target = _seasonal_target(self._YEAR, 8, 1, 10, 1)
        aug_1 = pd.Timestamp('2021-08-01', tz='UTC')
        sep_30 = pd.Timestamp('2021-09-30', tz='UTC')
        oct_1 = pd.Timestamp('2021-10-01', tz='UTC')
        jul_31 = pd.Timestamp('2021-07-31', tz='UTC')
        assert target[aug_1] == 0.0
        assert target[sep_30] == 0.0
        assert target[oct_1] == 1.0
        assert target[jul_31] == 1.0

    def test_day_granularity_within_a_month(self):
        # Sell mid-August (15th), buy mid-September (15th): only that half-month
        # is out of market.
        target = _seasonal_target(self._YEAR, 8, 15, 9, 15)
        assert target[pd.Timestamp('2021-08-14', tz='UTC')] == 1.0
        assert target[pd.Timestamp('2021-08-15', tz='UTC')] == 0.0
        assert target[pd.Timestamp('2021-09-14', tz='UTC')] == 0.0
        assert target[pd.Timestamp('2021-09-15', tz='UTC')] == 1.0

    def test_window_wrapping_year_end(self):
        # Sell in November, buy back in February → Nov, Dec and Jan are out.
        target = _seasonal_target(self._YEAR, 11, 1, 2, 1)
        out_months = {d.month for d, v in target.items() if v == 0.0}
        assert out_months == {11, 12, 1}

    def test_same_date_stays_fully_invested(self):
        # Degenerate configuration (sell date == buy date) → no out-of-market
        # window, fully invested every day.
        target = _seasonal_target(self._YEAR, 8, 1, 8, 1)
        assert (target == 1.0).all()


# ---------------------------------------------------------------------------
# Layer 9 – simulate_rotation / _rotation_order_events (generic engine)
# ---------------------------------------------------------------------------

class TestSimulateRotation:
    _IDX = pd.date_range('2020-01-01', periods=6, freq='D', tz='UTC')

    def test_empty_price_df_returns_flat_lump_sum(self):
        portfolio, invested = simulate_rotation(pd.DataFrame(), [], 10_000.0)
        assert portfolio.empty
        assert invested == 10_000.0

    def test_no_events_holds_flat_cash(self):
        price = pd.DataFrame({'A': [100.0] * 6}, index=self._IDX)
        portfolio, invested = simulate_rotation(price, [], 10_000.0)
        assert (portfolio == 10_000.0).all()
        assert invested == 10_000.0

    def test_events_outside_index_are_ignored(self):
        price = pd.DataFrame({'A': [100.0] * 6}, index=self._IDX)
        bogus_date = pd.Timestamp('2019-01-01', tz='UTC')
        portfolio, invested = simulate_rotation(price, [(bogus_date, 1.0, {'A': 1.0})], 10_000.0)
        assert (portfolio == 10_000.0).all()
        assert invested == 10_000.0

    def test_rotation_between_two_assets(self):
        # Fully into A on day 0; sold to cash on day 2 (A has doubled by then);
        # rotated fully into B on day 3.
        price = pd.DataFrame({
            'A': [100.0, 100.0, 200.0, 200.0, 200.0, 200.0],
            'B': [50.0, 50.0, 50.0, 50.0, 100.0, 100.0],
        }, index=self._IDX)
        events = [
            (self._IDX[0], 1.0, {'A': 1.0}),
            (self._IDX[2], 0.0, {'A': 1.0}),
            (self._IDX[3], 1.0, {'B': 1.0}),
        ]
        portfolio, invested = simulate_rotation(price, events, 10_000.0)
        assert portfolio.iloc[0] == pytest.approx(10_000.0)
        assert portfolio.iloc[1] == pytest.approx(10_000.0)
        # Sold on day 2 at A=200: the position had doubled before the sale.
        assert portfolio.iloc[2] == pytest.approx(20_000.0)
        # Rotated fully into B on day 3 (still worth 20,000 right after the trade).
        assert portfolio.iloc[3] == pytest.approx(20_000.0)
        # B then doubles (50 -> 100) by day 4, so the position doubles too.
        assert portfolio.iloc[4] == pytest.approx(40_000.0)
        assert portfolio.iloc[5] == pytest.approx(40_000.0)
        assert invested == 10_000.0


class TestRotationOrderEvents:
    _IDX = pd.date_range('2020-01-01', periods=4, freq='D', tz='UTC')

    def test_empty_frame_or_no_events_returns_no_events(self):
        assert _rotation_order_events(pd.DataFrame(), [], 10_000.0) == []
        price = pd.DataFrame({'A': [100.0] * 4}, index=self._IDX)
        assert _rotation_order_events(price, [], 10_000.0) == []

    def test_buy_and_sell_sides(self):
        price = pd.DataFrame({'A': [100.0, 100.0, 200.0, 200.0]}, index=self._IDX)
        events = [
            (self._IDX[0], 1.0, {'A': 1.0}),
            (self._IDX[2], 0.0, {'A': 1.0}),
        ]
        out = _rotation_order_events(price, events, 10_000.0)
        assert len(out) == 2
        buy, sell = out
        assert buy['side'] == 'Buy'
        assert sell['side'] == 'Sell'
        assert buy['inflow'] == 0.0 and sell['inflow'] == 0.0
        assert buy['asset_values'] == pytest.approx({'A': 10_000.0})
        assert sell['asset_values'] == pytest.approx({'A': 0.0})
        assert buy['assets_after'] + buy['cash_after'] == pytest.approx(10_000.0)
        assert sell['assets_after'] + sell['cash_after'] == pytest.approx(20_000.0)

    def test_events_outside_index_are_skipped(self):
        price = pd.DataFrame({'A': [100.0] * 4}, index=self._IDX)
        bogus = pd.Timestamp('2019-01-01', tz='UTC')
        assert _rotation_order_events(price, [(bogus, 1.0, {'A': 1.0})], 10_000.0) == []

    def test_fx_context_adds_local_price_and_rate(self):
        price = pd.DataFrame({'AAPL': [100.0, 100.0]}, index=self._IDX[:2])
        rate = pd.Series([0.9, 0.9], index=self._IDX[:2])
        fx = FxColumns(
            asset_local_ccy={'AAPL': 'USD'},
            asset_rate={'AAPL': rate},
            pair_rate={'USDEUR=X': rate},
        )
        (ev,) = _rotation_order_events(price, [(self._IDX[0], 1.0, {'AAPL': 1.0})], 10_000.0, fx)
        assert ev['asset_prices_local']['AAPL'] == pytest.approx(100.0 / 0.9)
        assert ev['fx_rates']['USDEUR=X'] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Layer 10 – Loser Rotation strategy plugin
# ---------------------------------------------------------------------------

class TestLoserRotationPureFunctions:
    def test_first_trading_day_of_year_present_and_absent(self):
        idx = pd.date_range('2020-01-03', '2020-12-31', freq='D', tz='UTC')
        assert _first_trading_day_of_year(idx, 2020) == idx[0]
        assert _first_trading_day_of_year(idx, 2021) is None

    def test_first_trading_day_on_or_after(self):
        idx = pd.date_range('2020-01-01', '2020-12-31', freq='D', tz='UTC')
        assert _first_trading_day_on_or_after(idx, 2020, 7, 1) == pd.Timestamp('2020-07-01', tz='UTC')
        # No year rows at all.
        assert _first_trading_day_on_or_after(idx, 2021, 7, 1) is None

    def test_first_trading_day_on_or_after_none_when_date_past_history(self):
        # History for the year ends before the requested (month, day).
        idx = pd.date_range('2020-01-01', '2020-06-30', freq='D', tz='UTC')
        assert _first_trading_day_on_or_after(idx, 2020, 7, 1) is None


class TestSelectLosers:
    _IDX = pd.to_datetime(['2020-01-02', '2020-07-01']).tz_localize('UTC')

    def test_ranks_by_ytd_return_ascending(self):
        daily_df = pd.DataFrame({
            'A': [100.0, 150.0],  # +50%
            'B': [100.0, 80.0],   # -20% (worst)
            'C': [100.0, 90.0],   # -10%
        }, index=self._IDX)
        assert _select_losers(daily_df, self._IDX[0], self._IDX[1], n_losers=2) == ['B', 'C']

    def test_clamps_to_eligible_count(self):
        daily_df = pd.DataFrame({'A': [100.0, 90.0]}, index=self._IDX)
        assert _select_losers(daily_df, self._IDX[0], self._IDX[1], n_losers=5) == ['A']

    def test_excludes_assets_missing_either_price(self):
        daily_df = pd.DataFrame({
            'A': [100.0, 90.0],
            'B': [np.nan, 80.0],  # missing at the year-start anchor
        }, index=self._IDX)
        assert _select_losers(daily_df, self._IDX[0], self._IDX[1], n_losers=2) == ['A']

    def test_returns_empty_when_none_eligible(self):
        daily_df = pd.DataFrame({'A': [np.nan, 90.0]}, index=self._IDX)
        assert _select_losers(daily_df, self._IDX[0], self._IDX[1], n_losers=2) == []


class TestBuildRotationEvents:
    def test_builds_annual_buy_and_sell_events(self):
        idx = pd.date_range('2020-01-01', '2021-12-31', freq='D', tz='UTC')
        daily_df = pd.DataFrame({
            'A': np.linspace(100, 50, len(idx)),   # steadily falling -> always the loser
            'B': np.linspace(100, 200, len(idx)),  # steadily rising -> always the winner
        }, index=idx)
        events = _build_rotation_events(daily_df, daily_df, 7, 1, 10, 1, n_losers=1, weights=None)
        dates = [e[0] for e in events]
        assert dates == sorted(dates)
        buys = [e for e in events if e[1] == 1.0]
        sells = [e for e in events if e[1] == 0.0]
        assert len(buys) == 2 and len(sells) == 2
        assert all(e[0].month == 7 and e[0].day == 1 for e in buys)
        assert all(e[0].month == 10 and e[0].day == 1 for e in sells)
        # A is always the (single) loser, so it's the only symbol ever weighted.
        assert all(e[2] == {'A': 1.0} for e in buys + sells)

    def test_wraparound_sell_date_lands_in_next_year(self):
        idx = pd.date_range('2020-01-01', '2021-12-31', freq='D', tz='UTC')
        daily_df = pd.DataFrame({
            'A': 100.0,
            'B': np.linspace(100, 300, len(idx)),
        }, index=idx)
        # Buy in November, sell in February -> the sell falls in the following year.
        events = _build_rotation_events(daily_df, daily_df, 11, 1, 2, 1, n_losers=1, weights=None)
        buys = sorted(e[0] for e in events if e[1] == 1.0)
        sells = sorted(e[0] for e in events if e[1] == 0.0)
        assert buys and all(d.month == 11 for d in buys)
        assert sells and all(d.month == 2 for d in sells)
        assert sells[0].year == buys[0].year + 1

    def test_events_outside_window_are_dropped(self):
        idx = pd.date_range('2020-01-01', '2021-12-31', freq='D', tz='UTC')
        daily_df = pd.DataFrame({
            'A': 100.0,
            'B': np.linspace(100, 300, len(idx)),
        }, index=idx)
        # The windowed price_df only covers the second year.
        price_df = daily_df.loc['2021-01-01':]
        events = _build_rotation_events(daily_df, price_df, 7, 1, 10, 1, n_losers=1, weights=None)
        assert events
        assert all(e[0].year == 2021 for e in events)

    def test_weights_restrict_to_selected_losers_only(self):
        idx = pd.date_range('2020-01-01', '2020-12-31', freq='D', tz='UTC')
        daily_df = pd.DataFrame({
            'A': np.linspace(100, 50, len(idx)),
            'B': np.linspace(100, 200, len(idx)),
        }, index=idx)
        events = _build_rotation_events(
            daily_df, daily_df, 7, 1, 10, 1, n_losers=1, weights={'A': 2.0, 'B': 5.0}
        )
        buy = next(e for e in events if e[1] == 1.0)
        # Only the selected loser (A) appears, keeping its own relative weight.
        assert buy[2] == {'A': 2.0}

    def test_no_eligible_losers_contributes_no_events(self):
        # A single-day frame has no distinct year-start anchor to compare against
        # (year_start == buy_date, so the "return" is always 0 for every asset —
        # still eligible/rankable, this instead checks the truly-empty-frame path).
        assert _build_rotation_events(pd.DataFrame(), pd.DataFrame(), 7, 1, 10, 1, 1, None) == []

    def test_calendar_mismatch_does_not_wrongly_exclude_an_asset(self):
        # Asset A ("loser", steadily falling) misses its own market's first
        # trading day of 2021 (e.g. a market-specific holiday); Asset B
        # ("winner", steadily rising) trades that day, so the *combined*
        # calendar's "first trading day of 2021" is a day A itself didn't
        # trade. Without bridging that single-day gap, A would be wrongly
        # excluded from the 2021 ranking entirely, and B (the actual winner)
        # would be picked instead purely because of the calendar mismatch —
        # exactly the bug seen with a real US-listed + Xetra-listed basket.
        idx = pd.date_range('2020-01-01', '2021-12-31', freq='D', tz='UTC')
        gap_date = pd.Timestamp('2021-01-01', tz='UTC')
        a_idx = idx[idx != gap_date]
        series_a = pd.Series(np.linspace(100, 50, len(a_idx)), index=a_idx)
        series_b = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        daily_df = pd.DataFrame({'A': series_a, 'B': series_b})

        events = _build_rotation_events(daily_df, daily_df, 7, 1, 10, 1, n_losers=1, weights=None)
        buys_2021 = [e for e in events if e[1] == 1.0 and e[0].year == 2021]
        assert buys_2021
        assert buys_2021[0][2] == {'A': 1.0}

    def test_four_assets_all_priced_every_year_rotates_every_year(self):
        # Mirrors the real-world scenario (4 continuously-priced assets,
        # n_losers=2): every year in the window must contribute a buy/sell
        # pair selecting exactly the 2 worst performers — no year skipped.
        idx = pd.date_range('2018-01-01', '2021-12-31', freq='D', tz='UTC')
        n = len(idx)
        daily_df = pd.DataFrame({
            'A': np.linspace(100, 20, n),   # worst performer
            'B': np.linspace(100, 30, n),   # second-worst performer
            'C': np.linspace(100, 300, n),  # winner
            'D': np.linspace(100, 400, n),  # winner
        }, index=idx)
        events = _build_rotation_events(daily_df, daily_df, 7, 1, 10, 1, n_losers=2, weights=None)
        years = sorted({e[0].year for e in events})
        assert years == [2018, 2019, 2020, 2021]
        for year in years:
            buy = next(e for e in events if e[1] == 1.0 and e[0].year == year)
            assert set(buy[2]) == {'A', 'B'}


class TestLoserRotationStrategy:
    def test_is_registered_in_registry(self):
        assert "Loser Rotation" in list_strategies()
        assert get_strategy("Loser Rotation") is LoserRotationStrategy

    def test_get_name_and_icon(self):
        assert LoserRotationStrategy.get_name() == 'Loser Rotation'
        icon = LoserRotationStrategy.get_icon()
        assert isinstance(icon, str) and icon

    def test_get_description_is_non_empty_string(self):
        assert isinstance(LoserRotationStrategy.get_description(), str)
        assert LoserRotationStrategy.get_description()

    def test_get_long_description_returns_rich_markdown(self):
        long = LoserRotationStrategy.get_long_description()
        assert isinstance(long, str) and long
        assert len(long) > len(LoserRotationStrategy.get_description())

    def test_config_schema_keys_and_defaults(self):
        schema = LoserRotationStrategy.get_config_schema()
        by_key = {p.key: p for p in schema}
        assert set(by_key) == {
            'initial_investment', 'n_losers', 'buy_month', 'buy_day', 'sell_month', 'sell_day',
        }
        for param in schema:
            assert param.default is not None, f"Param '{param.key}' has no default"
        assert by_key['initial_investment'].default == float(INITIAL_INVESTMENT)
        assert by_key['n_losers'].default == 3
        assert by_key['buy_month'].default == 'July'
        assert by_key['buy_day'].default == 1
        assert by_key['sell_month'].default == 'October'
        assert by_key['sell_day'].default == 1
        assert by_key['buy_month'].default in by_key['buy_month'].options
        assert by_key['sell_month'].default in by_key['sell_month'].options

    def test_run_returns_exactly_9_metric_keys(self):
        strategy = LoserRotationStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert portfolio is not None
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS
        assert len(metrics) == 9
        assert orders is not None and len(orders) >= 1

    def test_run_with_empty_filenames_returns_none_none(self):
        strategy = LoserRotationStrategy()
        portfolio, metrics, orders = strategy.run(
            BASE_URL, [], _START, _END, SAMPLE_META, params={}
        )
        assert portfolio is None and metrics is None and orders is None

    def test_run_with_too_short_date_range_returns_none_none(self):
        strategy = LoserRotationStrategy()
        start = pd.Timestamp('2020-06-30', tz='UTC')
        end = pd.Timestamp('2020-06-30', tz='UTC')
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META, params={}
            )
        assert portfolio is None and metrics is None and orders is None

    def test_custom_initial_investment_scales_invested(self):
        strategy = LoserRotationStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            _, metrics, _ = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META,
                params={'initial_investment': 25_000.0},
            )
        assert metrics is not None
        invested = float(metrics['Invested'].replace(',', ''))
        assert invested == pytest.approx(25_000.0)

    def test_n_losers_larger_than_basket_still_runs(self):
        # Single-asset basket with n_losers=10 (clamped down to the 1 eligible asset).
        strategy = LoserRotationStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_rising()):
            portfolio, metrics, orders = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META,
                params={'n_losers': 10},
            )
        assert portfolio is not None and metrics is not None
        assert orders is not None and orders


class TestLoserRotationSelection:
    """Full run() test: the yearly rotation buys into the correct worst YTD performer."""

    META = pd.DataFrame({
        'asset_class': ['stocks'] * 3,
        'symbol': ['WIN', 'FLAT', 'LOSE'],
        'name': ['Winner', 'Flat', 'Loser'],
        'filename': ['win.parquet', 'flat.parquet', 'lose.parquet'],
    })

    def _three_asset_read(self):
        winner = _daily_rising(low=100.0, high=400.0)
        flat = _daily_ohlcv(100.0)
        loser = _daily_rising(low=100.0, high=20.0)

        def _read(path, columns=None):
            if 'win' in path:
                return winner
            if 'lose' in path:
                return loser
            return flat
        return _read

    def test_rotation_buys_the_worst_ytd_performer(self):
        strategy = LoserRotationStrategy()
        with patch('src.backtest.pd.read_parquet', side_effect=self._three_asset_read()):
            _, metrics, orders = strategy.run(
                BASE_URL, ['win.parquet', 'flat.parquet', 'lose.parquet'],
                _START, _END, self.META, params={'n_losers': 1},
            )
        assert metrics is not None
        assert orders is not None and orders
        buys = [o for o in orders if o['side'] == 'Buy']
        assert buys
        for buy in buys:
            invested_syms = {sym for sym, val in buy['asset_values'].items() if val > 0}
            assert invested_syms == {'LOSE'}
