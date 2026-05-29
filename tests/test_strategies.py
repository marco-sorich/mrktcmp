import os
import pytest
import pandas as pd
from unittest.mock import patch

os.environ.pop("BASE_URL", None)

import src.strategies.registry as _registry_module  # noqa: E402
from src.strategies.base import BacktestStrategy, ConfigParam  # noqa: E402
from src.strategies.registry import get_strategy, list_strategies, register  # noqa: E402
from src.strategies.dca import DCAStrategy  # noqa: E402
from src.backtest import run_backtest  # noqa: E402

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
    'Profit/Loss', 'Best Month', 'Worst Month',
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
            portfolio, metrics = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert portfolio is not None
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS

    def test_run_returns_11_metric_keys(self):
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, metrics = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
        assert metrics is not None
        assert len(metrics) == 11

    def test_run_with_empty_filenames_returns_none_none(self):
        strategy = DCAStrategy()
        portfolio, metrics = strategy.run(
            BASE_URL, [], _START, _END, SAMPLE_META, params={}
        )
        assert portfolio is None
        assert metrics is None

    def test_run_with_too_short_date_range_returns_none_none(self):
        # Only 1 month of data → compute_metrics needs ≥3 → run() returns (None, None).
        strategy = DCAStrategy()
        start = pd.Timestamp('2020-06-30', tz='UTC')
        end = pd.Timestamp('2020-06-30', tz='UTC')
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            portfolio, metrics = strategy.run(
                BASE_URL, ['aapl.parquet'], start, end, SAMPLE_META, params={}
            )
        assert portfolio is None
        assert metrics is None

    def test_custom_monthly_investment_halves_invested_total(self):
        # At the same constant price, halving the monthly investment halves the
        # invested total.  Use that relationship to verify params are respected.
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, m_default = strategy.run(
                BASE_URL, ['aapl.parquet'], _START, _END, SAMPLE_META, params={}
            )
            _, m_half = strategy.run(
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
            p, m = run_backtest(
                BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META
            )
        assert p is not None
        assert isinstance(m, dict)
        assert set(m.keys()) == _EXPECTED_METRIC_KEYS

    def test_with_strategy_plugin_returns_correct_structure(self):
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            p, m = run_backtest(
                BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META,
                strategy=strategy,
                strategy_params={'monthly_investment': 500.0},
            )
        assert p is not None
        assert isinstance(m, dict)
        assert set(m.keys()) == _EXPECTED_METRIC_KEYS

    def test_strategy_params_honoured_in_run_backtest(self):
        strategy = DCAStrategy()
        with patch('src.backtest.pd.read_parquet', return_value=_daily_ohlcv(100.0)):
            _, m_default = run_backtest(
                BASE_URL, ['aapl.parquet'], self._START, self._END, SAMPLE_META,
                strategy=strategy,
                strategy_params={},
            )
            _, m_half = run_backtest(
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
        p, m = run_backtest(
            BASE_URL, [], self._START, self._END, SAMPLE_META,
            strategy=strategy,
        )
        assert p is None and m is None
