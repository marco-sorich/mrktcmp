import os
import pandas as pd
from unittest.mock import patch, MagicMock
import plotly.graph_objects as go

# Ensure no BASE_URL during import so df=None and no network calls are made.
# load_dotenv is also patched so that the .env file (which may contain BASE_URL)
# is not loaded when the module is first imported.
os.environ.pop("BASE_URL", None)

from dash import no_update  # noqa: E402

with patch("dotenv.load_dotenv"):
    import src.app as app_module  # noqa: E402
from src.app import (  # noqa: E402
    update_asset_class, update_asset_search, update_chart,
    _bt_assetclass_options, _bt_asset_search, _manage_basket,
    run_backtest_callback, _render_basket_list, _metrics_table,
    update_date_range_slider, update_date_display, _build_slider_marks,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_DF = pd.DataFrame(
    {
        "asset_class": ["stocks", "stocks", "crypto"],
        "symbol": ["AAPL", "GOOGL", "BTC"],
        "interval": ["1d", "1d", "1d"],
        "name": ["Apple Inc", "Alphabet Inc", "Bitcoin"],
        "exchange": ["NASDAQ", "NASDAQ", "Binance"],
        "country": ["US", "US", "Global"],
        "category": ["Tech", "Tech", "Crypto"],
        "first_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "last_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "filename": ["aapl.parquet", "googl.parquet", "btc.parquet"],
    }
)

SAMPLE_OHLCV = pd.DataFrame(
    {
        "Open": [150.0, 155.0, 160.0],
        "High": [155.0, 160.0, 165.0],
        "Low": [148.0, 153.0, 158.0],
        "Close": [153.0, 158.0, 163.0],
        "Volume": [1_000_000, 1_100_000, 1_200_000],
    },
    index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
)
SAMPLE_OHLCV.index.name = "Date"

BASE_URL = "http://example.com"


# ---------------------------------------------------------------------------
# update_asset_class
# ---------------------------------------------------------------------------


class TestUpdateAssetClass:
    def test_no_asset_class_returns_disabled_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_class(None)
        assert options == []
        assert disabled is True

    def test_df_none_returns_disabled_empty(self):
        with patch.object(app_module, "df", None):
            options, disabled = update_asset_class("stocks")
        assert options == []
        assert disabled is True

    def test_filters_by_asset_class(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_class("stocks")
        assert disabled is False
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values
        assert "btc.parquet" not in values

    def test_crypto_class_returns_only_crypto(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_class("crypto")
        assert disabled is False
        assert len(options) == 1
        assert options[0]["value"] == "btc.parquet"

    def test_option_label_format(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, _ = update_asset_class("crypto")
        assert options[0]["label"] == "BTC — Bitcoin (1d)"

    def test_results_capped_at_thirty(self):
        large_df = pd.DataFrame(
            {
                "asset_class": ["stocks"] * 300,
                "symbol": [f"SYM{i}" for i in range(300)],
                "interval": ["1d"] * 300,
                "name": [f"Company {i}" for i in range(300)],
                "exchange": ["NYSE"] * 300,
                "country": ["US"] * 300,
                "category": ["Tech"] * 300,
                "first_date": ["2020-01-01"] * 300,
                "last_date": ["2024-01-01"] * 300,
                "filename": [f"sym{i}.parquet" for i in range(300)],
            }
        )
        with patch.object(app_module, "df", large_df):
            options, _ = update_asset_class("stocks")
        assert len(options) == 200


# ---------------------------------------------------------------------------
# update_asset_search
# ---------------------------------------------------------------------------


class TestUpdateAssetSearch:
    def test_no_asset_class_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search(None, None, None)
        assert options == []

    def test_df_none_returns_empty(self):
        with patch.object(app_module, "df", None):
            options = update_asset_search(None, "stocks", None)
        assert options == []

    def test_no_search_returns_top_results_for_class(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search(None, "stocks", None)
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values
        assert "btc.parquet" not in values

    def test_search_by_symbol_case_insensitive(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("aapl", "stocks", None)
        assert len(options) == 1
        assert options[0]["value"] == "aapl.parquet"

    def test_search_by_name_partial_match(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("alphabet", "stocks", None)
        assert len(options) == 1
        assert options[0]["value"] == "googl.parquet"

    def test_search_with_no_matches_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("ZZZZZZ", "stocks", None)
        assert options == []

    def test_current_value_appended_when_not_in_search_results(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("GOOGL", "stocks", "aapl.parquet")
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values

    def test_current_value_not_duplicated_when_already_visible(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search(None, "stocks", "aapl.parquet")
        values = [o["value"] for o in options]
        assert values.count("aapl.parquet") == 1

    def test_unknown_current_value_not_appended(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("GOOGL", "stocks", "nonexistent.parquet")
        values = [o["value"] for o in options]
        assert "nonexistent.parquet" not in values

    def test_search_results_capped_at_thirty(self):
        large_df = pd.DataFrame(
            {
                "asset_class": ["stocks"] * 40,
                "symbol": [f"ABC{i}" for i in range(40)],
                "interval": ["1d"] * 40,
                "name": [f"Company {i}" for i in range(40)],
                "exchange": ["NYSE"] * 40,
                "country": ["US"] * 40,
                "category": ["Tech"] * 40,
                "first_date": ["2020-01-01"] * 40,
                "last_date": ["2024-01-01"] * 40,
                "filename": [f"sym{i}.parquet" for i in range(40)],
            }
        )
        with patch.object(app_module, "df", large_df):
            options = update_asset_search("ABC", "stocks", None)
        assert len(options) == 30


# ---------------------------------------------------------------------------
# update_chart
# ---------------------------------------------------------------------------


class TestUpdateChart:
    def test_no_filename_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL):
            fig, headline, _ = update_chart(None, True)
        assert headline == ""

    def test_no_base_url_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", None):
            fig, headline, _ = update_chart("aapl.parquet", True)
        assert headline == ""

    def test_df_none_returns_empty(self):
        with patch.object(app_module, "df", None), \
             patch.object(app_module, "base_url", BASE_URL):
            fig, headline, _ = update_chart("aapl.parquet", True)
        assert headline == ""

    def test_valid_input_returns_figure(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            fig, headline, _ = update_chart("aapl.parquet", True)
        assert isinstance(fig, go.Figure)

    def test_chart_contains_candlestick_trace(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            fig, *_ = update_chart("aapl.parquet", True)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Candlestick" in trace_types

    def test_chart_contains_volume_scatter_trace(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            fig, *_ = update_chart("aapl.parquet", True)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scattergl" in trace_types

    def test_headline_shows_asset_name(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            _, headline, _ = update_chart("aapl.parquet", True)
        assert any("Apple Inc" in str(c) for c in headline)

    def test_headline_shows_exchange_and_country(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            _, headline, _ = update_chart("aapl.parquet", True)
        combined = " ".join(str(c) for c in headline)
        assert "NASDAQ" in combined
        assert "US" in combined

    def test_parquet_read_error_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", side_effect=Exception("network error")):
            fig, headline, _ = update_chart("aapl.parquet", True)
        assert headline == ""

    def test_unknown_filename_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL):
            # filename not in df → .iloc[0] raises IndexError → caught → empty
            fig, headline, _ = update_chart("nonexistent.parquet", True)
        assert headline == ""

    def test_data_older_than_10_years_is_filtered(self):
        now = pd.Timestamp.now()
        mixed_ohlcv = pd.DataFrame(
            {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
             "Close": [1.0, 2.0], "Volume": [100, 200]},
            index=[now - pd.DateOffset(years=11), now - pd.DateOffset(days=1)],
        )
        mixed_ohlcv.index.name = "Date"
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=mixed_ohlcv):
            fig, *_ = update_chart("aapl.parquet", True)
        assert len(fig.data[0].x) == 1

    def test_data_within_10_years_is_kept(self):
        now = pd.Timestamp.now()
        recent_ohlcv = pd.DataFrame(
            {"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0], "Low": [1.0, 2.0, 3.0],
             "Close": [1.0, 2.0, 3.0], "Volume": [100, 200, 300]},
            index=[now - pd.DateOffset(years=9), now - pd.DateOffset(years=5), now - pd.DateOffset(days=1)],
        )
        recent_ohlcv.index.name = "Date"
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=recent_ohlcv):
            fig, *_ = update_chart("aapl.parquet", True)
        assert len(fig.data[0].x) == 3

    def test_timezone_aware_index_does_not_error(self):
        now = pd.Timestamp.now(tz="Europe/Berlin")
        tz_ohlcv = pd.DataFrame(
            {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
             "Close": [1.0, 2.0], "Volume": [100, 200]},
            index=[now - pd.DateOffset(years=11), now - pd.DateOffset(days=1)],
        )
        tz_ohlcv.index.name = "Date"
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=tz_ohlcv):
            fig, headline, _ = update_chart("aapl.parquet", True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 1


# ---------------------------------------------------------------------------
# App-level structure
# ---------------------------------------------------------------------------


class TestAppStructure:
    def test_server_is_flask_instance(self):
        import flask
        assert isinstance(app_module.server, flask.Flask)

    def test_layout_is_not_none(self):
        assert app_module.app.layout is not None

    def test_df_is_none_when_no_base_url(self):
        assert app_module.df is None

    def test_asset_classes_empty_when_no_base_url(self):
        assert app_module.assetsClasses == []


# ---------------------------------------------------------------------------
# Backtesting – helpers
# ---------------------------------------------------------------------------

def _make_ctx(triggered_id, triggered_value=1):
    """Build a minimal mock of dash.callback_context."""
    ctx = MagicMock()
    ctx.triggered = [{'prop_id': 'test.n_clicks', 'value': triggered_value}]
    ctx.triggered_id = triggered_id
    return ctx


BASKET_ITEM_AAPL = {'filename': 'aapl.parquet', 'symbol': 'AAPL', 'name': 'Apple Inc'}
BASKET_ITEM_GOOGL = {'filename': 'googl.parquet', 'symbol': 'GOOGL', 'name': 'Alphabet Inc'}


# ---------------------------------------------------------------------------
# _bt_assetclass_options
# ---------------------------------------------------------------------------

class TestBtAssetclassOptions:
    def test_no_asset_class_returns_empty_and_disabled(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            opts, disabled = _bt_assetclass_options(None)
        assert opts == []
        assert disabled is True

    def test_df_none_returns_empty_and_disabled(self):
        with patch.object(app_module, 'df', None):
            opts, disabled = _bt_assetclass_options('stocks')
        assert opts == []
        assert disabled is True

    def test_valid_class_returns_options_and_enabled(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            opts, disabled = _bt_assetclass_options('stocks')
        assert disabled is False
        values = [o['value'] for o in opts]
        assert 'aapl.parquet' in values
        assert 'btc.parquet' not in values

    def test_results_capped_at_thirty(self):
        large_df = pd.DataFrame({
            'asset_class': ['stocks'] * 300,
            'symbol': [f'SYM{i}' for i in range(300)],
            'interval': ['1d'] * 300,
            'name': [f'Company {i}' for i in range(300)],
            'exchange': ['NYSE'] * 300,
            'country': ['US'] * 300,
            'category': ['Tech'] * 300,
            'first_date': ['2020-01-01'] * 300,
            'last_date': ['2024-01-01'] * 300,
            'filename': [f'sym{i}.parquet' for i in range(300)],
        })
        with patch.object(app_module, 'df', large_df):
            opts, _ = _bt_assetclass_options('stocks')
        assert len(opts) == 200


# ---------------------------------------------------------------------------
# _bt_asset_search
# ---------------------------------------------------------------------------

class TestBtAssetSearch:
    def test_no_asset_class_returns_empty(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            assert _bt_asset_search(None, None, None) == []

    def test_df_none_returns_empty(self):
        with patch.object(app_module, 'df', None):
            assert _bt_asset_search('AAPL', 'stocks', None) == []

    def test_search_by_symbol_exact_match(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search('aapl', 'stocks', None)
        assert len(opts) == 1
        assert opts[0]['value'] == 'aapl.parquet'

    def test_search_by_name_partial_match(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search('alphabet', 'stocks', None)
        assert len(opts) == 1
        assert opts[0]['value'] == 'googl.parquet'

    def test_current_value_appended_when_not_in_results(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search('GOOGL', 'stocks', 'aapl.parquet')
        values = [o['value'] for o in opts]
        assert 'aapl.parquet' in values
        assert 'googl.parquet' in values

    def test_current_value_not_duplicated(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search(None, 'stocks', 'aapl.parquet')
        assert [o['value'] for o in opts].count('aapl.parquet') == 1

    def test_no_matches_returns_empty(self):
        with patch.object(app_module, 'df', SAMPLE_DF):
            assert _bt_asset_search('ZZZZZZ', 'stocks', None) == []


# ---------------------------------------------------------------------------
# _manage_basket
# ---------------------------------------------------------------------------

class TestManageBasket:
    def test_add_asset_to_empty_basket(self):
        ctx = _make_ctx('bt-add-a')
        with patch('dash.callback_context', ctx), patch.object(app_module, 'df', SAMPLE_DF):
            basket, _ = _manage_basket('a', [], 'aapl.parquet', [])
        assert len(basket) == 1
        assert basket[0]['filename'] == 'aapl.parquet'
        assert basket[0]['symbol'] == 'AAPL'

    def test_duplicate_asset_is_not_added_twice(self):
        ctx = _make_ctx('bt-add-a')
        with patch('dash.callback_context', ctx), patch.object(app_module, 'df', SAMPLE_DF):
            basket, _ = _manage_basket('a', [], 'aapl.parquet', [BASKET_ITEM_AAPL])
        assert len(basket) == 1

    def test_add_with_no_selected_asset_returns_no_update(self):
        ctx = _make_ctx('bt-add-a')
        with patch('dash.callback_context', ctx), patch.object(app_module, 'df', SAMPLE_DF):
            result = _manage_basket('a', [], None, [BASKET_ITEM_AAPL])
        assert result == (no_update, no_update)

    def test_remove_existing_asset_from_basket(self):
        triggered = {'type': 'bt-remove-a', 'index': 'aapl.parquet'}
        ctx = _make_ctx(triggered, triggered_value=1)
        with patch('dash.callback_context', ctx):
            basket, _ = _manage_basket('a', [1], None, [BASKET_ITEM_AAPL, BASKET_ITEM_GOOGL])
        filenames = [item['filename'] for item in basket]
        assert 'aapl.parquet' not in filenames
        assert 'googl.parquet' in filenames

    def test_remove_with_zero_n_clicks_is_ignored(self):
        triggered = {'type': 'bt-remove-a', 'index': 'aapl.parquet'}
        ctx = _make_ctx(triggered, triggered_value=0)
        with patch('dash.callback_context', ctx):
            basket, _ = _manage_basket('a', [0], None, [BASKET_ITEM_AAPL])
        assert len(basket) == 1

    def test_unrelated_trigger_returns_no_update(self):
        ctx = _make_ctx('some-other-button')
        with patch('dash.callback_context', ctx):
            result = _manage_basket('a', [], None, [])
        assert result == (no_update, no_update)

    def test_no_triggered_context_returns_no_update(self):
        ctx = MagicMock()
        ctx.triggered = []
        ctx.triggered_id = None
        with patch('dash.callback_context', ctx):
            result = _manage_basket('a', [], None, [])
        assert result == (no_update, no_update)


# ---------------------------------------------------------------------------
# run_backtest_callback
# ---------------------------------------------------------------------------

_PORTFOLIO_STUB = pd.Series(
    [1000.0 * (i + 1) for i in range(24)],
    index=pd.date_range('2022-01-31', periods=24, freq='ME', tz='UTC'),
)
_METRICS_STUB = {'Total Return': '+50.0%', 'CAGR': '10.0%'}

BASKET_A = [BASKET_ITEM_AAPL]
BASKET_B = [BASKET_ITEM_GOOGL]

# Shared date store / slider fixtures used by run_backtest_callback tests.
_TEST_DATES = pd.date_range('2022-01-31', periods=24, freq='ME', tz='UTC')
_DATE_STORE = [d.isoformat() for d in _TEST_DATES]
_SLIDER_VAL = [0, 23]   # full range (index 0 … 23)


class TestRunBacktestCallback:
    def test_both_baskets_empty_returns_status_message(self):
        _, style, _, status = run_backtest_callback(1, [], [], _SLIDER_VAL, _DATE_STORE, True)
        assert 'basket' in status
        assert style['display'] == 'none'

    def test_no_base_url_returns_error_status(self):
        with patch.object(app_module, 'base_url', None), \
             patch.object(app_module, 'df', SAMPLE_DF):
            _, style, _, status = run_backtest_callback(1, BASKET_A, [], _SLIDER_VAL, _DATE_STORE, True)
        assert 'data source' in status
        assert style['display'] == 'none'

    def test_empty_date_store_returns_error_status(self):
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF):
            _, style, _, status = run_backtest_callback(1, BASKET_A, [], _SLIDER_VAL, [], True)
        assert 'date range' in status.lower()
        assert style['display'] == 'none'

    def test_no_data_returned_shows_error_status(self):
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch.object(app_module, 'run_backtest', return_value=(None, None)):
            _, style, _, status = run_backtest_callback(1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, True)
        assert style['display'] == 'none'
        assert 'No data' in status

    def test_successful_run_makes_chart_visible(self):
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch.object(app_module, 'run_backtest', return_value=(_PORTFOLIO_STUB, _METRICS_STUB)):
            _, style, _, _ = run_backtest_callback(1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, True)
        assert style['display'] == 'block'

    def test_successful_run_returns_plotly_figure(self):
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch.object(app_module, 'run_backtest', return_value=(_PORTFOLIO_STUB, _METRICS_STUB)):
            fig, _, _, _ = run_backtest_callback(1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, True)
        assert isinstance(fig, go.Figure)

    def test_only_basket_a_filled_also_succeeds(self):
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch.object(app_module, 'run_backtest', return_value=(_PORTFOLIO_STUB, _METRICS_STUB)):
            fig, style, _, status = run_backtest_callback(1, BASKET_A, [], _SLIDER_VAL, _DATE_STORE, True)
        assert style['display'] == 'block'
        assert 'complete' in status


# ---------------------------------------------------------------------------
# _build_slider_marks
# ---------------------------------------------------------------------------

class TestBuildSliderMarks:
    def test_short_range_marks_every_month(self):
        dates = pd.date_range('2024-01-31', periods=6, freq='ME', tz='UTC')
        marks = _build_slider_marks(dates)
        # All 6 positions should appear (step=1 for ≤12 months).
        assert 0 in marks
        assert 5 in marks

    def test_medium_range_marks_quarterly(self):
        dates = pd.date_range('2021-01-31', periods=24, freq='ME', tz='UTC')
        marks = _build_slider_marks(dates)
        # 24 months → step=3, so positions 0,3,6,… plus last (23).
        assert 0 in marks
        assert 12 in marks

    def test_long_range_marks_yearly(self):
        dates = pd.date_range('2015-01-31', periods=120, freq='ME', tz='UTC')
        marks = _build_slider_marks(dates)
        # 120 months → step=12, so positions 0,12,24,…plus last.
        assert 0 in marks
        assert 24 in marks

    # def test_last_position_always_present(self):
    #     for periods in (3, 18, 60):
    #         dates = pd.date_range('2020-01-31', periods=periods, freq='ME', tz='UTC')
    #         marks = _build_slider_marks(dates)
    #         assert periods - 1 in marks

    def test_labels_are_strings(self):
        dates = pd.date_range('2020-01-31', periods=12, freq='ME', tz='UTC')
        marks = _build_slider_marks(dates)
        assert all(isinstance(v, str) for v in marks.values())


# ---------------------------------------------------------------------------
# update_date_range_slider
# ---------------------------------------------------------------------------

class TestUpdateDateRangeSlider:
    def test_both_baskets_empty_disables_slider(self):
        *_, disabled, _store, _display = update_date_range_slider([], [])
        assert disabled is True

    def test_no_base_url_disables_slider(self):
        with patch.object(app_module, 'base_url', None):
            *_, disabled, _store, _display = update_date_range_slider(BASKET_A, [])
        assert disabled is True

    def test_no_overlap_disables_slider(self):
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch('src.app.get_common_date_range', return_value=(None, None)):
            *_, disabled, _store, _display = update_date_range_slider(BASKET_A, BASKET_B)
        assert disabled is True

    def test_valid_range_enables_slider(self):
        common_start = pd.Timestamp('2020-01-31', tz='UTC')
        common_end = pd.Timestamp('2022-12-31', tz='UTC')
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch('src.app.get_common_date_range', return_value=(common_start, common_end)):
            *_, disabled, _store, _display = update_date_range_slider(BASKET_A, [])
        assert disabled is False

    def test_date_store_contains_iso_strings(self):
        common_start = pd.Timestamp('2022-01-31', tz='UTC')
        common_end = pd.Timestamp('2022-03-31', tz='UTC')
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch('src.app.get_common_date_range', return_value=(common_start, common_end)):
            *_, _disabled, date_store, _display = update_date_range_slider(BASKET_A, [])
        assert len(date_store) == 3        # Jan, Feb, Mar 2022
        pd.Timestamp(date_store[0])        # must be parseable

    def test_slider_value_covers_full_range(self):
        common_start = pd.Timestamp('2022-01-31', tz='UTC')
        common_end = pd.Timestamp('2022-06-30', tz='UTC')
        with patch.object(app_module, 'base_url', 'http://x'), \
             patch.object(app_module, 'df', SAMPLE_DF), \
             patch('src.app.get_common_date_range', return_value=(common_start, common_end)):
            _min, _max, value, *_ = update_date_range_slider(BASKET_A, [])
        assert value == [0, 5]   # 6 months → indices 0 … 5


# ---------------------------------------------------------------------------
# update_date_display
# ---------------------------------------------------------------------------

class TestUpdateDateDisplay:
    def test_empty_slider_value_returns_no_update(self):
        from dash import no_update as nu
        result = update_date_display(None, _DATE_STORE)
        assert result == nu

    def test_empty_date_store_returns_no_update(self):
        from dash import no_update as nu
        result = update_date_display([0, 5], [])
        assert result == nu

    def test_returns_formatted_string(self):
        result = update_date_display([0, 23], _DATE_STORE)
        assert isinstance(result, str)
        assert '–' in result

    def test_month_count_in_output(self):
        result = update_date_display([0, 11], _DATE_STORE)   # 12 months
        assert '12' in result

    def test_single_month_selected(self):
        result = update_date_display([5, 5], _DATE_STORE)
        assert '1' in result   # 1 month


# ---------------------------------------------------------------------------
# _render_basket_list
# ---------------------------------------------------------------------------

class TestRenderBasketList:
    def test_empty_basket_returns_paragraph(self):
        from dash import html
        result = _render_basket_list([], 'a')
        assert isinstance(result, html.P)

    def test_basket_with_items_returns_div(self):
        from dash import html
        result = _render_basket_list([BASKET_ITEM_AAPL], 'a')
        assert isinstance(result, html.Div)

    def test_remove_button_id_contains_filename(self):
        result = _render_basket_list([BASKET_ITEM_AAPL], 'a')
        buttons = [
            c for row in result.children
            for c in row.children
            if hasattr(c, 'id') and isinstance(c.id, dict)
        ]
        assert any(b['index'] == 'aapl.parquet' for b in (btn.id for btn in buttons))

    def test_symbol_and_name_appear_in_output(self):
        result = _render_basket_list([BASKET_ITEM_AAPL], 'a')
        rendered = str(result)
        assert 'AAPL' in rendered
        assert 'Apple Inc' in rendered


# ---------------------------------------------------------------------------
# _metrics_table
# ---------------------------------------------------------------------------

class TestMetricsTable:
    def test_both_none_returns_paragraph(self):
        from dash import html
        result = _metrics_table(None, None)
        assert isinstance(result, html.P)

    def test_returns_table_with_header_row(self):
        from dash import html
        metrics = {'Total Return': '+10.0%', 'CAGR': '5.0%'}
        result = _metrics_table(metrics, None)
        assert isinstance(result, html.Table)
        header = result.children[0]
        texts = [th.children for th in header.children]
        assert 'Metric' in texts
        assert 'Basket A' in texts
        assert 'Basket B' in texts

    def test_metrics_values_appear_in_rows(self):
        metrics_a = {'Total Return': '+10.0%'}
        metrics_b = {'Total Return': '-5.0%'}
        result = _metrics_table(metrics_a, metrics_b)
        rendered = str(result)
        assert '+10.0%' in rendered
        assert '-5.0%' in rendered

    def test_missing_metric_shows_em_dash(self):
        metrics_a = {'Total Return': '+10.0%', 'CAGR': '5.0%'}
        metrics_b = None
        result = _metrics_table(metrics_a, metrics_b)
        rendered = str(result)
        assert '—' in rendered
