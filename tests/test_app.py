import os
import pandas as pd
from unittest.mock import patch, MagicMock
import plotly.graph_objects as go

# Ensure no BASE_URL during import so df=None and no network calls are made.
os.environ.pop("BASE_URL", None)

from dash import no_update  # noqa: E402

with patch("dotenv.load_dotenv"):
    import src.app as app_module          # noqa: E402
    import src.config as config_module    # noqa: E402

from src.callbacks.backtesting import (   # noqa: E402
    _bt_assetclass_options, _bt_asset_search, _manage_basket,
    run_backtest_callback, update_date_range_slider, update_date_display,
    _build_slider_marks, _downsample_for_plot, render_order_table, download_orders,
    reset_results_on_input_change,
)
from src.components import (  # noqa: E402
    _render_basket_list, _metrics_table, _order_rows, _order_table_component,
    _base_currency_options, _asset_currency_map, _basket_item_label,
    _weight_percentages,
)
from src.callbacks.backtesting import (  # noqa: E402
    _asset_option_label, _sync_weights, _symbol_weights,
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
# App-level structure
# ---------------------------------------------------------------------------


class TestAppStructure:
    def test_server_is_flask_instance(self):
        import flask
        assert isinstance(app_module.server, flask.Flask)

    def test_layout_is_not_none(self):
        assert app_module.app.layout is not None

    def test_df_is_none_when_no_base_url(self):
        assert config_module.df is None

    def test_asset_classes_empty_when_no_base_url(self):
        assert config_module.assetsClasses == []


# ---------------------------------------------------------------------------
# Backtesting – helpers
# ---------------------------------------------------------------------------

def _make_ctx(triggered_id, triggered_value=1):
    """Build a minimal mock of dash.callback_context."""
    ctx = MagicMock()
    ctx.triggered = [{'prop_id': 'test.n_clicks', 'value': triggered_value}]
    ctx.triggered_id = triggered_id
    return ctx


def _collect_dict_ids(component):
    """Recursively collect every dict-valued component id under *component*."""
    ids = []
    cid = getattr(component, 'id', None)
    if isinstance(cid, dict):
        ids.append(cid)
    children = getattr(component, 'children', None)
    if isinstance(children, (list, tuple)):
        for child in children:
            ids.extend(_collect_dict_ids(child))
    elif children is not None:
        ids.extend(_collect_dict_ids(children))
    return ids


BASKET_ITEM_AAPL = {'filename': 'aapl.parquet', 'symbol': 'AAPL', 'name': 'Apple Inc'}
BASKET_ITEM_GOOGL = {'filename': 'googl.parquet', 'symbol': 'GOOGL', 'name': 'Alphabet Inc'}


# ---------------------------------------------------------------------------
# _bt_assetclass_options
# ---------------------------------------------------------------------------

class TestBtAssetclassOptions:
    def test_no_asset_class_returns_empty_and_disabled(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
            opts, disabled = _bt_assetclass_options(None)
        assert opts == []
        assert disabled is True

    def test_df_none_returns_empty_and_disabled(self):
        with patch.object(config_module, 'df', None):
            opts, disabled = _bt_assetclass_options('stocks')
        assert opts == []
        assert disabled is True

    def test_valid_class_returns_options_and_enabled(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
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
        with patch.object(config_module, 'df', large_df):
            opts, _ = _bt_assetclass_options('stocks')
        assert len(opts) == 200


# ---------------------------------------------------------------------------
# _bt_asset_search
# ---------------------------------------------------------------------------

class TestBtAssetSearch:
    def test_no_asset_class_returns_empty(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
            assert _bt_asset_search(None, None, None) == []

    def test_df_none_returns_empty(self):
        with patch.object(config_module, 'df', None):
            assert _bt_asset_search('AAPL', 'stocks', None) == []

    def test_search_by_symbol_exact_match(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search('aapl', 'stocks', None)
        assert len(opts) == 1
        assert opts[0]['value'] == 'aapl.parquet'

    def test_search_by_name_partial_match(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search('alphabet', 'stocks', None)
        assert len(opts) == 1
        assert opts[0]['value'] == 'googl.parquet'

    def test_current_value_appended_when_not_in_results(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search('GOOGL', 'stocks', 'aapl.parquet')
        values = [o['value'] for o in opts]
        assert 'aapl.parquet' in values
        assert 'googl.parquet' in values

    def test_current_value_not_duplicated(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
            opts = _bt_asset_search(None, 'stocks', 'aapl.parquet')
        assert [o['value'] for o in opts].count('aapl.parquet') == 1

    def test_no_matches_returns_empty(self):
        with patch.object(config_module, 'df', SAMPLE_DF):
            assert _bt_asset_search('ZZZZZZ', 'stocks', None) == []


# ---------------------------------------------------------------------------
# _manage_basket
# ---------------------------------------------------------------------------

class TestManageBasket:
    def test_add_asset_to_empty_basket(self):
        ctx = _make_ctx('bt-add-a')
        with patch('dash.callback_context', ctx), patch.object(config_module, 'df', SAMPLE_DF):
            basket, _ = _manage_basket('a', [], 'aapl.parquet', [])
        assert len(basket) == 1
        assert basket[0]['filename'] == 'aapl.parquet'
        assert basket[0]['symbol'] == 'AAPL'

    def test_duplicate_asset_is_not_added_twice(self):
        ctx = _make_ctx('bt-add-a')
        with patch('dash.callback_context', ctx), patch.object(config_module, 'df', SAMPLE_DF):
            basket, _ = _manage_basket('a', [], 'aapl.parquet', [BASKET_ITEM_AAPL])
        assert len(basket) == 1

    def test_add_with_no_selected_asset_returns_no_update(self):
        ctx = _make_ctx('bt-add-a')
        with patch('dash.callback_context', ctx), patch.object(config_module, 'df', SAMPLE_DF):
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

# A single finalized OrderRow (all keys) used to exercise _order_table.
# period_return is None so the em-dash ('—') rendering path is covered.
# bh_value is None so the em-dash path is covered for that column too.
# asset_values splits the 1,000 assets_after across two symbols and asset_prices
# carries each symbol's quote, so the dynamic per-asset value + price columns are
# exercised by the table/download tests.
_ORDERS_STUB = [{
    'date': pd.Timestamp('2022-01-31', tz='UTC'),
    'side': 'Buy',
    'value_before': 0.0,
    'inflow': 1000.0,
    'assets_after': 1000.0,
    'cash_after': 0.0,
    'asset_values': {'AAPL': 600.0, 'MSFT': 400.0},
    'asset_prices': {'AAPL': 150.0, 'MSFT': 250.0},
    'value_after': 1000.0,
    'bh_value': None,
    'net_deposits': 1000.0,
    'pnl_abs': 0.0,
    'pnl_pct': 0.0,
    'equity_exposure': 1.0,
    'cash_quote': 0.0,
    'period_return': None,
}]

# A foreign-currency variant: AAPL trades in USD, so the row carries the local
# (trading-currency) close and the day's FX rate alongside the converted close,
# exercising the order table's split price + FX-pair columns.
_ORDERS_STUB_FX = [{
    **_ORDERS_STUB[0],
    'asset_values': {'AAPL': 1000.0},
    'asset_prices': {'AAPL': 135.0},        # converted close (base = EUR)
    'asset_prices_local': {'AAPL': 150.0},  # trading-currency close (USD)
    'fx_rates': {'USDEUR=X': 0.90},
}]

# Same, but the FX rate is missing for that row (→ em-dash in the rate column).
_ORDERS_STUB_FX_NO_RATE = [{
    **_ORDERS_STUB_FX[0],
    'fx_rates': {'USDEUR=X': None},
}]

BASKET_A = [BASKET_ITEM_AAPL]
BASKET_B = [BASKET_ITEM_GOOGL]

_TEST_DATES = pd.date_range('2022-01-31', periods=24, freq='ME', tz='UTC')
_DATE_STORE = [d.isoformat() for d in _TEST_DATES]
_SLIDER_VAL = [0, 23]


_STRATEGY_CFG = {'strategy': 'DCA', 'params': {'monthly_investment': 1000.0}}


class TestRunBacktestCallback:
    def test_both_baskets_empty_returns_status_message(self):
        _, style, _, status, _ = run_backtest_callback(
            1, [], [], _SLIDER_VAL, _DATE_STORE, None, None, 'EUR')
        assert 'basket' in status
        assert style['display'] == 'none'

    def test_no_base_url_returns_error_status(self):
        with patch.object(config_module, 'base_url', None), \
             patch.object(config_module, 'df', SAMPLE_DF):
            _, style, _, status, _ = run_backtest_callback(
                1, BASKET_A, [], _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG, 'EUR')
        assert 'data source' in status
        assert style['display'] == 'none'

    def test_empty_date_store_returns_error_status(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF):
            _, style, _, status, _ = run_backtest_callback(
                1, BASKET_A, [], _SLIDER_VAL, [], _STRATEGY_CFG, _STRATEGY_CFG, 'EUR')
        assert 'date range' in status.lower()
        assert style['display'] == 'none'

    def test_no_data_returned_shows_error_status(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest', return_value=(None, None, None)):
            _, style, _, status, _ = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG, 'EUR')
        assert style['display'] == 'none'
        assert 'No data' in status

    def test_successful_run_makes_chart_visible(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            _, style, _, _, _ = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG, 'EUR')
        assert style['display'] == 'block'

    def test_successful_run_returns_plotly_figure(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            fig, _, _, _, _ = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG, 'EUR')
        assert isinstance(fig, go.Figure)

    def test_successful_run_populates_orders_store(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            *_, orders_store = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG, 'EUR')
        # Both baskets ran → the store carries each basket's display rows.
        assert orders_store['a'][0]['Buy/Sell'] == 'Buy'
        assert orders_store['b'][0]['Buy/Sell'] == 'Buy'

    def test_only_basket_a_filled_also_succeeds(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            fig, style, _, status, orders_store = run_backtest_callback(
                1, BASKET_A, [], _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, None, 'EUR')
        assert style['display'] == 'block'
        assert 'complete' in status
        # Basket B was empty → its stored rows are None (placeholder on render).
        assert orders_store['b'] is None
        assert orders_store['a'][0]['Buy/Sell'] == 'Buy'


class TestResetResultsOnInputChange:
    """reset_results_on_input_change restores the initial page-load state."""

    def test_returns_visible_chart_style(self):
        _, style, *_ = reset_results_on_input_change(None, None, 'EUR', None, None)
        assert style['display'] == 'block'

    def test_returns_empty_figure(self):
        fig, *_ = reset_results_on_input_change(None, None, 'EUR', None, None)
        assert isinstance(fig, go.Figure)
        assert fig.data == ()

    def test_returns_metrics_table_with_placeholder_dashes(self):
        from dash import html
        _, _, metrics, _, _ = reset_results_on_input_change(None, None, 'EUR', None, None)
        assert isinstance(metrics, html.Table)
        rendered = str(metrics)
        assert '—' in rendered

    def test_returns_empty_status_and_orders_store(self):
        *_, status, orders = reset_results_on_input_change(None, None, 'EUR', None, None)
        assert status == ''
        assert orders == {}

    def test_resets_regardless_of_basket_content(self):
        basket = [{'filename': 'aapl.parquet', 'symbol': 'AAPL', 'name': 'Apple', 'currency': 'USD'}]
        _, style, _, _, orders = reset_results_on_input_change(basket, None, 'USD', None, None)
        assert style['display'] == 'block'
        assert orders == {}


class TestRenderOrderTable:
    def test_active_basket_rows_rendered_as_markdown(self):
        from dash import dcc
        store = {'a': _order_rows(_ORDERS_STUB), 'b': None}
        comp = render_order_table('a', store)
        assert isinstance(comp, dcc.Markdown)
        assert '<td>Buy</td>' in comp.children

    def test_empty_or_missing_shows_placeholder(self):
        from dash import html
        store = {'a': _order_rows(_ORDERS_STUB), 'b': None}
        assert isinstance(render_order_table('b', store), html.P)   # B empty → placeholder
        assert isinstance(render_order_table('a', None), html.P)    # no store yet
        assert isinstance(render_order_table(None, {}), html.P)     # defaults to 'a', empty


# ---------------------------------------------------------------------------
# _build_slider_marks
# ---------------------------------------------------------------------------

class TestBuildSliderMarks:
    def test_short_range_marks_every_month(self):
        dates = pd.date_range('2024-01-31', periods=6, freq='ME', tz='UTC')
        marks = _build_slider_marks(dates)
        assert 0 in marks
        assert 5 in marks

    def test_medium_range_marks_quarterly(self):
        dates = pd.date_range('2021-01-31', periods=24, freq='ME', tz='UTC')
        marks = _build_slider_marks(dates)
        assert 0 in marks
        assert 12 in marks

    def test_long_range_marks_yearly(self):
        dates = pd.date_range('2015-01-31', periods=120, freq='ME', tz='UTC')
        marks = _build_slider_marks(dates)
        assert 0 in marks
        assert 24 in marks

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
        with patch.object(config_module, 'base_url', None):
            *_, disabled, _store, _display = update_date_range_slider(BASKET_A, [])
        assert disabled is True

    def test_no_overlap_disables_slider(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.get_common_date_range', return_value=(None, None)):
            *_, disabled, _store, _display = update_date_range_slider(BASKET_A, BASKET_B)
        assert disabled is True

    def test_valid_range_enables_slider(self):
        common_start = pd.Timestamp('2020-01-31', tz='UTC')
        common_end = pd.Timestamp('2022-12-31', tz='UTC')
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.get_common_date_range', return_value=(common_start, common_end)):
            *_, disabled, _store, _display = update_date_range_slider(BASKET_A, [])
        assert disabled is False

    def test_date_store_contains_iso_strings(self):
        common_start = pd.Timestamp('2022-01-31', tz='UTC')
        common_end = pd.Timestamp('2022-03-31', tz='UTC')
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.get_common_date_range', return_value=(common_start, common_end)):
            *_, _disabled, date_store, _display = update_date_range_slider(BASKET_A, [])
        assert len(date_store) == 3
        pd.Timestamp(date_store[0])

    def test_slider_value_covers_full_range(self):
        common_start = pd.Timestamp('2022-01-31', tz='UTC')
        common_end = pd.Timestamp('2022-06-30', tz='UTC')
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.get_common_date_range', return_value=(common_start, common_end)):
            _min, _max, value, *_ = update_date_range_slider(BASKET_A, [])
        assert value == [0, 5]


# ---------------------------------------------------------------------------
# update_date_display
# ---------------------------------------------------------------------------

class TestUpdateDateDisplay:
    def test_empty_slider_value_returns_no_update(self):
        result = update_date_display(None, _DATE_STORE)
        assert result == no_update

    def test_empty_date_store_returns_no_update(self):
        result = update_date_display([0, 5], [])
        assert result == no_update

    def test_returns_formatted_string(self):
        result = update_date_display([0, 23], _DATE_STORE)
        assert isinstance(result, str)
        assert '–' in result

    def test_month_count_in_output(self):
        result = update_date_display([0, 11], _DATE_STORE)
        assert '12' in result

    def test_single_month_selected(self):
        result = update_date_display([5, 5], _DATE_STORE)
        assert '1' in result


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
        # The remove button now lives inside a per-row controls sub-Div alongside
        # the weight input, so collect dict-id components recursively.
        ids = _collect_dict_ids(result)
        remove_ids = [i for i in ids if i.get('type') == 'bt-remove-a']
        assert any(i['index'] == 'aapl.parquet' for i in remove_ids)

    def test_weight_input_and_percent_rendered(self):
        # Each row carries a weight input and a live percentage label keyed by the
        # asset's filename; a single asset is 100% of the basket.
        result = _render_basket_list([BASKET_ITEM_AAPL], 'a', {'aapl.parquet': 1.0})
        ids = _collect_dict_ids(result)
        assert {'type': 'bt-weight-a', 'index': 'aapl.parquet'} in ids
        assert {'type': 'bt-weight-pct-a', 'index': 'aapl.parquet'} in ids
        assert '100%' in str(result)

    def test_weight_percentages_reflect_relative_weights(self):
        # Two assets weighted 3:1 → 75% / 25%.
        result = _render_basket_list(
            [BASKET_ITEM_AAPL, BASKET_ITEM_GOOGL], 'a',
            {'aapl.parquet': 3.0, 'googl.parquet': 1.0},
        )
        rendered = str(result)
        assert '75%' in rendered
        assert '25%' in rendered

    def test_symbol_and_name_appear_in_output(self):
        result = _render_basket_list([BASKET_ITEM_AAPL], 'a')
        rendered = str(result)
        assert 'AAPL' in rendered
        assert 'Apple Inc' in rendered

    def test_currency_tag_appears_when_present(self):
        item = {**BASKET_ITEM_AAPL, 'currency': 'USD'}
        assert 'AAPL — Apple Inc (USD)' in str(_render_basket_list([item], 'a'))

    def test_blank_currency_gets_no_tag(self):
        # Blank/placeholder currency → no parenthesised tag, just symbol — name.
        assert _basket_item_label({**BASKET_ITEM_AAPL, 'currency': '0'}) == 'AAPL — Apple Inc'
        assert _basket_item_label(BASKET_ITEM_AAPL) == 'AAPL — Apple Inc'  # key absent
        assert _basket_item_label({**BASKET_ITEM_AAPL, 'currency': 'GBp'}) == 'AAPL — Apple Inc (GBp)'


# ---------------------------------------------------------------------------
# _asset_currency_map / _asset_option_label
# ---------------------------------------------------------------------------

class TestAssetCurrencyMap:
    def test_maps_symbol_to_currency(self):
        assert _asset_currency_map(['aapl.parquet', 'a.parquet'], _CURRENCY_DF) == {
            'AAPL': 'USD', 'USDEUR=X': 'EUR',
        }

    def test_no_currency_column_returns_empty(self):
        df = pd.DataFrame({'symbol': ['AAPL'], 'filename': ['aapl.parquet']})
        assert _asset_currency_map(['aapl.parquet'], df) == {}

    def test_none_catalogue_returns_empty(self):
        assert _asset_currency_map(['aapl.parquet'], None) == {}


class TestAssetOptionLabel:
    def test_label_includes_currency_tag(self):
        row = pd.Series({'symbol': 'AAPL', 'name': 'Apple', 'interval': '1d', 'currency': 'USD'})
        assert _asset_option_label(row) == 'AAPL — Apple (1d · USD)'

    def test_label_drops_blank_currency(self):
        row = pd.Series({'symbol': 'IDX', 'name': 'Index', 'interval': '1d', 'currency': '0'})
        assert _asset_option_label(row) == 'IDX — Index (1d)'

    def test_label_without_currency_column(self):
        row = pd.Series({'symbol': 'AAPL', 'name': 'Apple', 'interval': '1d'})
        assert _asset_option_label(row) == 'AAPL — Apple (1d)'


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


# ---------------------------------------------------------------------------
# _order_table
# ---------------------------------------------------------------------------

class TestOrderTable:
    def test_rows_none_for_empty(self):
        assert _order_rows(None) is None
        assert _order_rows([]) is None

    def test_rows_are_formatted_keyed_by_column_label(self):
        rows = _order_rows(_ORDERS_STUB)
        assert len(rows) == len(_ORDERS_STUB)
        row = rows[0]
        assert row['Buy/Sell'] == 'Buy'            # side value rendered verbatim
        assert row['Date'] == '2022-01-31'         # date formatted as YYYY-MM-DD
        assert row['Inflow'] == '1,000'            # currency uses a thousands separator
        assert row['Period return'] == '—'         # period_return is None → em-dash
        # Dynamic per-asset columns: a value (no decimals) and a price (2 dp)
        # column per basket asset.
        assert row['AAPL value'] == '600'
        assert row['MSFT value'] == '400'
        assert row['AAPL price'] == '150.00'
        assert row['MSFT price'] == '250.00'

    def test_pnl_header_names_selected_base_currency(self):
        # The P&L column header carries the chosen reporting currency code.
        assert 'P&L (EUR)' in _order_rows(_ORDERS_STUB)[0]          # default
        assert 'P&L (USD)' in _order_rows(_ORDERS_STUB, 'USD')[0]   # explicit
        assert 'P&L (€)' not in _order_rows(_ORDERS_STUB, 'USD')[0]

    def test_unknown_currency_keeps_single_plain_price_column(self):
        # No asset_currency given → currency unknown → one unlabelled price column
        # (the back-compatible behaviour, exercised by the stub-based tests above).
        row = _order_rows(_ORDERS_STUB)[0]
        assert 'AAPL price' in row
        assert 'AAPL price (EUR)' not in row

    def test_base_currency_asset_gets_single_labelled_price_column(self):
        # Asset trades in the reporting currency → one column labelled with it.
        row = _order_rows(_ORDERS_STUB, 'EUR', {'AAPL': 'EUR', 'MSFT': 'EUR'})[0]
        assert row['AAPL price (EUR)'] == '150.00'
        assert 'AAPL price (USD)' not in row

    def test_foreign_currency_asset_shows_both_price_columns(self):
        # AAPL trades in USD, reporting in EUR → both quotes side by side.
        rows = _order_rows(_ORDERS_STUB_FX, 'EUR', {'AAPL': 'USD'})
        row = rows[0]
        assert row['AAPL price (USD)'] == '150.00'   # trading-currency close
        assert row['AAPL price (EUR)'] == '135.00'   # converted close (150 × 0.90)
        # The FX pair used for the conversion gets its own rate column, last.
        assert row['USDEUR=X'] == '0.9000'
        assert list(row).index('USDEUR=X') == len(row) - 1

    def test_missing_fx_rate_renders_em_dash(self):
        rows = _order_rows(_ORDERS_STUB_FX_NO_RATE, 'EUR', {'AAPL': 'USD'})
        assert rows[0]['USDEUR=X'] == '—'


# ---------------------------------------------------------------------------
# _base_currency_options
# ---------------------------------------------------------------------------

_CURRENCY_DF = pd.DataFrame({
    'asset_class': ['Stocks', 'currency', 'currency', 'currency', 'currency'],
    'symbol': ['AAPL', 'USDEUR=X', 'GBPEUR=X', 'EURGBp=X', 'BADxx=X'],
    'name': ['Apple', 'USD/EUR', 'GBP/EUR', 'EUR/GBp', 'BAD/xx'],
    'filename': ['aapl.parquet', 'a.parquet', 'b.parquet', 'c.parquet', 'd.parquet'],
    'currency': ['USD', 'EUR', 'EUR', 'GBp', '0'],
})


class TestBaseCurrencyOptions:
    def test_options_from_quote_currencies_sorted_with_default(self):
        with patch.object(config_module, 'df', _CURRENCY_DF), \
             patch.object(config_module, 'default_base_currency', 'EUR'):
            opts = _base_currency_options()
        assert opts == sorted(opts)        # sorted
        assert 'EUR' in opts               # the default is always present
        # GBp (pence sub-unit) and the '0' placeholder are excluded.
        assert 'GBp' not in opts and '0' not in opts

    def test_fallback_to_default_without_catalogue(self):
        with patch.object(config_module, 'df', None), \
             patch.object(config_module, 'default_base_currency', 'USD'):
            assert _base_currency_options() == ['USD']

    def test_component_renders_rows_as_native_table(self):
        from dash import dcc, html
        assert isinstance(_order_table_component(None), html.P)   # empty → placeholder
        comp = _order_table_component(_order_rows(_ORDERS_STUB))
        # A single dcc.Markdown holding a native HTML table (one component) rather
        # than hundreds of html.Tr/html.Td, so a long order log renders fast.
        assert isinstance(comp, dcc.Markdown)
        markup = comp.children
        assert '<table class="order-table">' in markup
        assert '<th>Date</th>' in markup and '<th>Buy/Sell</th>' in markup
        assert '<td>Buy</td>' in markup and '<td>2022-01-31</td>' in markup
        assert '<td>—</td>' in markup              # None → em-dash
        # '&' in the header is HTML-escaped; the reporting currency (default EUR)
        # is appended to the P&L header by _order_rows.
        assert 'P&amp;L (EUR)' in markup
        # Dynamic per-asset value + price columns render as their own header + cell.
        assert '<th>AAPL value</th>' in markup and '<th>AAPL price</th>' in markup
        assert '<th>MSFT value</th>' in markup and '<th>MSFT price</th>' in markup
        assert '<td>600</td>' in markup and '<td>400</td>' in markup
        assert '<td>150.00</td>' in markup and '<td>250.00</td>' in markup
        assert markup.count('<tr>') == len(_ORDERS_STUB) + 1   # header + one per order


class TestDownloadOrders:
    def test_csv_export_of_active_basket(self):
        with patch('dash.callback_context') as ctx:
            ctx.triggered_id = 'bt-dl-csv'
            out = download_orders(1, 0, 'a', {'a': _order_rows(_ORDERS_STUB), 'b': None})
        assert out['filename'] == 'orders_basket_A.csv'
        assert 'Buy' in out['content']        # data present
        assert 'Date' in out['content']       # header row present
        assert 'AAPL' in out['content']       # per-asset value column exported too

    def test_xlsx_export_is_base64(self):
        with patch('dash.callback_context') as ctx:
            ctx.triggered_id = 'bt-dl-xlsx'
            out = download_orders(0, 1, 'a', {'a': _order_rows(_ORDERS_STUB), 'b': None})
        assert out['filename'] == 'orders_basket_A.xlsx'
        assert out.get('base64') is True

    def test_empty_basket_downloads_nothing(self):
        from dash import no_update
        with patch('dash.callback_context') as ctx:
            ctx.triggered_id = 'bt-dl-csv'
            out = download_orders(1, 0, 'b', {'a': _order_rows(_ORDERS_STUB), 'b': None})
        assert out is no_update


# ---------------------------------------------------------------------------
# Per-asset weighting (UI level)
# ---------------------------------------------------------------------------

class TestWeightPercentages:
    def test_shares_of_total(self):
        assert _weight_percentages({'a': 3.0, 'b': 1.0}) == {'a': '75%', 'b': '25%'}

    def test_all_zero_renders_em_dash(self):
        assert _weight_percentages({'a': 0.0, 'b': 0.0}) == {'a': '—', 'b': '—'}

    def test_single_asset_is_full(self):
        assert _weight_percentages({'a': 5.0}) == {'a': '100%'}


class TestSyncWeights:
    def _meta(self, basket_id, filenames, kind):
        return [{'id': {'type': f'bt-{kind}-{basket_id}', 'index': fn}} for fn in filenames]

    def test_builds_weights_and_percentages(self):
        files = ['aapl.parquet', 'googl.parquet']
        weights, pct = _sync_weights(
            [3.0, 1.0],
            self._meta('a', files, 'weight'),
            self._meta('a', files, 'weight-pct'),
        )
        assert weights == {'aapl.parquet': 3.0, 'googl.parquet': 1.0}
        assert pct == ['75%', '25%']

    def test_cleared_input_falls_back_to_default_weight(self):
        files = ['aapl.parquet', 'googl.parquet']
        weights, _ = _sync_weights(
            [None, 2.0],
            self._meta('a', files, 'weight'),
            self._meta('a', files, 'weight-pct'),
        )
        # A cleared field defaults back to 1.0 rather than zeroing the asset.
        assert weights == {'aapl.parquet': 1.0, 'googl.parquet': 2.0}

    def test_negative_input_clamped_to_zero(self):
        weights, _ = _sync_weights(
            [-5.0], self._meta('a', ['aapl.parquet'], 'weight'),
            self._meta('a', ['aapl.parquet'], 'weight-pct'),
        )
        assert weights == {'aapl.parquet': 0.0}


class TestSymbolWeights:
    def test_maps_filename_weights_to_symbols(self):
        assert _symbol_weights([BASKET_ITEM_AAPL], {'aapl.parquet': 2.0}) == {'AAPL': 2.0}

    def test_missing_weight_defaults_to_one(self):
        assert _symbol_weights([BASKET_ITEM_AAPL], {}) == {'AAPL': 1.0}

    def test_empty_basket_is_none(self):
        assert _symbol_weights([], {'aapl.parquet': 2.0}) is None

    def test_all_zero_weights_is_none(self):
        # Every weight non-positive → None so the engine falls back to equal weight.
        assert _symbol_weights([BASKET_ITEM_AAPL], {'aapl.parquet': 0.0}) is None


class TestRunBacktestForwardsWeights:
    def test_symbol_weights_passed_to_run_backtest(self):
        captured = []

        def _fake(*args, **kwargs):
            captured.append(kwargs.get('weights'))
            return (_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)

        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest', side_effect=_fake):
            run_backtest_callback(
                1, BASKET_A, [], _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, None, 'EUR',
                {'aapl.parquet': 2.0}, {},
            )
        # Basket A's filename weight is translated to its symbol for the engine.
        assert {'AAPL': 2.0} in captured


# ---------------------------------------------------------------------------
# _downsample_for_plot
# ---------------------------------------------------------------------------

class TestDownsampleForPlot:
    def test_short_series_returned_unchanged(self):
        s = pd.Series(range(100), index=pd.date_range('2020-01-01', periods=100, freq='D'))
        # Already small enough → same object, no copy.
        assert _downsample_for_plot(s, max_points=2000) is s

    def test_long_series_thinned_but_endpoints_kept(self):
        n = 7000
        s = pd.Series(range(n), index=pd.date_range('2000-01-01', periods=n, freq='D'))
        out = _downsample_for_plot(s, max_points=2000)
        assert len(out) <= 2001                 # ~max_points (+ appended last point)
        assert out.index[0] == s.index[0]       # first point preserved
        assert out.index[-1] == s.index[-1]     # final point always preserved
        assert out.iloc[-1] == s.iloc[-1]


# ---------------------------------------------------------------------------
# /healthz endpoint
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Tests for the GET /healthz health check endpoint."""

    def _client(self):
        return app_module.server.test_client()

    def test_healthy_returns_200(self):
        """All checks pass → 200 with status 'ok'."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch.object(config_module, 'df', SAMPLE_DF), \
             patch.object(config_module, 'base_url', 'https://example.com'), \
             patch('requests.head', return_value=mock_resp):
            r = self._client().get('/healthz')
        assert r.status_code == 200
        data = r.get_json()
        assert data['status'] == 'ok'
        assert data['checks']['catalogue'] == 'ok'
        assert data['checks']['parquet_access'] == 'ok'

    def test_no_catalogue_returns_503(self):
        """df is None → catalogue check fails → 503."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch.object(config_module, 'df', None), \
             patch.object(config_module, 'base_url', 'https://example.com'), \
             patch('requests.head', return_value=mock_resp):
            r = self._client().get('/healthz')
        assert r.status_code == 503
        data = r.get_json()
        assert data['status'] == 'error'
        assert data['checks']['catalogue'] == 'error'

    def test_no_base_url_returns_503(self):
        """base_url is None → parquet_access check fails → 503."""
        with patch.object(config_module, 'df', SAMPLE_DF), \
             patch.object(config_module, 'base_url', None):
            r = self._client().get('/healthz')
        assert r.status_code == 503
        data = r.get_json()
        assert data['status'] == 'error'
        assert data['checks']['parquet_access'] == 'error'

    def test_unreachable_parquet_returns_503(self):
        """HEAD request raises ConnectionError → parquet_access fails → 503."""
        import requests as real_requests
        with patch.object(config_module, 'df', SAMPLE_DF), \
             patch.object(config_module, 'base_url', 'https://example.com'), \
             patch('requests.head', side_effect=real_requests.ConnectionError):
            r = self._client().get('/healthz')
        assert r.status_code == 503
        data = r.get_json()
        assert data['checks']['parquet_access'] == 'error'

    def test_http_non_ok_response_returns_503(self):
        """HEAD returns 404 → parquet_access check reports error."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        with patch.object(config_module, 'df', SAMPLE_DF), \
             patch.object(config_module, 'base_url', 'https://example.com'), \
             patch('requests.head', return_value=mock_resp):
            r = self._client().get('/healthz')
        assert r.status_code == 503
        data = r.get_json()
        assert data['checks']['parquet_access'] == 'error'

    def test_response_includes_version(self):
        """Response body always carries the app version string."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch.object(config_module, 'df', SAMPLE_DF), \
             patch.object(config_module, 'base_url', 'https://example.com'), \
             patch('requests.head', return_value=mock_resp):
            r = self._client().get('/healthz')
        data = r.get_json()
        assert 'version' in data
        assert isinstance(data['version'], str)

    def test_local_path_exists(self):
        """base_url is a local directory and master.parquet exists → 200."""
        with patch.object(config_module, 'df', SAMPLE_DF), \
             patch.object(config_module, 'base_url', '/data'), \
             patch('os.path.exists', return_value=True):
            r = self._client().get('/healthz')
        assert r.status_code == 200

    def test_local_path_missing(self):
        """base_url is a local directory but master.parquet is absent → 503."""
        with patch.object(config_module, 'df', SAMPLE_DF), \
             patch.object(config_module, 'base_url', '/data'), \
             patch('os.path.exists', return_value=False):
            r = self._client().get('/healthz')
        assert r.status_code == 503
        assert r.get_json()['checks']['parquet_access'] == 'error'
