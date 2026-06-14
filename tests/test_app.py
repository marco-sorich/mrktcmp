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
    _build_slider_marks, _downsample_for_plot, render_order_table,
)
from src.components import (  # noqa: E402
    _render_basket_list, _metrics_table, _order_table_markup, _order_table_component,
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

# A single finalized OrderRow (all 13 keys) used to exercise _order_table.
# period_return is None so the em-dash ('—') rendering path is covered.
_ORDERS_STUB = [{
    'date': pd.Timestamp('2022-01-31', tz='UTC'),
    'side': 'Buy',
    'value_before': 0.0,
    'inflow': 1000.0,
    'assets_after': 1000.0,
    'cash_after': 0.0,
    'value_after': 1000.0,
    'net_deposits': 1000.0,
    'pnl_abs': 0.0,
    'pnl_pct': 0.0,
    'equity_exposure': 1.0,
    'cash_quote': 0.0,
    'period_return': None,
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
            1, [], [], _SLIDER_VAL, _DATE_STORE, None, None)
        assert 'basket' in status
        assert style['display'] == 'none'

    def test_no_base_url_returns_error_status(self):
        with patch.object(config_module, 'base_url', None), \
             patch.object(config_module, 'df', SAMPLE_DF):
            _, style, _, status, _ = run_backtest_callback(
                1, BASKET_A, [], _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG)
        assert 'data source' in status
        assert style['display'] == 'none'

    def test_empty_date_store_returns_error_status(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF):
            _, style, _, status, _ = run_backtest_callback(
                1, BASKET_A, [], _SLIDER_VAL, [], _STRATEGY_CFG, _STRATEGY_CFG)
        assert 'date range' in status.lower()
        assert style['display'] == 'none'

    def test_no_data_returned_shows_error_status(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest', return_value=(None, None, None)):
            _, style, _, status, _ = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG)
        assert style['display'] == 'none'
        assert 'No data' in status

    def test_successful_run_makes_chart_visible(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            _, style, _, _, _ = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG)
        assert style['display'] == 'block'

    def test_successful_run_returns_plotly_figure(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            fig, _, _, _, _ = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG)
        assert isinstance(fig, go.Figure)

    def test_successful_run_populates_orders_store(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            *_, orders_store = run_backtest_callback(
                1, BASKET_A, BASKET_B, _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, _STRATEGY_CFG)
        # Both baskets ran → the store carries each basket's table HTML markup.
        assert '<td>Buy</td>' in orders_store['a']
        assert '<td>Buy</td>' in orders_store['b']

    def test_only_basket_a_filled_also_succeeds(self):
        with patch.object(config_module, 'base_url', 'http://x'), \
             patch.object(config_module, 'df', SAMPLE_DF), \
             patch('src.callbacks.backtesting.run_backtest',
                   return_value=(_PORTFOLIO_STUB, _METRICS_STUB, _ORDERS_STUB)):
            fig, style, _, status, orders_store = run_backtest_callback(
                1, BASKET_A, [], _SLIDER_VAL, _DATE_STORE, _STRATEGY_CFG, None)
        assert style['display'] == 'block'
        assert 'complete' in status
        # Basket B was empty → its stored markup is None (placeholder on render).
        assert orders_store['b'] is None
        assert '<td>Buy</td>' in orders_store['a']


class TestRenderOrderTable:
    def test_active_basket_markup_rendered_as_markdown(self):
        from dash import dcc
        store = {'a': '<table class="order-table"><td>Buy</td></table>', 'b': None}
        comp = render_order_table('a', store)
        assert isinstance(comp, dcc.Markdown)
        assert '<td>Buy</td>' in comp.children

    def test_empty_or_missing_shows_placeholder(self):
        from dash import html
        store = {'a': '<table>x</table>', 'b': None}
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


# ---------------------------------------------------------------------------
# _order_table
# ---------------------------------------------------------------------------

class TestOrderTable:
    def test_markup_none_for_empty(self):
        assert _order_table_markup(None) is None
        assert _order_table_markup([]) is None

    def test_markup_is_native_table_with_headers_and_rows(self):
        markup = _order_table_markup(_ORDERS_STUB)
        # A native HTML table string (rendered later as one dcc.Markdown) rather
        # than hundreds of html.Tr/html.Td, so a long order log renders fast.
        assert '<table class="order-table">' in markup
        assert '<th>Date</th>' in markup
        assert '<th>Buy/Sell</th>' in markup
        assert markup.count('<tr>') == len(_ORDERS_STUB) + 1   # header + one row per order

    def test_markup_formatted_values_and_em_dash_for_none(self):
        markup = _order_table_markup(_ORDERS_STUB)
        assert '<td>Buy</td>' in markup            # side value rendered verbatim
        assert '<td>2022-01-31</td>' in markup     # date formatted as YYYY-MM-DD
        assert '<td>1,000</td>' in markup          # currency uses a thousands separator
        assert '<td>—</td>' in markup              # period_return is None → em-dash
        assert 'P&amp;L (€)' in markup             # '&' in the header is HTML-escaped

    def test_component_wraps_markup_or_placeholder(self):
        from dash import dcc, html
        assert isinstance(_order_table_component(None), html.P)            # empty → placeholder
        comp = _order_table_component('<table class="order-table"></table>')
        assert isinstance(comp, dcc.Markdown)
        assert 'order-table' in comp.children


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
