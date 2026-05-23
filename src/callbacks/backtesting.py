import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update, ALL
import dash

import src.config as _config
from src.utils import log_time
from src.components import _render_basket_list, _metrics_table
from src.backtest import run_backtest, get_common_date_range


@callback(
    Output('bt-asset-a', 'options'),
    Output('bt-asset-a', 'disabled'),
    Input('bt-assetclass-a', 'value'),
)
@log_time
def bt_assetclass_a(asset_class):
    return _bt_assetclass_options(asset_class)


@callback(
    Output('bt-asset-b', 'options'),
    Output('bt-asset-b', 'disabled'),
    Input('bt-assetclass-b', 'value'),
)
@log_time
def bt_assetclass_b(asset_class):
    return _bt_assetclass_options(asset_class)


def _bt_assetclass_options(asset_class):
    if not asset_class or _config.df is None:
        return [], True
    filtered = _config.df[_config.df['asset_class'] == asset_class].head(200)
    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]
    return options, False


@callback(
    Output('bt-asset-a', 'options', allow_duplicate=True),
    Input('bt-asset-a', 'search_value'),
    State('bt-assetclass-a', 'value'),
    State('bt-asset-a', 'value'),
    prevent_initial_call=True,
)
@log_time
def bt_search_a(search_value, asset_class, current_value):
    return _bt_asset_search(search_value, asset_class, current_value)


@callback(
    Output('bt-asset-b', 'options', allow_duplicate=True),
    Input('bt-asset-b', 'search_value'),
    State('bt-assetclass-b', 'value'),
    State('bt-asset-b', 'value'),
    prevent_initial_call=True,
)
@log_time
def bt_search_b(search_value, asset_class, current_value):
    return _bt_asset_search(search_value, asset_class, current_value)


def _bt_asset_search(search_value, asset_class, current_value):
    if not asset_class or _config.df is None:
        return []
    filtered = _config.df[_config.df['asset_class'] == asset_class]

    if search_value:
        sl = search_value.lower()
        sym = filtered['symbol'].str.lower()
        name = filtered['name'].str.lower()
        score = np.select(
            [sym == sl,
             sym.str.startswith(sl, na=False),
             name.str.startswith(sl, na=False),
             sym.str.contains(sl, na=False),
             name.str.contains(sl, na=False)],
            [0, 1, 2, 3, 4],
            default=99,
        )
        mask = score < 99
        filtered = (filtered[mask]
                    .assign(_score=score[mask])
                    .sort_values('_score')
                    .drop(columns='_score')
                    .head(30))
    else:
        filtered = filtered.head(30)

    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]

    if current_value and not any(o['value'] == current_value for o in options):
        sel = _config.df[_config.df['filename'] == current_value]
        if not sel.empty:
            row = sel.iloc[0]
            options.append({
                'label': f"{row['symbol']} — {row['name']} ({row['interval']})",
                'value': current_value,
            })
    return options


@callback(
    Output('bt-basket-store-a', 'data'),
    Output('bt-basket-list-a', 'children'),
    Input('bt-add-a', 'n_clicks'),
    Input({'type': 'bt-remove-a', 'index': ALL}, 'n_clicks'),
    State('bt-asset-a', 'value'),
    State('bt-basket-store-a', 'data'),
    prevent_initial_call=True,
)
@log_time
def manage_basket_a(add_clicks, remove_clicks, selected_asset, basket_data):
    return _manage_basket('a', remove_clicks, selected_asset, basket_data)


@callback(
    Output('bt-basket-store-b', 'data'),
    Output('bt-basket-list-b', 'children'),
    Input('bt-add-b', 'n_clicks'),
    Input({'type': 'bt-remove-b', 'index': ALL}, 'n_clicks'),
    State('bt-asset-b', 'value'),
    State('bt-basket-store-b', 'data'),
    prevent_initial_call=True,
)
@log_time
def manage_basket_b(add_clicks, remove_clicks, selected_asset, basket_data):
    return _manage_basket('b', remove_clicks, selected_asset, basket_data)


def _manage_basket(basket_id, remove_clicks, selected_asset, basket_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    triggered_id = ctx.triggered_id
    triggered_value = ctx.triggered[0].get('value', 0) or 0
    basket = list(basket_data or [])

    if isinstance(triggered_id, dict) and triggered_id.get('type') == f'bt-remove-{basket_id}':
        if triggered_value > 0:
            filename = triggered_id['index']
            basket = [item for item in basket if item['filename'] != filename]
    elif triggered_id == f'bt-add-{basket_id}' and selected_asset and _config.df is not None:
        if not any(item['filename'] == selected_asset for item in basket):
            meta = _config.df[_config.df['filename'] == selected_asset]
            if not meta.empty:
                row = meta.iloc[0]
                basket.append({'filename': selected_asset, 'symbol': row['symbol'], 'name': row['name']})
    else:
        return no_update, no_update

    return basket, _render_basket_list(basket, basket_id)


def _build_slider_marks(date_range):
    n = len(date_range)
    step = 1 if n <= 12 else 12 if n <= 36 else 24
    fmt = '%b' if n <= 12 else '%Y'
    marks = {}
    for i, d in enumerate(date_range):
        if i % step == 0:
            marks[i] = d.strftime(fmt)
    return marks


@callback(
    Output('bt-date-range', 'min'),
    Output('bt-date-range', 'max'),
    Output('bt-date-range', 'value'),
    Output('bt-date-range', 'marks'),
    Output('bt-date-range', 'disabled'),
    Output('bt-date-store', 'data'),
    Output('bt-date-display', 'children'),
    Input('bt-basket-store-a', 'data'),
    Input('bt-basket-store-b', 'data'),
)
@log_time
def update_date_range_slider(basket_a, basket_b):
    _disabled = (0, 1, [0, 1], {}, True, [], '')
    filenames_a = [item['filename'] for item in (basket_a or [])]
    filenames_b = [item['filename'] for item in (basket_b or [])]

    if not filenames_a and not filenames_b:
        return (*_disabled[:6], 'Add assets to a basket to see the available date range.')

    if not _config.base_url or _config.df is None:
        return (*_disabled[:6], 'No data source configured.')

    common_start, common_end = get_common_date_range(
        _config.base_url, filenames_a, filenames_b, _config.df,
    )

    if common_start is None:
        return (*_disabled[:6], 'No overlapping date range found across the selected assets.')

    date_range = pd.date_range(common_start, common_end, freq='ME')
    n = len(date_range)
    date_store = [d.isoformat() for d in date_range]
    marks = _build_slider_marks(date_range)
    d0 = date_range[0].strftime('%b %Y')
    d1 = date_range[-1].strftime('%b %Y')
    display = f'Available: {d0} – {d1}  ({n} months)'
    return 0, n - 1, [0, n - 1], marks, False, date_store, display


@callback(
    Output('bt-date-display', 'children', allow_duplicate=True),
    Input('bt-date-range', 'value'),
    State('bt-date-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def update_date_display(slider_value, date_store):
    if not slider_value or not date_store:
        return no_update
    i0, i1 = slider_value[0], slider_value[1]
    d0 = pd.Timestamp(date_store[i0]).strftime('%b %Y')
    d1 = pd.Timestamp(date_store[i1]).strftime('%b %Y')
    n_months = i1 - i0 + 1
    return f'Selected: {d0} – {d1}  ({n_months} months)'


@callback(
    Output('bt-chart', 'figure'),
    Output('bt-chart', 'style'),
    Output('bt-metrics', 'children'),
    Output('bt-status', 'children'),
    Input('bt-run', 'n_clicks'),
    State('bt-basket-store-a', 'data'),
    State('bt-basket-store-b', 'data'),
    State('bt-date-range', 'value'),
    State('bt-date-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def run_backtest_callback(n_clicks, basket_a, basket_b, slider_value, date_store):
    empty_chart = go.Figure()
    hidden = {'width': '100%', 'display': 'none'}
    visible = {'width': '100%', 'display': 'block'}

    if not basket_a and not basket_b:
        return empty_chart, hidden, '', 'Please fill at least one basket.'

    if not _config.base_url or _config.df is None:
        return empty_chart, hidden, '', 'No data source available.'

    if not date_store or not slider_value or len(date_store) < 2:
        return empty_chart, hidden, '', 'No date range available. Add assets first.'

    start_date = pd.Timestamp(date_store[slider_value[0]])
    end_date = pd.Timestamp(date_store[slider_value[1]])

    filenames_a = [item['filename'] for item in (basket_a or [])]
    filenames_b = [item['filename'] for item in (basket_b or [])]

    portfolio_a, metrics_a = (
        run_backtest(_config.base_url, filenames_a, start_date, end_date, _config.df)
        if filenames_a else (None, None)
    )
    portfolio_b, metrics_b = (
        run_backtest(_config.base_url, filenames_b, start_date, end_date, _config.df)
        if filenames_b else (None, None)
    )

    if portfolio_a is None and portfolio_b is None:
        return empty_chart, hidden, '', 'No data available for the selected period.'

    fig = go.Figure()
    if portfolio_a is not None:
        fig.add_trace(go.Scatter(
            x=portfolio_a.index, y=portfolio_a.round(2),
            name='Basket A', line=dict(color='#1a56db', width=2),
        ))
    if portfolio_b is not None:
        fig.add_trace(go.Scatter(
            x=portfolio_b.index, y=portfolio_b.round(2),
            name='Basket B', line=dict(color='#c0392b', width=2),
        ))

    months_shown = max(
        len(portfolio_a) if portfolio_a is not None else 0,
        len(portfolio_b) if portfolio_b is not None else 0,
    )
    d0_label = start_date.strftime('%b %Y')
    d1_label = end_date.strftime('%b %Y')
    fig.update_layout(
        title=f'Portfolio Value  {d0_label} – {d1_label}  ({months_shown} months, 1,000 €/month)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (€)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=8, r=8, t=48, b=8),
    )

    metrics_div = _metrics_table(metrics_a, metrics_b)
    status = f'Backtest complete – {d0_label} to {d1_label} ({months_shown} months).'
    _config.log.info("Backtest completed: %d months, A=%s, B=%s",
                     months_shown, len(filenames_a), len(filenames_b))
    return fig, visible, metrics_div, status
