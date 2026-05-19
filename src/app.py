import functools
import logging
import os
import sys
import time

import dash

import io

from dash import html, dcc, Input, Output, State, callback, Patch, no_update, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Ensure backtest.py is importable both when run directly and via gunicorn.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import run_backtest  # noqa: E402

start_time = time.time()

_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.DEBUG)

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(_log_level)
# Route only INFO/DEBUG to stdout; WARNING and above go to stderr so process
# supervisors (e.g. gunicorn, systemd) can treat error output separately.
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(level=_log_level, handlers=[_stdout_handler, _stderr_handler])
log = logging.getLogger(__name__)


base_url = os.getenv("BASE_URL")

assetsClasses = []
df = None

if not base_url or base_url.strip() == "":
    log.critical("BASE_URL environment variable is not set.")
else:
    try:
        df = pd.read_parquet(f"{base_url}/master.parquet")
        df.sort_values(['asset_class', 'symbol', 'exchange'], inplace=True, ignore_index=True)
        assetsClasses = df['asset_class'].unique().tolist()
        log.info("Data loaded.")
    except Exception:
        log.exception("Failed to load master.csv from BASE_URL")

# Create the Dash app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.enable_dev_tools(debug=os.getenv("DASH_DEBUG", "false").lower() == "true")

log.debug(f'Initialization time: {(time.time() - start_time)*1000:,.2f}ms')

# Expose the Flask server for gunicorn
server = app.server

_BASKET_ITEM_STYLE = {
    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
    'padding': '4px 8px', 'marginBottom': '2px', 'background': '#f5f5f5',
    'borderRadius': '4px', 'fontSize': '13px',
}
_BTN_SMALL = {
    'padding': '2px 8px', 'fontSize': '12px', 'cursor': 'pointer',
    'border': '1px solid #ccc', 'borderRadius': '3px', 'background': 'white',
}
_METRIC_TABLE_STYLE = {'borderCollapse': 'collapse', 'width': '100%', 'fontSize': '13px'}


def _basket_ui(basket_id):
    label = 'A' if basket_id == 'a' else 'B'
    return html.Div([
        html.H3(f'Basket {label}', style={'marginBottom': '8px'}),
        dcc.RadioItems(
            assetsClasses,
            id=f'bt-assetclass-{basket_id}',
            inline=True,
            style={'marginBottom': '8px'},
        ),
        html.Div([
            dcc.Dropdown(
                id=f'bt-asset-{basket_id}',
                placeholder='Search asset…',
                disabled=True,
                style={'flex': 1},
            ),
            html.Button(
                '＋',
                id=f'bt-add-{basket_id}',
                n_clicks=0,
                style={**_BTN_SMALL, 'fontSize': '16px', 'padding': '2px 12px'},
            ),
        ], style={'display': 'flex', 'gap': '6px', 'alignItems': 'center', 'marginBottom': '8px'}),
        html.Div(id=f'bt-basket-list-{basket_id}', style={'minHeight': '32px'}),
        dcc.Store(id=f'bt-basket-store-{basket_id}', data=[]),
    ], style={'flex': 1, 'minWidth': 0})


def _render_basket_list(basket_data, basket_id):
    if not basket_data:
        return html.P('No assets', style={'color': '#aaa', 'fontStyle': 'italic', 'margin': '4px 0'})
    return html.Div([
        html.Div([
            html.Span(f"{item['symbol']} — {item['name']}", style={'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
            html.Button(
                '✕',
                id={'type': f'bt-remove-{basket_id}', 'index': item['filename']},
                n_clicks=0,
                style=_BTN_SMALL,
            ),
        ], style=_BASKET_ITEM_STYLE)
        for item in basket_data
    ])


def _metrics_table(metrics_a, metrics_b):
    if not metrics_a and not metrics_b:
        return html.P('No results.', style={'color': '#aaa'})
    keys = list((metrics_a or metrics_b).keys())
    rows = [
        html.Tr([
            html.Th('Metric', style={'textAlign': 'left', 'padding': '4px 8px', 'background': '#f0f0f0'}),
            html.Th('Basket A', style={'textAlign': 'right', 'padding': '4px 8px', 'background': '#e8f0fe'}),
            html.Th('Basket B', style={'textAlign': 'right', 'padding': '4px 8px', 'background': '#fce8e6'}),
        ])
    ] + [
        html.Tr([
            html.Td(k, style={'padding': '3px 8px', 'borderBottom': '1px solid #eee'}),
            html.Td((metrics_a or {}).get(k, '—'),
                    style={'textAlign': 'right', 'padding': '3px 8px',
                           'borderBottom': '1px solid #eee', 'color': '#1a56db'}),
            html.Td((metrics_b or {}).get(k, '—'),
                    style={'textAlign': 'right', 'padding': '3px 8px',
                           'borderBottom': '1px solid #eee', 'color': '#c0392b'}),
        ])
        for k in keys
    ]
    return html.Table(rows, style=_METRIC_TABLE_STYLE)


# Define the layout
app.layout = html.Div([
    html.H1("mrktcmp _ markets compare"),
    dcc.Tabs(id='main-tabs', value='tab-chart', children=[

        dcc.Tab(label='Market Data', value='tab-chart', children=[
            html.Div([
                dcc.RadioItems(assetsClasses, id='assetclasses-type', inline=True),
                dcc.Dropdown(id='asset-type', placeholder='Type to search…')
            ]),
            html.Div(id='asset-headline'),
            dcc.Graph(id='price-chart', style={'width': '100%'}),
            dcc.Store(id='ohlcv-data'),
        ]),

        dcc.Tab(label='Backtesting', value='tab-backtest', children=[
            html.Div([
                _basket_ui('a'),
                html.Div(style={'width': '24px'}),
                _basket_ui('b'),
            ], style={'display': 'flex', 'gap': '8px', 'marginTop': '12px'}),

            html.Div([
                html.Label('Period (years):', style={'fontWeight': 'bold', 'marginRight': '8px'}),
                dcc.Slider(
                    id='bt-years',
                    min=1, max=30, step=1, value=5,
                    marks={i: f'{i}' for i in [1, 2, 3, 5, 10, 15, 20, 25, 30]},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                ),
            ], style={'marginTop': '20px', 'marginBottom': '8px'}),

            html.Button(
                '▶ Start Backtest',
                id='bt-run',
                n_clicks=0,
                style={'padding': '8px 20px', 'fontSize': '14px', 'cursor': 'pointer',
                       'marginBottom': '16px'},
            ),

            html.Div(id='bt-status', style={'color': '#888', 'fontSize': '13px', 'marginBottom': '8px'}),
            dcc.Graph(id='bt-chart', style={'width': '100%', 'display': 'none'}),
            html.Div(id='bt-metrics', style={'marginTop': '16px'}),
            dcc.Store(id='bt-result-store', data={}),
        ]),
    ]),
], style={'maxWidth': '100%', 'padding': '0 8px', 'boxSizing': 'border-box'})


def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        log.debug(f'{func.__name__} callback time: {(time.time() - t0)*1000:,.2f}ms')
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Existing Market Data callbacks
# ---------------------------------------------------------------------------

@callback(
    Output('asset-type', 'options'),
    Output('asset-type', 'disabled'),
    Input('assetclasses-type', 'value'),
    running=[(Output('asset-type', 'disabled'), True, False)]
)
@log_time
def update_asset_class(asset_class):
    options, disabled = [], True
    if asset_class and df is not None:
        filtered = df[df['asset_class'] == asset_class].head(30)
        options = [
            {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
            for _, row in filtered.iterrows()
        ]
        disabled = False
        log.info("Asset class selected: %s", asset_class)
    return options, disabled


@callback(
    Output('asset-type', 'options', allow_duplicate=True),
    Input('asset-type', 'search_value'),
    State('assetclasses-type', 'value'),
    State('asset-type', 'value'),
    prevent_initial_call=True
)
@log_time
def update_asset_search(search_value, asset_class, current_value):
    if not asset_class or df is None:
        return []
    filtered = df[df['asset_class'] == asset_class]
    if search_value:
        sl = search_value.lower()
        sym = filtered['symbol'].str.lower()
        name = filtered['name'].str.lower()
        # Vectorised scoring keeps the loop off Python; priority: exact symbol (0) →
        # symbol prefix (1) → name prefix (2) → symbol contains (3) → name contains (4).
        score = np.select(
            [sym == sl,
             sym.str.startswith(sl, na=False),
             name.str.startswith(sl, na=False),
             sym.str.contains(sl, na=False),
             name.str.contains(sl, na=False)],
            [0, 1, 2, 3, 4],
            default=99
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
    # Preserve the currently selected asset even when it falls outside the search
    # results so the dropdown does not silently lose its value on each keystroke.
    if current_value and not any(o['value'] == current_value for o in options):
        sel = df[df['filename'] == current_value]
        if not sel.empty:
            row = sel.iloc[0]
            options.append({'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': current_value})
    return options


@callback(
    Output('price-chart', 'figure'),
    Output('asset-headline', 'children'),
    Output('ohlcv-data', 'data'),
    Input('asset-type', 'value')
)
@log_time
def update_chart(filename):
    empty = go.Figure(), "", None
    if not filename or not base_url or df is None:
        return empty
    try:
        row = df[df['filename'] == filename].iloc[0]
        headline = [
            html.H2(row['name'], style={'marginBottom': '2px'}),
            html.P(f"{row['exchange']} — {row['country']}", style={'marginTop': '0', 'color': 'gray'})
        ]

        ohlcv = pd.read_parquet(f"{base_url}/{filename}")
        ohlcv = ohlcv[ohlcv.index >= pd.Timestamp.now(tz=ohlcv.index.tz) - pd.DateOffset(years=10)]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['Open'],
            high=ohlcv['High'],
            low=ohlcv['Low'],
            close=ohlcv['Close'],
            name='Price'
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=ohlcv.index,
            y=ohlcv['Volume'],
            name='Volume',
            fill='tozeroy',
            line=dict(width=1)
        ), row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=8, b=8))

        grid_df = ohlcv.reset_index()
        grid_df.rename(columns={grid_df.columns[0]: 'Date'}, inplace=True)
        grid_df['Date'] = pd.to_datetime(grid_df['Date']).dt.strftime('%d-%b-%Y')
        for col in ('Open', 'High', 'Low', 'Close'):
            if col in grid_df.columns:
                grid_df[col] = grid_df[col].map('{:,.2f}'.format)
        if 'Volume' in grid_df.columns:
            grid_df['Volume'] = grid_df['Volume'].map('{:,}'.format)

        log.info("Data loaded from %s", filename)
        # Store only High/Low/Volume; Close is already encoded in the Candlestick trace.
        store = ohlcv[['High', 'Low', 'Volume']].to_json(date_format='iso', orient='split')
        return fig, headline, store
    except Exception:
        log.exception("Failed to load chart data for %s", filename)
        return empty


@callback(
    Output('price-chart', 'figure', allow_duplicate=True),
    Input('price-chart', 'relayoutData'),
    State('ohlcv-data', 'data'),
    prevent_initial_call=True
)
@log_time
def sync_yaxis_on_xzoom(relayout_data, ohlcv_json):
    if not relayout_data or ohlcv_json is None:
        return no_update
    if relayout_data.get('xaxis.autorange') or relayout_data.get('autosize'):
        patch = Patch()
        patch['layout']['yaxis']['autorange'] = True
        patch['layout']['yaxis2']['autorange'] = True
        return patch
    x0 = relayout_data.get('xaxis.range[0]')
    x1 = relayout_data.get('xaxis.range[1]')
    if x0 is None or x1 is None:
        return no_update
    ohlcv = pd.read_json(io.StringIO(ohlcv_json), orient='split', convert_axes=True)
    x0_ts, x1_ts = pd.Timestamp(x0), pd.Timestamp(x1)
    if ohlcv.index.tz is not None and x0_ts.tz is None:
        x0_ts = x0_ts.tz_localize('UTC').tz_convert(ohlcv.index.tz)
        x1_ts = x1_ts.tz_localize('UTC').tz_convert(ohlcv.index.tz)
    visible = ohlcv[(ohlcv.index >= x0_ts) & (ohlcv.index <= x1_ts)]
    if visible.empty:
        return no_update
    low, high = visible['Low'].min(), visible['High'].max()
    margin = (high - low) * 0.02
    patch = Patch()
    patch['layout']['yaxis']['autorange'] = False
    patch['layout']['yaxis']['range'] = [low - margin, high + margin]
    patch['layout']['yaxis2']['autorange'] = False
    patch['layout']['yaxis2']['range'] = [0, visible['Volume'].max() * 1.1]
    return patch


# ---------------------------------------------------------------------------
# Backtesting – asset class selectors
# ---------------------------------------------------------------------------

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
    if not asset_class or df is None:
        return [], True
    filtered = df[df['asset_class'] == asset_class].head(30)
    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]
    return options, False


# ---------------------------------------------------------------------------
# Backtesting – asset search
# ---------------------------------------------------------------------------

@callback(
    Output('bt-asset-a', 'options', allow_duplicate=True),
    Input('bt-asset-a', 'search_value'),
    State('bt-assetclass-a', 'value'),
    State('bt-asset-a', 'value'),
    prevent_initial_call=True
)
@log_time
def bt_search_a(search_value, asset_class, current_value):
    return _bt_asset_search(search_value, asset_class, current_value)


@callback(
    Output('bt-asset-b', 'options', allow_duplicate=True),
    Input('bt-asset-b', 'search_value'),
    State('bt-assetclass-b', 'value'),
    State('bt-asset-b', 'value'),
    prevent_initial_call=True
)
@log_time
def bt_search_b(search_value, asset_class, current_value):
    return _bt_asset_search(search_value, asset_class, current_value)


def _bt_asset_search(search_value, asset_class, current_value):
    if not asset_class or df is None:
        return []
    filtered = df[df['asset_class'] == asset_class]
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
            default=99
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
        sel = df[df['filename'] == current_value]
        if not sel.empty:
            row = sel.iloc[0]
            options.append({'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': current_value})
    return options


# ---------------------------------------------------------------------------
# Backtesting – basket management (add / remove)
# ---------------------------------------------------------------------------

@callback(
    Output('bt-basket-store-a', 'data'),
    Output('bt-basket-list-a', 'children'),
    Input('bt-add-a', 'n_clicks'),
    Input({'type': 'bt-remove-a', 'index': ALL}, 'n_clicks'),
    State('bt-asset-a', 'value'),
    State('bt-basket-store-a', 'data'),
    prevent_initial_call=True
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
    prevent_initial_call=True
)
@log_time
def manage_basket_b(add_clicks, remove_clicks, selected_asset, basket_data):
    return _manage_basket('b', remove_clicks, selected_asset, basket_data)


def _manage_basket(basket_id, remove_clicks, selected_asset, basket_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    triggered_id = ctx.triggered_id
    # .get('value', 0) or 0 handles None, which Dash emits for newly rendered components.
    triggered_value = ctx.triggered[0].get('value', 0) or 0
    basket = list(basket_data or [])

    if isinstance(triggered_id, dict) and triggered_id.get('type') == f'bt-remove-{basket_id}':
        # Newly rendered remove buttons fire the ALL pattern with n_clicks=0; skip those.
        if triggered_value > 0:
            filename = triggered_id['index']
            basket = [item for item in basket if item['filename'] != filename]
    elif triggered_id == f'bt-add-{basket_id}' and selected_asset and df is not None:
        if not any(item['filename'] == selected_asset for item in basket):
            meta = df[df['filename'] == selected_asset]
            if not meta.empty:
                row = meta.iloc[0]
                basket.append({'filename': selected_asset, 'symbol': row['symbol'], 'name': row['name']})
    else:
        return no_update, no_update

    return basket, _render_basket_list(basket, basket_id)


# ---------------------------------------------------------------------------
# Backtesting – run
# ---------------------------------------------------------------------------

@callback(
    Output('bt-chart', 'figure'),
    Output('bt-chart', 'style'),
    Output('bt-metrics', 'children'),
    Output('bt-status', 'children'),
    Input('bt-run', 'n_clicks'),
    State('bt-basket-store-a', 'data'),
    State('bt-basket-store-b', 'data'),
    State('bt-years', 'value'),
    prevent_initial_call=True
)
@log_time
def run_backtest_callback(n_clicks, basket_a, basket_b, years):
    empty_chart = go.Figure()
    hidden = {'width': '100%', 'display': 'none'}
    visible = {'width': '100%', 'display': 'block'}

    if not basket_a and not basket_b:
        return empty_chart, hidden, '', 'Please fill at least one basket.'

    if not base_url or df is None:
        return empty_chart, hidden, '', 'No data source available.'

    filenames_a = [item['filename'] for item in (basket_a or [])]
    filenames_b = [item['filename'] for item in (basket_b or [])]

    portfolio_a, metrics_a = run_backtest(base_url, filenames_a, years, df) if filenames_a else (None, None)
    portfolio_b, metrics_b = run_backtest(base_url, filenames_b, years, df) if filenames_b else (None, None)

    if portfolio_a is None and portfolio_b is None:
        return empty_chart, hidden, '', 'No data available for the selected period.'

    fig = go.Figure()
    if portfolio_a is not None:
        fig.add_trace(go.Scatter(
            x=portfolio_a.index,
            y=portfolio_a.round(2),
            name='Basket A',
            line=dict(color='#1a56db', width=2),
        ))
    if portfolio_b is not None:
        fig.add_trace(go.Scatter(
            x=portfolio_b.index,
            y=portfolio_b.round(2),
            name='Basket B',
            line=dict(color='#c0392b', width=2),
        ))

    # Baskets may cover different date ranges; title reflects the longer one.
    months_shown = max(
        len(portfolio_a) if portfolio_a is not None else 0,
        len(portfolio_b) if portfolio_b is not None else 0,
    )
    actual_years = months_shown / 12

    fig.update_layout(
        title=f'Portfolio Value ({actual_years:.1f} years, {months_shown} months, 1,000 €/month)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (€)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=8, r=8, t=48, b=8),
    )

    metrics_div = _metrics_table(metrics_a, metrics_b)
    status = f'Backtest complete – {actual_years:.1f} years simulated.'
    log.info("Backtest completed: %d months, A=%s, B=%s",
             months_shown, len(filenames_a), len(filenames_b))
    return fig, visible, metrics_div, status


if __name__ == '__main__':
    app.run(debug=True)
