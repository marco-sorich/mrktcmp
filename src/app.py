import functools
import logging
import os
import sys
import time

import dash

from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

start_time = time.time()

_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.DEBUG)

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(_log_level)
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
app = dash.Dash(__name__)

log.debug(f'Initialization time: {(time.time() - start_time)*1000:,.2f}ms')

# Expose the Flask server for gunicorn
server = app.server


# Define the layout
app.layout = html.Div([
    html.H1("mrktcmp _ markets compare"),
    html.Div([
        dcc.RadioItems(assetsClasses, id='assetclasses-type', inline=True),
        dcc.Dropdown(id='asset-type', placeholder='Type to search…')
    ]),
    html.Div(id='asset-headline'),
    dcc.Graph(id='price-chart')
])


def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        log.debug(f'{func.__name__} callback time: {(time.time() - t0)*1000:,.2f}ms')
        return result
    return wrapper


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


@callback(
    Output('price-chart', 'figure'),
    Output('asset-headline', 'children'),
    Input('asset-type', 'value')
)
@log_time
def update_chart(filename):
    empty = go.Figure(), ""
    if not filename or not base_url or df is None:
        return empty
    try:
        row = df[df['filename'] == filename].iloc[0]
        headline = [
            html.H2(row['name'], style={'marginBottom': '2px'}),
            html.P(f"{row['exchange']} — {row['country']}", style={'marginTop': '0', 'color': 'gray'})
        ]

        ohlcv = pd.read_parquet(f"{base_url}/{filename}")

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
        fig.add_trace(go.Scatter(
            x=ohlcv.index,
            y=ohlcv['Volume'],
            name='Volume',
            fill='tozeroy',
            line=dict(width=1)
        ), row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(showlegend=False)

        grid_df = ohlcv.reset_index()
        grid_df.rename(columns={grid_df.columns[0]: 'Date'}, inplace=True)
        grid_df['Date'] = pd.to_datetime(grid_df['Date']).dt.strftime('%d-%b-%Y')
        for col in ('Open', 'High', 'Low', 'Close'):
            if col in grid_df.columns:
                grid_df[col] = grid_df[col].map('{:,.2f}'.format)
        if 'Volume' in grid_df.columns:
            grid_df['Volume'] = grid_df['Volume'].map('{:,}'.format)

        log.info("Data loaded from %s", filename)

        return fig, headline
    except Exception:
        log.exception("Failed to load chart data for %s", filename)
        return empty


if __name__ == '__main__':
    app.run(debug=True)
