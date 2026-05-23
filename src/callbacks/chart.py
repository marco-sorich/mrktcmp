import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, Input, Output, State, callback, Patch, no_update

import src.config as _config
from src.utils import log_time


@callback(
    Output('asset-type', 'options'),
    Output('asset-type', 'disabled'),
    Input('assetclasses-type', 'value'),
    running=[(Output('asset-type', 'disabled'), True, False)],
)
@log_time
def update_asset_class(asset_class):
    options, disabled = [], True
    if asset_class and _config.df is not None:
        filtered = _config.df[_config.df['asset_class'] == asset_class].head(200)
        options = [
            {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
            for _, row in filtered.iterrows()
        ]
        disabled = False
        _config.log.info("Asset class selected: %s", asset_class)
    return options, disabled


@callback(
    Output('asset-type', 'options', allow_duplicate=True),
    Input('asset-type', 'search_value'),
    State('assetclasses-type', 'value'),
    State('asset-type', 'value'),
    prevent_initial_call=True,
)
@log_time
def update_asset_search(search_value, asset_class, current_value):
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
    Output('price-chart', 'figure'),
    Output('asset-headline', 'children'),
    Output('ohlcv-data', 'data'),
    Input('asset-type', 'value'),
)
@log_time
def update_chart(filename):
    empty = go.Figure(), "", None

    if not filename or not _config.base_url or _config.df is None:
        return empty

    try:
        row = _config.df[_config.df['filename'] == filename].iloc[0]
        headline = [
            html.H2(row['name'], style={'marginBottom': '2px'}),
            html.P(f"{row['exchange']} — {row['country']}", style={'marginTop': '0', 'color': 'gray'}),
        ]

        ohlcv = pd.read_parquet(f"{_config.base_url}/{filename}")
        ohlcv = ohlcv[ohlcv.index >= pd.Timestamp.now(tz=ohlcv.index.tz) - pd.DateOffset(years=10)]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['Open'], high=ohlcv['High'],
            low=ohlcv['Low'], close=ohlcv['Close'],
            name='Price',
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=ohlcv.index, y=ohlcv['Volume'],
            name='Volume', fill='tozeroy', line=dict(width=1),
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

        _config.log.info("Data loaded from %s", filename)
        store = ohlcv[['High', 'Low', 'Volume']].to_json(date_format='iso', orient='split')
        return fig, headline, store

    except Exception:
        _config.log.exception("Failed to load chart data for %s", filename)
        return empty


@callback(
    Output('price-chart', 'figure', allow_duplicate=True),
    Input('price-chart', 'relayoutData'),
    State('ohlcv-data', 'data'),
    prevent_initial_call=True,
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
