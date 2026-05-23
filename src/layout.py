from dash import html, dcc

import src.config as _config
from src.components import _basket_ui


def create_layout():
    return html.Div([
        html.H1("mrktcmp _ markets compare"),
        dcc.Tabs(id='main-tabs', value='tab-chart', children=[

            dcc.Tab(label='Market Data', value='tab-chart', children=[
                html.Div([
                    dcc.RadioItems(_config.assetsClasses, id='assetclasses-type', inline=True),
                    dcc.Dropdown(id='asset-type', placeholder='Type to search…'),
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
                    html.Label('Analysis period:', style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    html.Div(
                        dcc.RangeSlider(
                            id='bt-date-range',
                            min=0, max=1, step=1, value=[0, 1],
                            marks={},
                            allowCross=False,
                            updatemode='drag',
                            disabled=True,
                            allow_direct_input=False,
                        ),
                        style={'width': '98%', 'margin': '0 auto'},
                    ),
                    html.Div(
                        id='bt-date-display',
                        children='Add assets to a basket to see the available date range.',
                        style={'color': '#666', 'fontSize': '13px', 'marginTop': '6px'},
                    ),
                    dcc.Store(id='bt-date-store', data=[]),
                ], style={'marginTop': '20px', 'marginBottom': '8px'}),

                html.Button(
                    '▶ Start Backtest',
                    id='bt-run',
                    n_clicks=0,
                    style={'padding': '8px 20px', 'fontSize': '14px', 'cursor': 'pointer',
                           'marginBottom': '16px'},
                ),
                html.Div(id='bt-status', style={'color': '#888', 'fontSize': '13px', 'marginBottom': '8px'}),

                html.Div([
                    html.Div(
                        dcc.Graph(id='bt-chart', style={'width': '100%', 'display': 'none'}),
                        className='bt-chart-wrapper',
                    ),
                    html.Div(id='bt-metrics', className='bt-metrics-wrapper'),
                ], className='results-container'),

                dcc.Store(id='bt-result-store', data={}),
            ]),
        ]),
    ], style={'maxWidth': '100%', 'padding': '0 8px', 'boxSizing': 'border-box'})
