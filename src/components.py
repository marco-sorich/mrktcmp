from dash import html, dcc

import src.config as _config
from src.styles import _BASKET_ITEM_STYLE, _BTN_SMALL, _METRIC_TABLE_STYLE


def _basket_ui(basket_id):
    label = 'A' if basket_id == 'a' else 'B'
    return html.Div([
        html.H3(f'Basket {label}', style={'marginBottom': '8px'}),
        dcc.RadioItems(
            _config.assetsClasses,
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
            html.Span(f"{item['symbol']} — {item['name']}",
                      style={'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
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
            html.Th('Metric',   style={'textAlign': 'left',  'padding': '4px 8px', 'background': '#f0f0f0'}),
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
