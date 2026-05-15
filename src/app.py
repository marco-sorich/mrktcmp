import os

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_ag_grid as dag
import plotly.graph_objects as go
import numpy as np
import pandas as pd

base_url = os.getenv("BASE_URL")

assetsClasses = []
df = None

if not base_url or base_url.strip() == "":
    warning_message = "⚠️ Warning: BASE_URL environment variable is not set."
else:
    try:
        df = pd.read_csv(f"{base_url}/master.csv",
                         dtype={
                             "asset_class":"string",
                             "symbol":"string",
                             "interval":"string",
                             "name":"string",
                             "exchange":"string",
                             "country":"string",
                             "category":"string",
                             "first_date":"string",
                             "last_date":"string",
                             "filename":"string"})
        df.sort_values(['asset_class', 'symbol', 'exchange'], inplace=True, ignore_index=True)
        assetsClasses = df['asset_class'].unique().tolist()
        warning_message = "✅ Data loaded."
    except Exception as e:
        warning_message = f"❌ Error loading data."

# Create the Dash app
app = dash.Dash(__name__)

# Expose the Flask server for gunicorn
server = app.server




# Define the layout
app.layout = html.Div([
    html.H1("Plotly Dash Example"),
    html.P(warning_message, style={'color': 'red' if 'Warning' in warning_message or 'Error' in warning_message else 'green'}),
    html.Div([
        dcc.RadioItems(assetsClasses, id='assetclasses-type', inline=True),
        dcc.Dropdown(id='asset-type', placeholder='Type to search…')
    ]),
    html.Div(id='asset-headline'),
    dag.AgGrid(
        id='ohlcv-grid',
        columnDefs=[],
        rowData=[],
        style={'height': '200px'}
    ),
    dcc.Graph(id='price-chart')
])




@callback(
    Output('asset-type', 'options'),
    Output('asset-type', 'disabled'),
    Input('assetclasses-type', 'value'),
    Input('asset-type', 'search_value'),
    State('asset-type', 'value'),
    running=[
        (Output('asset-type', 'disabled'), True, False)
    ]
)
def update_asset_type_options(asset_class, search_value, current_value):
    if not asset_class or df is None:
        return [], True
    filtered = df[df['asset_class'] == asset_class]
    if search_value:
        mask = (
            filtered['symbol'].str.contains(search_value, case=False, na=False) |
            filtered['name'].str.contains(search_value, case=False, na=False)
        )
        filtered = filtered[mask].head(10)
    else:
        filtered = filtered.head(10)
    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]
    if current_value and not any(o['value'] == current_value for o in options):
        sel = df[df['filename'] == current_value]
        if not sel.empty:
            row = sel.iloc[0]
            options.append({'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': current_value})
    return options, False


@callback(
    Output('price-chart', 'figure'),
    Output('ohlcv-grid', 'rowData'),
    Output('ohlcv-grid', 'columnDefs'),
    Output('asset-headline', 'children'),
    Input('asset-type', 'value')
)
def update_chart(filename):
    empty = go.Figure(), [], [], ""
    if not filename or not base_url or df is None:
        return empty
    try:
        row = df[df['filename'] == filename].iloc[0]
        headline = [
            html.H2(row['name'], style={'marginBottom': '2px'}),
            html.P(f"{row['exchange']} — {row['country']}", style={'marginTop': '0', 'color': 'gray'})
        ]

        ohlcv = pd.read_parquet(f"{base_url}/{filename}")

        fig = go.Figure(data=[go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['Open'],
            high=ohlcv['High'],
            low=ohlcv['Low'],
            close=ohlcv['Close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)

        grid_df = ohlcv.reset_index()
        grid_df.rename(columns={grid_df.columns[0]: 'Date'}, inplace=True)
        grid_df['Date'] = grid_df['Date'].astype(str)

        col_defs = [{'field': col} for col in grid_df.columns]
        row_data = grid_df.to_dict('records')

        return fig, row_data, col_defs, headline
    except Exception:
        return empty


if __name__ == '__main__':
    app.run(debug=True)
