import os

import dash
from dash import html, dcc, Input, Output, callback
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
        dcc.RadioItems(assetsClasses, id='assetclasses-type'),
        dcc.Dropdown(id='asset-type')
    ])
])




@callback(
    Output('asset-type', 'options'),
    Input('assetclasses-type', 'value')
)
def update_asset_type_options(value):
    if not value or df is None:
        return []
    return df[df['asset_class'] == value]['symbol'].tolist()


if __name__ == '__main__':
    app.run(debug=True)