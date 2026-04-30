import os

import dash
from dash import html, dcc
import plotly.graph_objects as go
import numpy as np
import h5py
import fsspec

url = os.getenv("DATA_URL")

# Structure of data file (HDF5 format)
# * ETFs
#   * A..Z
#      * MSCI etc.
# * Futures
# * Indices
# * Stocks

# Check if URL is provided and show warning if empty
if not url or url.strip() == "":
    warning_message = "⚠️ Warning: DATA_URL environment variable is not set."
else:
    try:
        with fsspec.open(url, "rb") as f:
            data = h5py.File(f, "r")
            # markets = list(hdf.keys())
            warning_message = f"✅ Data loaded."
    except Exception as e:
        warning_message = f"❌ Error loading data."

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the Dash app
app = dash.Dash(__name__)

# Expose the Flask server for gunicorn
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Plotly Dash Example"),
    html.P(warning_message, style={'color': 'red' if 'Warning' in warning_message or 'Error' in warning_message else 'green'}),
    html.Div([
        dcc.Graph(
            id='example-graph',
            figure=go.Figure(
                data=[go.Scatter(x=x, y=y, mode='lines', name='sin(x)')],
                layout=go.Layout(title='Simple Sine Wave')
            )
        )
    ])
])

if __name__ == '__main__':
    app.run(debug=True)