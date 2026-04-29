import dash
from dash import html, dcc
import plotly.graph_objects as go
import numpy as np

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