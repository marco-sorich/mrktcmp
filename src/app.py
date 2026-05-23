import os
import time

import dash
import dash_bootstrap_components as dbc

import src.config as _config
from src.layout import create_layout
import src.callbacks  # noqa: F401 – registers all callbacks

_t0 = time.time()

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
)
app.enable_dev_tools(debug=os.getenv("DASH_DEBUG", "false").lower() == "true")
app.layout = create_layout()

_config.log.debug(f'App initialization time: {(time.time() - _t0)*1000:,.2f}ms')

server = app.server

if __name__ == '__main__':
    app.run(debug=True)
