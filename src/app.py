# ---------------------------------------------------------------------------
# app.py – Dash application entry point
#
# This file is intentionally thin. Its only jobs are:
#   1. Create the Dash application object.
#   2. Set the page layout by calling create_layout().
#   3. Import the callbacks package so all @callback decorators execute and
#      register their Input/Output wiring with Dash before the server starts.
#   4. Expose the underlying Flask WSGI object as `server` so production
#      servers (gunicorn, uWSGI) can find it.
#
# All business logic, UI construction, and data loading live in dedicated
# modules:
#
#   config.py             – logging, env vars, master data (df, base_url)
#   styles.py             – shared CSS-in-Python style dicts
#   utils.py              – log_time decorator
#   components.py         – reusable UI component builders
#   layout.py             – create_layout() factory
#   callbacks/chart.py    – Market Data tab callbacks
#   callbacks/backtesting.py – Backtesting tab callbacks
#   backtest.py           – DCA simulation engine
#
# Running in development:
#   python src/app.py
#
# Running in production (example with gunicorn):
#   gunicorn "src.app:server" --bind 0.0.0.0:8050
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

# os: operating-system interface. Used here to read the DASH_DEBUG
# environment variable that controls the dev-tools overlay.
import os

# time: provides time.time() for measuring startup duration.
import time

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# dash: the main Dash package. dash.Dash is the application class that wraps
# Flask and translates the Python component tree into a React web app.
import dash

# dash_bootstrap_components: provides Bootstrap-themed Dash components and,
# more importantly here, the CDN links for Bootstrap CSS (dbc.themes.BOOTSTRAP)
# and Font Awesome icons (dbc.icons.FONT_AWESOME) that are loaded via
# external_stylesheets.
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

# config is imported for its side effects (loading .env, setting up logging,
# loading master.parquet) and to access the shared logger for the startup
# timing log line below.
import src.config as _config

# create_layout builds and returns the complete Dash component tree. It is
# called once here so the layout is constructed after the app object exists
# and after the catalogue data has been loaded by config.py.
from src.layout import create_layout

# Importing src.callbacks triggers __init__.py which in turn imports
# chart.py and backtesting.py. Each of those modules defines callback
# functions decorated with @callback; the decorator executes at import time
# and registers the callback's Input/Output wiring with Dash globally.
# The 'noqa: F401' comment suppresses the "imported but unused" linter
# warning – the side effect is the entire purpose of this import.
import src.callbacks  # noqa: F401

# Importing src.strategies triggers __init__.py which imports every plugin
# module; each module's @register decorator fires at import time, populating
# the strategy registry before the first user interaction.
import src.strategies  # noqa: F401

# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

# Record time before creating the app so we can log total startup duration.
_t0 = time.time()

# dash.Dash(__name__) creates the web application.
# __name__ tells Dash the name of the Python module so it can locate static
# files (CSS, images) relative to this file's directory. Dash automatically
# serves any files placed under src/assets/ (e.g. src/assets/layout.css).
#
# meta_tags adds an HTML <meta> viewport tag so the page scales correctly on
# mobile devices (otherwise the browser would zoom out to show a desktop view).
#
# external_stylesheets loads Bootstrap and Font Awesome from a CDN so we do
# not have to bundle those large files ourselves.
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
)

# Enable Dash's hot-reload and debug overlay only when DASH_DEBUG=true is set
# in the environment. In production this should be false to avoid exposing
# internal error details to users.
app.enable_dev_tools(debug=os.getenv("DASH_DEBUG", "false").lower() == "true")

# Set the page layout. create_layout() reads assetsClasses from config so it
# must be called after config.py has run (which happens at the import above).
app.layout = create_layout()

_config.log.debug(f'App initialization time: {(time.time() - _t0)*1000:,.2f}ms')

# app.server is the underlying Flask WSGI application that Dash wraps.
# Gunicorn and other WSGI servers need a reference to this object so they
# can serve the app in production. The name 'server' is a widely-used
# convention for Dash apps.
server = app.server

# ---------------------------------------------------------------------------
# Entry point – run the development server when executed directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # app.run starts Dash's built-in Flask development server.
    # debug=True enables hot-reload (auto-restarts on file changes) and shows
    # an error overlay in the browser for Python exceptions.
    # In production, use gunicorn with the 'server' variable instead.
    app.run(debug=True)
