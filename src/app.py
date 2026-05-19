# ---------------------------------------------------------------------------
# app.py – Dash web application for market comparison and DCA backtesting
#
# Dash is a Python framework for building interactive web applications.
# You write the entire UI in Python (no HTML/CSS/JavaScript required) and
# Dash translates it into a React/Flask web app automatically.
#
# The core idea in Dash is the *callback*: a plain Python function decorated
# with @callback that Dash calls automatically whenever a component's
# property changes. Each callback declares:
#   • Input  – which property change fires the callback
#   • Output – which property the callback's return value is written to
#   • State  – extra data read at fire-time but that does NOT trigger the
#              callback on its own
#
# This file has two major sections:
#   1. Market Data tab – price chart + volume for a single selected asset.
#   2. Backtesting tab – DCA simulation for two user-defined asset baskets.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

# functools.wraps: preserves the original function's name, docstring, and
# attributes when a decorator wraps it. Without this, the decorator would
# shadow the function's identity (e.g. func.__name__ would return 'wrapper').
import functools

# logging: Python's built-in structured log system. Preferred over print()
# because it supports log levels (DEBUG/INFO/WARNING/ERROR/CRITICAL) and
# lets you control output format and destination in one place.
import logging

# os: operating-system interface. Used here to read environment variables
# (os.getenv) and to resolve file paths (os.path.dirname / abspath).
import os

# sys: Python runtime interface. Used to manipulate sys.path (module search
# list) so that a sibling module (backtest.py) can always be imported, and
# to attach log handlers to sys.stdout / sys.stderr.
import sys

# time: provides time.time() which returns the current Unix timestamp as a
# float (seconds since 1970-01-01 00:00:00 UTC). Used here to measure how
# long callbacks take.
import time

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# dash: the main Dash package. Importing it here gives us dash.Dash (the app
# class), dash.callback_context (info about which Input fired), and utilities
# like no_update and Patch (described later).
import dash

# io.StringIO: turns a plain string into a file-like object so libraries that
# expect to read from a file can work with in-memory strings instead. Used
# when deserialising OHLCV data that was JSON-encoded into a dcc.Store.
import io

# Dash component modules and helpers:
#   html   – wrappers for every standard HTML tag (html.Div, html.H1, etc.)
#   dcc    – "Dash Core Components": interactive widgets like Dropdown,
#            Slider, Graph, Tabs, RadioItems, and Store.
#   Input  – marks a callback argument as a trigger (fires the callback).
#   Output – marks a callback return value as a target to write to.
#   State  – marks a callback argument that is read but does NOT trigger.
#   callback – the decorator that registers a function as a Dash callback.
#   Patch  – a lightweight "partial update" object: instead of returning the
#            whole figure, send only the changed parts. Saves bandwidth.
#   no_update – a sentinel value returned from a callback to tell Dash
#               "leave this Output exactly as it is; do not overwrite it".
#   ALL    – a pattern-match wildcard used in Input/Output/State to target
#            ALL components whose id dict shares a given 'type' key. Used
#            for the dynamically rendered remove-buttons in each basket.
from dash import html, dcc, Input, Output, State, callback, Patch, no_update, ALL

# Plotly graph objects: the low-level chart primitives. go.Figure is a
# container; go.Candlestick, go.Scatter, go.Scattergl are trace types that
# describe what to draw (candles, lines, etc.).
import plotly.graph_objects as go

# make_subplots: creates a Figure pre-configured with multiple subplot areas
# (rows/columns). Used to stack the price chart above the volume chart with
# a shared x-axis so zooming on one panel zooms the other.
from plotly.subplots import make_subplots

# numpy: fast numerical array library. Used here for np.select (vectorised
# conditional scoring) and np.sqrt (for annualising standard deviation).
import numpy as np

# pandas: tabular data library. DataFrame = a 2-D table with labelled rows
# and columns. Series = a single labelled column. Used throughout for all
# data wrangling.
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure backtest.py is importable regardless of how the app is launched
# ---------------------------------------------------------------------------

# When Python imports a module it looks in sys.path (a list of directories).
# If the app is started from the project root via `python src/app.py`, Python
# adds src/ to sys.path automatically. But when gunicorn (a production WSGI
# server) imports the module as a package (e.g. `src.app`), it may not add
# src/ to sys.path. Prepending the directory that contains this file (src/)
# ensures backtest.py is found in both scenarios.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# noqa: E402 – PEP 8 / flake8 require all imports at the top of the file.
# This import must come after the sys.path.insert above, so we suppress the
# "module level import not at top of file" lint warning with the noqa comment.
from backtest import run_backtest, get_common_date_range  # noqa: E402

# ---------------------------------------------------------------------------
# Startup timer – lets us log how long initialisation takes
# ---------------------------------------------------------------------------

# Record the wall-clock time before any initialisation work begins.
# We subtract this later (time.time() - start_time) to get elapsed seconds.
start_time = time.time()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Read the desired log level from the LOG_LEVEL environment variable.
# os.getenv returns None if the variable is not set; the second argument is
# the default. getattr(logging, 'INFO', logging.DEBUG) converts the string
# 'INFO' to the integer constant logging.INFO (= 20). If the string is
# unrecognised, it falls back to logging.DEBUG (= 10) so we still see logs.
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.DEBUG)

# Create a handler that writes to standard output (the terminal's normal
# output stream). This is what you see in the console when you run the app.
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(_log_level)

# Add a filter so that only DEBUG and INFO messages go to stdout.
# WARNING, ERROR, and CRITICAL messages are more serious and are sent to
# stderr (the error output stream) separately below. This matters in
# production: process supervisors like gunicorn and systemd capture stdout
# and stderr separately, so operators can filter error logs without noise.
# A filter is a callable that returns True to keep the record, False to drop.
# r.levelno < logging.WARNING means: keep only records below WARNING level.
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

# Create a second handler for stderr (standard error). Only WARNING and above
# are routed here. This ensures serious messages always reach operators even
# if stdout is piped away or suppressed.
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

# Apply both handlers globally. logging.basicConfig configures the root
# logger (the parent of all named loggers). level sets the minimum level
# that will pass through the root logger before reaching any handler.
logging.basicConfig(level=_log_level, handlers=[_stdout_handler, _stderr_handler])

# Create a named logger for this module. __name__ resolves to 'app' (or
# 'src.app' when imported as a package). Using a named logger makes it easy
# to identify which module produced each log message.
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load master asset catalogue from BASE_URL
# ---------------------------------------------------------------------------

# BASE_URL is an environment variable that points to the root URL (or path)
# where the parquet data files are stored. Example:
#   BASE_URL=https://example.com/data   → files at BASE_URL/master.parquet
#   BASE_URL=/mnt/data                  → files at /mnt/data/master.parquet
base_url = os.getenv("BASE_URL")

# These two variables hold the loaded data. They are module-level (global)
# so every callback can read them without passing them as arguments.
# assetsClasses: list of distinct asset class strings, e.g. ['stocks','crypto']
# df:            the master metadata DataFrame – one row per asset, columns
#                include: asset_class, symbol, name, filename, exchange,
#                country, interval.
assetsClasses = []
df = None

if not base_url or base_url.strip() == "":
    # Log at CRITICAL because the app cannot function without a data source.
    log.critical("BASE_URL environment variable is not set.")
else:
    try:
        # master.parquet is the catalogue file: one row per available asset.
        # pd.read_parquet reads the binary Parquet format directly into a
        # DataFrame; it is much faster than CSV for large tables.
        df = pd.read_parquet(f"{base_url}/master.parquet")

        # Sort the catalogue so dropdowns are presented alphabetically by
        # asset class, then symbol, then exchange. inplace=True modifies the
        # DataFrame in place (no copy). ignore_index=True resets the row
        # numbers to 0, 1, 2, … after sorting.
        df.sort_values(['asset_class', 'symbol', 'exchange'], inplace=True, ignore_index=True)

        # .unique() returns an array of distinct values in 'asset_class'.
        # .tolist() converts it from a numpy array to a plain Python list,
        # which is what Dash RadioItems / Dropdown options expect.
        assetsClasses = df['asset_class'].unique().tolist()
        log.info("Data loaded.")
    except Exception:
        # log.exception automatically includes the full traceback so we can
        # diagnose the problem without adding extra debug code.
        log.exception("Failed to load master.csv from BASE_URL")

# ---------------------------------------------------------------------------
# Create the Dash application
# ---------------------------------------------------------------------------

# dash.Dash(__name__) creates the web application.
# __name__ tells Dash the name of the Python module so it can locate static
# files (CSS, images) relative to this file.
# meta_tags adds an HTML <meta> viewport tag so the page scales correctly on
# mobile devices (otherwise the browser would zoom out to show a desktop view).
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Enable Dash's hot-reload and debug overlay only when DASH_DEBUG=true is set
# in the environment. In production this should be false to avoid exposing
# internal error details.
app.enable_dev_tools(debug=os.getenv("DASH_DEBUG", "false").lower() == "true")

# Log how many milliseconds start-up took. *1000 converts seconds to ms.
# :,.2f formats the number with a thousands separator (,) and 2 decimal places.
log.debug(f'Initialization time: {(time.time() - start_time)*1000:,.2f}ms')

# app.server is the underlying Flask WSGI application that Dash wraps.
# Gunicorn (and other WSGI servers) need a reference to this object so they
# can serve the app in production. The name 'server' is a convention.
server = app.server

# ---------------------------------------------------------------------------
# Shared style dictionaries
# ---------------------------------------------------------------------------
# In Dash, styles are plain Python dicts using camelCase CSS property names
# (e.g. CSS 'background-color' → Python 'backgroundColor'). Defining them
# once as module-level constants avoids repetition and makes future changes
# (e.g. tweaking colours) a one-line edit.

# _BASKET_ITEM_STYLE styles each row in the asset basket list.
#   display: flex         – activates Flexbox layout so children sit side by
#                          side on a horizontal line.
#   alignItems: center    – vertically centres children inside the flex row.
#   justifyContent: space-between – pushes the label to the left edge and
#                          the remove button to the right edge.
#   padding / marginBottom – inner spacing and gap between rows.
#   background / borderRadius / fontSize – cosmetic look.
_BASKET_ITEM_STYLE = {
    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
    'padding': '4px 8px', 'marginBottom': '2px', 'background': '#f5f5f5',
    'borderRadius': '4px', 'fontSize': '13px',
}

# _BTN_SMALL applies to the small "＋" and "✕" buttons.
#   padding / fontSize – make the button compact.
#   cursor: pointer     – shows the hand cursor on hover (UX convention for
#                         clickable elements that are not standard <a> links).
#   border / borderRadius / background – minimal border with rounded corners.
_BTN_SMALL = {
    'padding': '2px 8px', 'fontSize': '12px', 'cursor': 'pointer',
    'border': '1px solid #ccc', 'borderRadius': '3px', 'background': 'white',
}

# _METRIC_TABLE_STYLE styles the performance metrics comparison table.
#   borderCollapse: collapse – removes the double border between adjacent cells
#                             (normally HTML tables have a gap between cells).
#   width: 100%             – stretch the table to fill its container.
#   fontSize: 13px          – slightly smaller than body text for compactness.
_METRIC_TABLE_STYLE = {'borderCollapse': 'collapse', 'width': '100%', 'fontSize': '13px'}


# ---------------------------------------------------------------------------
# Helper: build the UI block for one asset basket
# ---------------------------------------------------------------------------

def _basket_ui(basket_id):
    """Return the complete HTML/component tree for a single basket panel.

    Parameters
    ----------
    basket_id : str – either 'a' or 'b'.  Used to build unique component IDs
                      (e.g. 'bt-assetclass-a', 'bt-add-b') so Dash can
                      distinguish the two baskets' components.

    Returns
    -------
    html.Div containing all controls for one basket (heading, asset-class
    radio, search dropdown, add button, basket list, and hidden data store).
    """
    # Map basket_id to a human-readable label for the heading.
    label = 'A' if basket_id == 'a' else 'B'

    # Outer panel style: flex:1 lets this panel grow equally with its sibling
    # basket panel in their shared flex row. minWidth:0 prevents a flex child
    # from overflowing when it contains long text.
    return html.Div([

        # Section heading displayed above the basket controls.
        html.H3(f'Basket {label}', style={'marginBottom': '8px'}),

        # dcc.RadioItems renders a group of radio buttons (mutually exclusive
        # choices). Here it lists the asset classes loaded at startup so the
        # user can narrow the search dropdown to one class (e.g. 'stocks').
        # inline=True puts the radio buttons on a single horizontal line.
        dcc.RadioItems(
            assetsClasses,               # list of option labels and values
            id=f'bt-assetclass-{basket_id}',  # unique component ID for callbacks
            inline=True,
            style={'marginBottom': '8px'},
        ),

        # Flex row: display:flex places the dropdown and add-button side by
        # side. gap=6px spaces them apart. alignItems:center aligns them
        # vertically. The dropdown has flex:1 so it grows to fill the row.
        html.Div([
            # dcc.Dropdown renders a searchable select input.
            # placeholder – the greyed-out hint text shown when empty.
            # disabled=True – greyed out until the user picks an asset class.
            # style={'flex': 1} – in a flex container flex:1 means "grow to
            #                     fill all available horizontal space".
            dcc.Dropdown(
                id=f'bt-asset-{basket_id}',
                placeholder='Search asset…',
                disabled=True,
                style={'flex': 1},
            ),
            # The add button. n_clicks=0 initialises the click counter so Dash
            # has a starting value to compare against. The ** unpacks
            # _BTN_SMALL and the extra overrides merge on top.
            html.Button(
                '＋',
                id=f'bt-add-{basket_id}',
                n_clicks=0,
                style={**_BTN_SMALL, 'fontSize': '16px', 'padding': '2px 12px'},
            ),
        ], style={'display': 'flex', 'gap': '6px', 'alignItems': 'center', 'marginBottom': '8px'}),

        # Container for the list of assets currently in the basket.
        # This is updated by the manage_basket_x callback whenever the user
        # adds or removes assets. minHeight ensures the panel does not
        # collapse to zero height when the basket is empty.
        html.Div(id=f'bt-basket-list-{basket_id}', style={'minHeight': '32px'}),

        # dcc.Store is an invisible component that holds JSON data in the
        # browser's memory for the duration of the session. We use it to
        # remember the list of assets in each basket between callbacks.
        # data=[] initialises it with an empty list.
        dcc.Store(id=f'bt-basket-store-{basket_id}', data=[]),

    ], style={'flex': 1, 'minWidth': 0})


# ---------------------------------------------------------------------------
# Helper: render the visible list of assets currently in a basket
# ---------------------------------------------------------------------------

def _render_basket_list(basket_data, basket_id):
    """Build the component tree for the basket's item list.

    Each item shows the asset's symbol and name alongside a remove (✕) button.

    Parameters
    ----------
    basket_data : list of dicts, each with keys 'filename', 'symbol', 'name'.
    basket_id   : str – 'a' or 'b'. Used to build pattern-matching IDs for
                        the remove buttons.

    Returns
    -------
    html.P (empty message) or html.Div (list of rows with remove buttons).
    """
    # If the basket is empty, show a placeholder message in light grey italic.
    if not basket_data:
        return html.P('No assets', style={'color': '#aaa', 'fontStyle': 'italic', 'margin': '4px 0'})

    # Build one row per asset using a Python list comprehension.
    # A list comprehension [ expr for item in iterable ] is a concise way to
    # build a list by applying expr to each element.
    return html.Div([
        html.Div([
            # Asset label: "AAPL — Apple Inc"
            # overflow: hidden + textOverflow: ellipsis truncates long names
            # with '…' instead of overflowing the container.
            html.Span(f"{item['symbol']} — {item['name']}",
                      style={'overflow': 'hidden', 'textOverflow': 'ellipsis'}),

            # Remove button with a *pattern-matching ID*.
            # Instead of a plain string id, we use a dict id:
            #   {'type': 'bt-remove-a', 'index': 'aapl.parquet'}
            # The 'type' key groups all remove buttons for basket 'a' together.
            # The 'index' key carries the filename so the callback knows which
            # asset was removed. Dash's ALL wildcard in the callback Input
            # matches every button whose id has type == 'bt-remove-a'.
            html.Button(
                '✕',
                id={'type': f'bt-remove-{basket_id}', 'index': item['filename']},
                n_clicks=0,
                style=_BTN_SMALL,
            ),
        ], style=_BASKET_ITEM_STYLE)
        for item in basket_data
    ])


# ---------------------------------------------------------------------------
# Helper: build the side-by-side metrics comparison table
# ---------------------------------------------------------------------------

def _metrics_table(metrics_a, metrics_b):
    """Build an HTML table comparing performance metrics for both baskets.

    Parameters
    ----------
    metrics_a : dict or None – metric_name → formatted string for basket A.
    metrics_b : dict or None – same for basket B.

    Returns
    -------
    html.P (if both are None/empty) or html.Table with three columns:
    Metric | Basket A | Basket B.
    """
    # If neither basket produced metrics, show a plain message.
    if not metrics_a and not metrics_b:
        return html.P('No results.', style={'color': '#aaa'})

    # Get the list of metric keys from whichever basket has data. The 'or'
    # operator returns the first truthy (non-empty/non-None) operand.
    # This ensures we always have a key list even when one basket is empty.
    keys = list((metrics_a or metrics_b).keys())

    # Build the table header row first, then append one data row per metric.
    # html.Tr = table row, html.Th = header cell, html.Td = data cell.
    rows = [
        html.Tr([
            html.Th('Metric',   style={'textAlign': 'left',  'padding': '4px 8px', 'background': '#f0f0f0'}),
            html.Th('Basket A', style={'textAlign': 'right', 'padding': '4px 8px', 'background': '#e8f0fe'}),
            html.Th('Basket B', style={'textAlign': 'right', 'padding': '4px 8px', 'background': '#fce8e6'}),
        ])
    ] + [
        # For each metric key k, look it up in each basket's dict.
        # .get(k, '—') returns the value for key k, or the em-dash '—' if
        # the basket dict is None or the key is absent (e.g. basket not run).
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


# ---------------------------------------------------------------------------
# Application layout
# ---------------------------------------------------------------------------
# app.layout defines the *entire* page structure as a tree of Dash/HTML
# components. Dash translates this into React components on the client side.
# The layout is set once at startup; callbacks then dynamically update
# individual component properties (like a figure, style, or options list)
# without re-rendering the whole page.
#
# Outer container style: maxWidth=100% fills the viewport. padding=0 8px
# adds small side margins. boxSizing=border-box includes padding in the
# declared width so content does not overflow horizontally.
app.layout = html.Div([

    # Page heading – rendered as an HTML <h1> tag.
    html.H1("mrktcmp _ markets compare"),

    # dcc.Tabs creates a tabbed interface.
    # id – unique ID so callbacks can read which tab is active.
    # value – the initially selected tab (matched against each Tab's value).
    # children – list of dcc.Tab components.
    dcc.Tabs(id='main-tabs', value='tab-chart', children=[

        # ---------------------------------------------------------------
        # Tab 1: Market Data – single-asset price + volume chart
        # ---------------------------------------------------------------
        dcc.Tab(label='Market Data', value='tab-chart', children=[

            # Asset-class selector (e.g. 'stocks', 'crypto', 'etfs').
            # RadioItems fires the update_asset_class callback which then
            # populates the asset search dropdown.
            html.Div([
                dcc.RadioItems(assetsClasses, id='assetclasses-type', inline=True),
                # Dropdown for the individual asset. Starts empty/disabled;
                # enabled once an asset class is chosen.
                dcc.Dropdown(id='asset-type', placeholder='Type to search…')
            ]),

            # Headline (asset name + exchange/country) injected by the
            # update_chart callback as an html.H2 + html.P pair.
            html.Div(id='asset-headline'),

            # The candlestick + volume chart.
            # dcc.Graph wraps a Plotly figure. The figure is set by the
            # update_chart callback.
            dcc.Graph(id='price-chart', style={'width': '100%'}),

            # Invisible data store: holds the serialised High/Low/Volume
            # columns so the y-axis zoom callback can recalculate ranges
            # without re-fetching the parquet file.
            dcc.Store(id='ohlcv-data'),
        ]),

        # ---------------------------------------------------------------
        # Tab 2: Backtesting – DCA simulation for two asset baskets
        # ---------------------------------------------------------------
        dcc.Tab(label='Backtesting', value='tab-backtest', children=[

            # Two basket panels side by side using a flex row.
            # gap: 8px adds horizontal space between the panels.
            # marginTop: 12px adds vertical breathing room below the tabs.
            html.Div([
                _basket_ui('a'),               # Basket A controls
                html.Div(style={'width': '24px'}),  # visual divider spacer
                _basket_ui('b'),               # Basket B controls
            ], style={'display': 'flex', 'gap': '8px', 'marginTop': '12px'}),

            # Year slider – lets the user choose how many years to simulate.
            html.Div([
                html.Label('Analysis period:', style={'fontWeight': 'bold', 'marginBottom': '4px'}),

                # dcc.RangeSlider has two draggable handles so the user can
                # select both a start and an end date.
                # min / max / value are integers (indices into the date list
                # stored in bt-date-store). They are set dynamically by the
                # update_date_range_slider callback when assets are added.
                # allowCross=False prevents the left handle from passing the
                # right handle and vice-versa (keeps the range always valid).
                # disabled=True at startup because no baskets are filled yet.
                dcc.RangeSlider(
                    id='bt-date-range',
                    min=0, max=1, step=1, value=[0, 1],
                    marks={},
                    allowCross=False,
                    disabled=True,
                ),

                # Human-readable date range display, e.g. "Jan 2020 – Dec 2024".
                # Updated by update_date_range_slider (on basket change) and
                # update_date_display (on every slider drag).
                html.Div(
                    id='bt-date-display',
                    children='Add assets to a basket to see the available date range.',
                    style={'color': '#666', 'fontSize': '13px', 'marginTop': '6px'},
                ),

                # Invisible store holding the ordered list of monthly date
                # strings (ISO format) that correspond to slider positions.
                # Position 0 → date_store[0], position N-1 → date_store[-1].
                dcc.Store(id='bt-date-store', data=[]),

            ], style={'marginTop': '20px', 'marginBottom': '8px'}),

            # Run button – clicking this fires the run_backtest_callback.
            html.Button(
                '▶ Start Backtest',
                id='bt-run',
                n_clicks=0,
                style={'padding': '8px 20px', 'fontSize': '14px', 'cursor': 'pointer',
                       'marginBottom': '16px'},
            ),

            # Status line – shows messages like "Backtest complete – 5.0 years".
            # Updated by run_backtest_callback; starts empty.
            html.Div(id='bt-status', style={'color': '#888', 'fontSize': '13px', 'marginBottom': '8px'}),

            # The portfolio value chart. display: none hides it until the
            # first backtest has been run; run_backtest_callback sets it to
            # display: block once results are available.
            dcc.Graph(id='bt-chart', style={'width': '100%', 'display': 'none'}),

            # Metrics comparison table rendered by _metrics_table().
            # Injected by run_backtest_callback as an html.Table.
            html.Div(id='bt-metrics', style={'marginTop': '16px'}),

            # Hidden result store (currently unused but reserved for future
            # drill-down interactions, e.g. clicking a month to inspect it).
            dcc.Store(id='bt-result-store', data={}),
        ]),
    ]),

], style={'maxWidth': '100%', 'padding': '0 8px', 'boxSizing': 'border-box'})


# ---------------------------------------------------------------------------
# Utility decorator: log the execution time of every callback
# ---------------------------------------------------------------------------

def log_time(func):
    """Decorator that logs how long a callback function takes to run.

    A decorator is a function that *wraps* another function to add behaviour
    before and/or after it. The @log_time syntax is shorthand for:
        func = log_time(func)

    functools.wraps(func) copies the original function's __name__, __doc__,
    etc. onto the wrapper so tools like debuggers and Dash still see the
    original function's name.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record the time just before calling the real function.
        t0 = time.time()
        # *args / **kwargs forwards all positional and keyword arguments
        # to the wrapped function unchanged.
        result = func(*args, **kwargs)
        # Log the elapsed time. The callback's name appears thanks to @wraps.
        log.debug(f'{func.__name__} callback time: {(time.time() - t0)*1000:,.2f}ms')
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Callback: populate the asset dropdown when an asset class is selected
# ---------------------------------------------------------------------------

@callback(
    # Output: set the 'options' and 'disabled' properties of asset-type dropdown.
    # Each Output(component_id, property) identifies exactly one target.
    Output('asset-type', 'options'),
    Output('asset-type', 'disabled'),
    # Input: fire this callback whenever the 'value' of assetclasses-type changes.
    Input('assetclasses-type', 'value'),
    # running: while this callback is executing, Dash automatically sets
    # asset-type.disabled = True; when done, it restores it to False.
    # This prevents the user from typing in the dropdown while results load.
    running=[(Output('asset-type', 'disabled'), True, False)]
)
@log_time
def update_asset_class(asset_class):
    """Populate the asset search dropdown for the selected asset class."""
    options, disabled = [], True
    if asset_class and df is not None:
        # Limit to 30 rows for an initial list; search refines further.
        filtered = df[df['asset_class'] == asset_class].head(30)
        # Build a list of option dicts: each has 'label' (displayed text) and
        # 'value' (the value stored when the user selects this option).
        options = [
            {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
            for _, row in filtered.iterrows()
        ]
        # Enable the dropdown now that options exist.
        disabled = False
        log.info("Asset class selected: %s", asset_class)
    # Returning two values corresponds to the two Output declarations above.
    return options, disabled


# ---------------------------------------------------------------------------
# Callback: live-search the asset dropdown as the user types
# ---------------------------------------------------------------------------

@callback(
    # allow_duplicate=True lets multiple callbacks share the same Output.
    # Without it, Dash raises an error if two callbacks write to the same target.
    # Here both update_asset_class and update_asset_search write to 'options'.
    Output('asset-type', 'options', allow_duplicate=True),
    # Input: fires every time the user types a character in the dropdown search box.
    Input('asset-type', 'search_value'),
    # State: read the current asset class and selected value WITHOUT re-firing.
    State('assetclasses-type', 'value'),
    State('asset-type', 'value'),
    # prevent_initial_call=True: do not fire this callback when the page first
    # loads (before the user has typed anything).
    prevent_initial_call=True
)
@log_time
def update_asset_search(search_value, asset_class, current_value):
    """Return ranked search results for the query typed into the dropdown."""
    if not asset_class or df is None:
        return []
    filtered = df[df['asset_class'] == asset_class]

    if search_value:
        sl = search_value.lower()
        sym = filtered['symbol'].str.lower()
        name = filtered['name'].str.lower()

        # Assign a numeric score to each row based on how closely it matches
        # the query. Lower score = better match. Rows with score == 99
        # (no match at all) are excluded.
        #
        # np.select(conditions, choices, default) evaluates each condition in
        # order (first match wins) and returns the corresponding choice value.
        # The result is a numpy array of scores, one per row in 'filtered'.
        #
        # Score meaning:
        #   0 – exact symbol match         (best; e.g. query "aapl" == "aapl")
        #   1 – symbol starts with query   (e.g. "aapl" prefix of "aaplx")
        #   2 – name starts with query     (e.g. "apple" prefix of "apple inc")
        #   3 – symbol contains query      (e.g. "aapl" inside "baapl")
        #   4 – name contains query        (e.g. "apple" inside "snapple")
        #  99 – no match at all            (excluded from results)
        score = np.select(
            [sym == sl,
             sym.str.startswith(sl, na=False),
             name.str.startswith(sl, na=False),
             sym.str.contains(sl, na=False),
             name.str.contains(sl, na=False)],
            [0, 1, 2, 3, 4],
            default=99
        )

        # mask is a boolean array: True for rows that matched at least one
        # condition (score < 99). This lets us filter to only matching rows.
        mask = score < 99

        # .assign(_score=...) adds a temporary '_score' column so we can
        # sort by it. .sort_values('_score') puts the best matches first.
        # .drop(columns='_score') removes the temp column before returning.
        # .head(30) limits results to the top 30 matches.
        filtered = (filtered[mask]
                    .assign(_score=score[mask])
                    .sort_values('_score')
                    .drop(columns='_score')
                    .head(30))
    else:
        # No query text: just show the first 30 assets for this class.
        filtered = filtered.head(30)

    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]

    # Preserve the currently selected asset even when it falls outside the
    # search results, so the dropdown does not silently lose its value on
    # each keystroke. Without this, selecting AAPL and then typing "micro"
    # would clear the selection because AAPL is no longer in the options list.
    if current_value and not any(o['value'] == current_value for o in options):
        sel = df[df['filename'] == current_value]
        if not sel.empty:
            row = sel.iloc[0]
            options.append({'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': current_value})

    return options


# ---------------------------------------------------------------------------
# Callback: render the price + volume chart for the selected asset
# ---------------------------------------------------------------------------

@callback(
    Output('price-chart', 'figure'),    # the main chart figure
    Output('asset-headline', 'children'),  # the asset name/exchange heading
    Output('ohlcv-data', 'data'),       # serialised OHLCV stored for zoom sync
    Input('asset-type', 'value')        # fires when the user picks an asset
)
@log_time
def update_chart(filename):
    """Load OHLCV data and build a candlestick + volume subplot chart."""
    # 'empty' is a convenience tuple to return when we cannot render a chart.
    empty = go.Figure(), "", None

    if not filename or not base_url or df is None:
        return empty

    try:
        # Look up the metadata row for the selected filename.
        # .iloc[0] picks the first (and only) matching row as a Series so
        # we can access columns by name: row['name'], row['exchange'], etc.
        row = df[df['filename'] == filename].iloc[0]

        # Build the asset heading: large name + smaller exchange info.
        headline = [
            html.H2(row['name'], style={'marginBottom': '2px'}),
            html.P(f"{row['exchange']} — {row['country']}", style={'marginTop': '0', 'color': 'gray'})
        ]

        # Load the OHLCV parquet file for this asset.
        ohlcv = pd.read_parquet(f"{base_url}/{filename}")

        # Limit to the last 10 years so the chart is not too dense. We
        # compare against the current timestamp in the same timezone as
        # the data index to avoid timezone mismatch errors.
        ohlcv = ohlcv[ohlcv.index >= pd.Timestamp.now(tz=ohlcv.index.tz) - pd.DateOffset(years=10)]

        # make_subplots divides the figure area into 2 rows.
        # rows=2, cols=1 → two vertically stacked panels.
        # shared_xaxes=True → panning/zooming the x-axis on one panel moves
        #   the other panel in sync.
        # row_heights=[0.75, 0.25] → price panel gets 75%, volume gets 25%.
        # vertical_spacing=0.02 → 2% gap between panels.
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.02)

        # Add a candlestick trace to the top panel (row=1).
        # Candlesticks show Open, High, Low, Close for each time period.
        # Green candles: Close > Open (price went up).
        # Red candles: Close < Open (price went down).
        fig.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['Open'],
            high=ohlcv['High'],
            low=ohlcv['Low'],
            close=ohlcv['Close'],
            name='Price'
        ), row=1, col=1)

        # Add a volume trace to the bottom panel (row=2).
        # go.Scattergl is the WebGL-accelerated version of go.Scatter – it
        # renders much faster for large datasets (thousands of bars).
        # fill='tozeroy' fills the area between the line and y=0, giving a
        # filled area chart (visually matches a bar chart for volume).
        fig.add_trace(go.Scattergl(
            x=ohlcv.index,
            y=ohlcv['Volume'],
            name='Volume',
            fill='tozeroy',
            line=dict(width=1)
        ), row=2, col=1)

        # Remove the range slider below the x-axis (redundant since we have
        # a volume panel below that already shows temporal context).
        fig.update_xaxes(rangeslider_visible=False)

        # Remove the legend and shrink margins for a cleaner look.
        # margin=dict(l=8, r=8, t=8, b=8) → 8px on all four sides.
        fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=8, b=8))

        # Prepare a copy of the DataFrame for potential grid display.
        # reset_index() moves the date index into a regular column.
        grid_df = ohlcv.reset_index()
        # Rename the index column (whatever it was called) to 'Date'.
        grid_df.rename(columns={grid_df.columns[0]: 'Date'}, inplace=True)
        # Format dates as "01-Jan-2023" strings for readability.
        grid_df['Date'] = pd.to_datetime(grid_df['Date']).dt.strftime('%d-%b-%Y')
        # Format price columns to 2 decimal places with thousands separator.
        for col in ('Open', 'High', 'Low', 'Close'):
            if col in grid_df.columns:
                grid_df[col] = grid_df[col].map('{:,.2f}'.format)
        if 'Volume' in grid_df.columns:
            # :, adds a thousands separator (e.g. 1000000 → "1,000,000").
            grid_df['Volume'] = grid_df['Volume'].map('{:,}'.format)

        log.info("Data loaded from %s", filename)

        # Store only High/Low/Volume in the dcc.Store component.
        # Close is already encoded in the Candlestick trace so we do not
        # duplicate it. .to_json(orient='split') serialises as:
        #   {"columns": [...], "index": [...], "data": [[...], ...]}
        # This compact format is faster to deserialise than the default.
        store = ohlcv[['High', 'Low', 'Volume']].to_json(date_format='iso', orient='split')
        return fig, headline, store

    except Exception:
        log.exception("Failed to load chart data for %s", filename)
        return empty


# ---------------------------------------------------------------------------
# Callback: adjust price y-axis when the user zooms or pans the x-axis
# ---------------------------------------------------------------------------

@callback(
    Output('price-chart', 'figure', allow_duplicate=True),
    # relayoutData is fired by Plotly every time the user interacts with the
    # chart (zoom, pan, double-click to reset). It is a dict describing what
    # changed; e.g. {'xaxis.range[0]': '2023-01-01', 'xaxis.range[1]': ...}.
    Input('price-chart', 'relayoutData'),
    State('ohlcv-data', 'data'),
    prevent_initial_call=True
)
@log_time
def sync_yaxis_on_xzoom(relayout_data, ohlcv_json):
    """Rescale the price y-axis to fit the visible candles after an x-axis zoom.

    Plotly does not automatically tighten the y-axis when you zoom in on the
    x-axis of a candlestick chart (it keeps the original full range). This
    callback detects x-range changes and recalculates the y-axis bounds from
    the data that is actually visible so candles fill the panel nicely.
    """
    if not relayout_data or ohlcv_json is None:
        # Nothing useful happened; leave the chart unchanged.
        return no_update

    # If the user double-clicked to reset zoom, re-enable autorange on both axes.
    # Patch() creates a partial update object: we only send the changed keys.
    if relayout_data.get('xaxis.autorange') or relayout_data.get('autosize'):
        patch = Patch()
        patch['layout']['yaxis']['autorange'] = True
        patch['layout']['yaxis2']['autorange'] = True
        return patch

    # Extract the new x-axis boundaries set by the zoom/pan gesture.
    x0 = relayout_data.get('xaxis.range[0]')
    x1 = relayout_data.get('xaxis.range[1]')

    # If neither key is present, this was some other layout event (e.g. a
    # legend click) that does not affect the x range. Ignore it.
    if x0 is None or x1 is None:
        return no_update

    # Deserialise the stored OHLCV JSON back into a DataFrame.
    # io.StringIO wraps the string as a file-like object so pd.read_json
    # can consume it. convert_axes=True restores the datetime index.
    ohlcv = pd.read_json(io.StringIO(ohlcv_json), orient='split', convert_axes=True)

    # Convert the boundary strings to pandas Timestamps for comparison.
    x0_ts, x1_ts = pd.Timestamp(x0), pd.Timestamp(x1)

    # If the stored data has timezone info but the Timestamps from Plotly do
    # not, localise then convert so the comparison works correctly.
    if ohlcv.index.tz is not None and x0_ts.tz is None:
        x0_ts = x0_ts.tz_localize('UTC').tz_convert(ohlcv.index.tz)
        x1_ts = x1_ts.tz_localize('UTC').tz_convert(ohlcv.index.tz)

    # Select only the rows that fall within the visible x window.
    visible = ohlcv[(ohlcv.index >= x0_ts) & (ohlcv.index <= x1_ts)]

    # If no rows are visible (e.g. the user zoomed to a gap in the data),
    # leave the axes as they are.
    if visible.empty:
        return no_update

    # Calculate the price range for visible bars.
    low, high = visible['Low'].min(), visible['High'].max()

    # Add a 2% margin so candles do not touch the top/bottom edges.
    margin = (high - low) * 0.02

    # Build a Patch to update only the layout axes, not the traces.
    # This is much cheaper than returning the entire figure.
    patch = Patch()
    patch['layout']['yaxis']['autorange'] = False
    patch['layout']['yaxis']['range'] = [low - margin, high + margin]
    patch['layout']['yaxis2']['autorange'] = False
    # Volume axis: 0 at the bottom, 10% above the max volume at the top.
    patch['layout']['yaxis2']['range'] = [0, visible['Volume'].max() * 1.1]
    return patch


# ---------------------------------------------------------------------------
# Backtesting callbacks – asset class selectors (one per basket)
# ---------------------------------------------------------------------------

@callback(
    Output('bt-asset-a', 'options'),
    Output('bt-asset-a', 'disabled'),
    Input('bt-assetclass-a', 'value'),
)
@log_time
def bt_assetclass_a(asset_class):
    """Populate Basket A's asset dropdown when the user picks an asset class."""
    return _bt_assetclass_options(asset_class)


@callback(
    Output('bt-asset-b', 'options'),
    Output('bt-asset-b', 'disabled'),
    Input('bt-assetclass-b', 'value'),
)
@log_time
def bt_assetclass_b(asset_class):
    """Populate Basket B's asset dropdown when the user picks an asset class."""
    return _bt_assetclass_options(asset_class)


def _bt_assetclass_options(asset_class):
    """Shared logic: return (options, disabled) for a basket's asset dropdown.

    Called by both bt_assetclass_a and bt_assetclass_b to avoid duplicating
    the same filtering / formatting code.

    Returns
    -------
    ([], True)  – empty options + dropdown disabled (no asset class chosen).
    (options, False) – populated options + dropdown enabled.
    """
    if not asset_class or df is None:
        # No class selected yet: return empty options and keep the dropdown
        # greyed out (disabled=True).
        return [], True

    # Filter to the chosen class, limit to 30 rows for the initial list.
    filtered = df[df['asset_class'] == asset_class].head(30)

    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]
    # False = not disabled (i.e. the dropdown is enabled).
    return options, False


# ---------------------------------------------------------------------------
# Backtesting callbacks – live search within each basket's dropdown
# ---------------------------------------------------------------------------

@callback(
    Output('bt-asset-a', 'options', allow_duplicate=True),
    Input('bt-asset-a', 'search_value'),
    State('bt-assetclass-a', 'value'),
    State('bt-asset-a', 'value'),
    prevent_initial_call=True
)
@log_time
def bt_search_a(search_value, asset_class, current_value):
    """Refine Basket A's dropdown options as the user types a search query."""
    return _bt_asset_search(search_value, asset_class, current_value)


@callback(
    Output('bt-asset-b', 'options', allow_duplicate=True),
    Input('bt-asset-b', 'search_value'),
    State('bt-assetclass-b', 'value'),
    State('bt-asset-b', 'value'),
    prevent_initial_call=True
)
@log_time
def bt_search_b(search_value, asset_class, current_value):
    """Refine Basket B's dropdown options as the user types a search query."""
    return _bt_asset_search(search_value, asset_class, current_value)


def _bt_asset_search(search_value, asset_class, current_value):
    """Shared search logic for both basket dropdowns.

    Identical scoring algorithm to update_asset_search (see its comments for
    a detailed explanation). Extracted to avoid code duplication.
    """
    if not asset_class or df is None:
        return []

    filtered = df[df['asset_class'] == asset_class]

    if search_value:
        sl = search_value.lower()
        sym = filtered['symbol'].str.lower()
        name = filtered['name'].str.lower()

        # Score each row: 0 = exact match (best), 99 = no match (excluded).
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

    # Keep the currently selected asset in the options list even if it does
    # not match the current query string (prevents silent deselection).
    if current_value and not any(o['value'] == current_value for o in options):
        sel = df[df['filename'] == current_value]
        if not sel.empty:
            row = sel.iloc[0]
            options.append({'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': current_value})

    return options


# ---------------------------------------------------------------------------
# Backtesting callbacks – basket management (add / remove assets)
# ---------------------------------------------------------------------------

@callback(
    Output('bt-basket-store-a', 'data'),     # updated JSON list for basket A
    Output('bt-basket-list-a', 'children'),  # updated visible list for basket A
    Input('bt-add-a', 'n_clicks'),           # add button clicked
    # ALL pattern: fires when *any* remove button for basket A is clicked.
    # The Input value is a list of n_clicks, one per matching button.
    Input({'type': 'bt-remove-a', 'index': ALL}, 'n_clicks'),
    State('bt-asset-a', 'value'),            # currently selected asset filename
    State('bt-basket-store-a', 'data'),      # current basket contents (list of dicts)
    prevent_initial_call=True
)
@log_time
def manage_basket_a(add_clicks, remove_clicks, selected_asset, basket_data):
    """Handle add/remove actions for Basket A."""
    # Both add and remove share the same implementation; basket_id='a'
    # distinguishes which basket's IDs to look for.
    return _manage_basket('a', remove_clicks, selected_asset, basket_data)


@callback(
    Output('bt-basket-store-b', 'data'),
    Output('bt-basket-list-b', 'children'),
    Input('bt-add-b', 'n_clicks'),
    Input({'type': 'bt-remove-b', 'index': ALL}, 'n_clicks'),
    State('bt-asset-b', 'value'),
    State('bt-basket-store-b', 'data'),
    prevent_initial_call=True
)
@log_time
def manage_basket_b(add_clicks, remove_clicks, selected_asset, basket_data):
    """Handle add/remove actions for Basket B."""
    return _manage_basket('b', remove_clicks, selected_asset, basket_data)


def _manage_basket(basket_id, remove_clicks, selected_asset, basket_data):
    """Core logic for adding/removing an asset from a basket.

    Parameters
    ----------
    basket_id      : str  – 'a' or 'b'.
    remove_clicks  : list – n_clicks for each remove button (may be empty).
    selected_asset : str  – filename of the asset currently selected in the
                            dropdown (or None if nothing is selected).
    basket_data    : list – current list of asset dicts in the dcc.Store.

    Returns
    -------
    (updated_basket, updated_list_component)
    or (no_update, no_update) when nothing should change.
    """
    # dash.callback_context provides runtime information about the callback
    # that just fired, available only inside a callback function.
    ctx = dash.callback_context

    # ctx.triggered is a list of dicts describing every Input that changed.
    # If it is empty the callback was fired spuriously; do nothing.
    if not ctx.triggered:
        return no_update, no_update

    # ctx.triggered_id is the id of the single Input that actually caused the
    # callback to fire. For a plain button it is a string like 'bt-add-a'.
    # For a pattern-matching button it is a dict like
    #   {'type': 'bt-remove-a', 'index': 'aapl.parquet'}.
    triggered_id = ctx.triggered_id

    # ctx.triggered[0]['value'] is the new property value that changed. For
    # n_clicks this is the new click count. .get('value', 0) or 0 handles
    # the case where Dash emits None for newly rendered components whose
    # n_clicks initialises to 0 – those should NOT trigger a removal.
    triggered_value = ctx.triggered[0].get('value', 0) or 0

    # Copy the basket list so we can mutate it safely. 'or []' handles the
    # case where basket_data is None (dcc.Store initialised but never written).
    basket = list(basket_data or [])

    if isinstance(triggered_id, dict) and triggered_id.get('type') == f'bt-remove-{basket_id}':
        # A remove button was clicked. The dict-id carries 'index' = filename.

        # Guard: newly rendered remove buttons fire the ALL-pattern callback
        # with n_clicks=0 when added to the layout. Skipping those prevents
        # a phantom removal every time a new asset is added to the basket.
        if triggered_value > 0:
            filename = triggered_id['index']
            # Keep all items except the one whose filename matches.
            basket = [item for item in basket if item['filename'] != filename]

    elif triggered_id == f'bt-add-{basket_id}' and selected_asset and df is not None:
        # The add button was clicked and the dropdown has a selection.

        # Prevent duplicate entries: only add if the asset is not already in
        # the basket. any() returns True as soon as one match is found.
        if not any(item['filename'] == selected_asset for item in basket):
            meta = df[df['filename'] == selected_asset]
            if not meta.empty:
                row = meta.iloc[0]
                # Append a minimal dict: filename, symbol, and display name.
                basket.append({'filename': selected_asset, 'symbol': row['symbol'], 'name': row['name']})

    else:
        # The callback fired for some other reason (e.g. the remove button
        # list was updated with new buttons that all report n_clicks=0, or
        # the add button was clicked without an asset selected). Do nothing.
        return no_update, no_update

    # Return the new basket data (written to dcc.Store) and the refreshed
    # visible list component (written to the basket-list Div's children).
    return basket, _render_basket_list(basket, basket_id)


# ---------------------------------------------------------------------------
# Helper: build slider marks from an ordered list of monthly dates
# ---------------------------------------------------------------------------

def _build_slider_marks(date_range):
    """Return a {position: label} dict for the RangeSlider.

    The density of labels adapts to the length of the date range so the
    slider is never over-crowded:
      ≤ 12 months  → every month labelled.
      13–36 months → every 3 months (quarterly).
      > 36 months  → every 12 months (yearly).

    The first and last positions are always labelled so the user can always
    read the absolute boundaries of the available data.

    Parameters
    ----------
    date_range : DatetimeIndex – ordered monthly dates (from pd.date_range).

    Returns
    -------
    dict mapping integer slider position → short date string ('%b %y').
    """
    n = len(date_range)
    # Choose how many months to skip between each visible mark.
    step = 1 if n <= 12 else 3 if n <= 36 else 12
    marks = {}
    for i, d in enumerate(date_range):
        # Label this position if it falls on a step boundary.
        if i % step == 0:
            marks[i] = d.strftime('%b %y')  # e.g. 'Jan 20'
    # Always include the final position so the right boundary is readable.
    marks[n - 1] = date_range[-1].strftime('%b %y')
    return marks


# ---------------------------------------------------------------------------
# Callback: update the date-range slider when baskets change
# ---------------------------------------------------------------------------

@callback(
    # Seven outputs: slider configuration (5) + date store + display text.
    Output('bt-date-range', 'min'),
    Output('bt-date-range', 'max'),
    Output('bt-date-range', 'value'),
    Output('bt-date-range', 'marks'),
    Output('bt-date-range', 'disabled'),
    Output('bt-date-store', 'data'),
    Output('bt-date-display', 'children'),
    # Fires whenever either basket's contents change.
    Input('bt-basket-store-a', 'data'),
    Input('bt-basket-store-b', 'data'),
)
@log_time
def update_date_range_slider(basket_a, basket_b):
    """Recompute the available date range whenever the basket contents change.

    Loads just the 'Close' column from each asset's parquet file to find the
    monthly date bounds, then intersects all ranges to find the overlap that
    is common to every selected asset.  Updates the RangeSlider bounds and
    marks to reflect this common window, and stores the ordered list of dates
    so the run callback can look up exact Timestamps from slider positions.
    """
    # Convenience tuple to return when the slider should be disabled.
    # min=0, max=1, value=[0,1] gives the slider a valid (non-zero-width)
    # range even when disabled; an empty range would cause a Dash warning.
    _disabled = (0, 1, [0, 1], {}, True, [], '')

    # Extract filename lists from each basket (or empty lists if unset).
    filenames_a = [item['filename'] for item in (basket_a or [])]
    filenames_b = [item['filename'] for item in (basket_b or [])]

    # Nothing to compute if both baskets are empty.
    if not filenames_a and not filenames_b:
        return (*_disabled[:6],
                'Add assets to a basket to see the available date range.')

    if not base_url or df is None:
        return (*_disabled[:6], 'No data source configured.')

    # Compute the intersection of all asset date ranges across both baskets.
    # get_common_date_range returns (None, None) when there is no overlap.
    common_start, common_end = get_common_date_range(
        base_url, filenames_a, filenames_b, df
    )

    if common_start is None:
        return (*_disabled[:6],
                'No overlapping date range found across the selected assets.')

    # Build the ordered list of month-end dates within the common window.
    # pd.date_range with freq='ME' generates one date per calendar month-end.
    date_range = pd.date_range(common_start, common_end, freq='ME')
    n = len(date_range)

    # Serialise dates as ISO strings so they can be stored in dcc.Store (which
    # holds JSON). pd.Timestamp.isoformat() produces e.g. '2020-01-31T00:00:00+00:00'.
    date_store = [d.isoformat() for d in date_range]

    marks = _build_slider_marks(date_range)

    # Show the full common range as the initial selection.
    d0 = date_range[0].strftime('%b %Y')
    d1 = date_range[-1].strftime('%b %Y')
    display = f'Available: {d0} – {d1}  ({n} months)'

    # Return: min, max, value (full range), marks, not-disabled, date store, display.
    return 0, n - 1, [0, n - 1], marks, False, date_store, display


# ---------------------------------------------------------------------------
# Callback: update the date display text as the user drags the slider
# ---------------------------------------------------------------------------

@callback(
    # allow_duplicate=True: both this callback and update_date_range_slider
    # write to 'bt-date-display'. Dash requires explicit permission for that.
    Output('bt-date-display', 'children', allow_duplicate=True),
    Input('bt-date-range', 'value'),
    State('bt-date-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def update_date_display(slider_value, date_store):
    """Refresh the human-readable date label whenever the slider moves.

    The slider reports integer indices; we look them up in the date_store list
    to get the actual Timestamps and format them for display.
    """
    # Guard against the slider firing before the date store has been populated.
    if not slider_value or not date_store:
        return no_update
    i0, i1 = slider_value[0], slider_value[1]
    # Parse ISO strings back to Timestamps for formatting.
    d0 = pd.Timestamp(date_store[i0]).strftime('%b %Y')
    d1 = pd.Timestamp(date_store[i1]).strftime('%b %Y')
    n_months = i1 - i0 + 1
    return f'Selected: {d0} – {d1}  ({n_months} months)'


# ---------------------------------------------------------------------------
# Callback: run the DCA backtest and render results
# ---------------------------------------------------------------------------

@callback(
    Output('bt-chart', 'figure'),      # the portfolio value line chart
    Output('bt-chart', 'style'),       # show/hide the chart container
    Output('bt-metrics', 'children'),  # the metrics comparison table
    Output('bt-status', 'children'),   # status / error message text
    Input('bt-run', 'n_clicks'),       # fires when the Run button is clicked
    State('bt-basket-store-a', 'data'),  # basket A contents (read, not trigger)
    State('bt-basket-store-b', 'data'),  # basket B contents
    # slider_value is [start_index, end_index] into the date_store list.
    State('bt-date-range', 'value'),
    # date_store is a list of ISO-format date strings, one per slider step.
    State('bt-date-store', 'data'),
    prevent_initial_call=True  # do not run at page load (no data yet)
)
@log_time
def run_backtest_callback(n_clicks, basket_a, basket_b, slider_value, date_store):
    """Execute the DCA simulation for both baskets and update the UI.

    The backtest engine (backtest.run_backtest) simulates investing a fixed
    amount every month, then computes performance metrics. Here we:
    1. Validate that at least one basket has assets and dates are available.
    2. Convert slider indices to actual Timestamps.
    3. Run the backtest for each non-empty basket.
    4. Plot both portfolios on a single chart.
    5. Build the metrics comparison table.
    """
    # Prepare reusable figure / style constants so returns are concise.
    empty_chart = go.Figure()
    hidden = {'width': '100%', 'display': 'none'}    # hide the chart div
    visible = {'width': '100%', 'display': 'block'}  # show the chart div

    # Require at least one basket to have assets before running.
    if not basket_a and not basket_b:
        return empty_chart, hidden, '', 'Please fill at least one basket.'

    if not base_url or df is None:
        return empty_chart, hidden, '', 'No data source available.'

    # Guard: the date slider must have been populated by update_date_range_slider
    # before the user can run. If it has not (e.g. all files failed to load),
    # we cannot resolve the slider positions to actual Timestamps.
    if not date_store or not slider_value or len(date_store) < 2:
        return empty_chart, hidden, '', 'No date range available. Add assets first.'

    # Convert the slider's integer positions back to pandas Timestamps.
    # slider_value is [i0, i1]; date_store[i0] is an ISO string like
    # '2020-01-31T00:00:00+00:00'. pd.Timestamp() parses it correctly.
    start_date = pd.Timestamp(date_store[slider_value[0]])
    end_date = pd.Timestamp(date_store[slider_value[1]])

    # Extract filenames from each basket's list of asset dicts.
    # A list comprehension iterates through basket_a (or []) if it is None.
    filenames_a = [item['filename'] for item in (basket_a or [])]
    filenames_b = [item['filename'] for item in (basket_b or [])]

    # Run the simulation for each basket, but skip the call entirely if the
    # basket has no assets (saves an unnecessary function call).
    # run_backtest returns (portfolio_series, metrics_dict) on success, or
    # (None, None) if no data was available.
    portfolio_a, metrics_a = (
        run_backtest(base_url, filenames_a, start_date, end_date, df)
        if filenames_a else (None, None)
    )
    portfolio_b, metrics_b = (
        run_backtest(base_url, filenames_b, start_date, end_date, df)
        if filenames_b else (None, None)
    )

    # If both backtests failed (e.g. all parquet files missing), abort.
    if portfolio_a is None and portfolio_b is None:
        return empty_chart, hidden, '', 'No data available for the selected period.'

    # Build the portfolio value chart.
    fig = go.Figure()

    if portfolio_a is not None:
        # go.Scatter draws a line chart. round(2) avoids floating-point noise
        # in hover tooltips (e.g. 1000.0000000002 → 1000.0).
        fig.add_trace(go.Scatter(
            x=portfolio_a.index,          # x-axis: monthly dates
            y=portfolio_a.round(2),        # y-axis: portfolio value in EUR
            name='Basket A',
            line=dict(color='#1a56db', width=2),  # blue line, 2px thick
        ))

    if portfolio_b is not None:
        fig.add_trace(go.Scatter(
            x=portfolio_b.index,
            y=portfolio_b.round(2),
            name='Basket B',
            line=dict(color='#c0392b', width=2),  # red line
        ))

    # The two baskets may cover different date ranges (e.g. if one basket
    # contains an asset that only recently started trading). Title the chart
    # using the longer simulation so users understand what they are looking at.
    months_shown = max(
        len(portfolio_a) if portfolio_a is not None else 0,
        len(portfolio_b) if portfolio_b is not None else 0,
    )

    d0_label = start_date.strftime('%b %Y')
    d1_label = end_date.strftime('%b %Y')
    fig.update_layout(
        title=f'Portfolio Value  {d0_label} – {d1_label}  ({months_shown} months, 1,000 €/month)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (€)',
        # hovermode='x unified': when you hover anywhere on the chart, a
        # single tooltip appears showing the values for ALL traces at that
        # x position, instead of separate tooltips per trace.
        hovermode='x unified',
        # Place the legend in a horizontal bar above the chart to save space.
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        # Tight margins to maximise plot area; t=48 leaves room for the title.
        margin=dict(l=8, r=8, t=48, b=8),
    )

    # Build the side-by-side metrics table and assemble the status message.
    metrics_div = _metrics_table(metrics_a, metrics_b)
    status = f'Backtest complete – {d0_label} to {d1_label} ({months_shown} months).'

    log.info("Backtest completed: %d months, A=%s, B=%s",
             months_shown, len(filenames_a), len(filenames_b))

    # Return four values matching the four Output declarations above.
    return fig, visible, metrics_div, status


# ---------------------------------------------------------------------------
# Entry point – run the development server when executed directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # app.run starts Dash's built-in Flask development server.
    # debug=True enables hot-reload (auto-restarts on file changes) and shows
    # an error overlay in the browser for Python exceptions.
    # In production, use gunicorn with the 'server' variable instead.
    app.run(debug=True)
