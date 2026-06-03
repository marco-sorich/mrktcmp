# ---------------------------------------------------------------------------
# callbacks/chart.py – Market Data tab callbacks
#
# This module contains the four Dash callbacks that power the Market Data tab:
#
#   update_asset_class   – Populates the asset dropdown when the user picks
#                          an asset class (stocks, crypto, …).
#   update_asset_search  – Refines the dropdown options as the user types,
#                          using a ranked scoring algorithm.
#   update_chart         – Loads OHLCV data from a parquet file and renders
#                          the candlestick + volume subplot chart.
#   sync_yaxis_on_xzoom  – Recalculates the price y-axis whenever the user
#                          zooms or pans the x-axis so candles always fill
#                          the visible panel height.
#
# All callbacks use the standalone @callback decorator (from dash import
# callback) rather than @app.callback. This means they can live in a
# separate module without needing a reference to the Dash app object.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

# io.StringIO: turns a plain string into a file-like object so libraries
# that expect to read from a file can work with in-memory strings instead.
# Used when deserialising OHLCV data that was JSON-encoded into a dcc.Store.
import io

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# numpy: fast numerical array library. Used for np.select (vectorised
# conditional scoring) to rank search results efficiently.
import numpy as np

# pandas: tabular data library. Used to load parquet files, filter rows
# by date, and serialise/deserialise OHLCV data for the zoom callback.
import pandas as pd

# Plotly graph objects: the low-level chart primitives.
#   go.Figure        – the container for all chart elements.
#   go.Candlestick   – renders OHLCV data as candlestick bars.
#   go.Scattergl     – a WebGL-accelerated line/area trace for volume.
import plotly.graph_objects as go

# make_subplots: creates a Figure pre-configured with multiple subplot
# areas. Used to stack the price panel (top, 75%) above the volume panel
# (bottom, 25%) with a shared x-axis so zooming one panel zooms the other.
from plotly.subplots import make_subplots

# Dash component and callback utilities:
#   html     – HTML tag wrappers (html.H2, html.P, …).
#   Input    – marks a callback argument as a trigger (fires the callback).
#   Output   – marks a return value as a target to write to.
#   State    – read at fire-time but does NOT trigger the callback on its own.
#   callback – the standalone decorator that registers a function as a
#              Dash callback without requiring the app object.
#   Patch    – a lightweight "partial update" object: instead of returning
#              the whole figure, send only the changed keys. Saves bandwidth.
#   no_update – a sentinel value that tells Dash "leave this Output exactly
#               as it is; do not overwrite it".
from dash import html, Input, Output, State, callback, Patch, no_update

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

# config is imported as a module object so that df, base_url, and log are
# resolved at call time. This allows test patches applied to config_module to
# be seen by the callback functions at the moment they execute.
import src.config as _config

# log_time is a decorator that measures and logs each callback's wall-clock
# execution time at DEBUG level.
from src.utils import log_time


# ---------------------------------------------------------------------------
# Callback: populate the asset dropdown when an asset class is selected
# ---------------------------------------------------------------------------

@callback(
    # Output: set the 'options' and 'disabled' properties of asset-type.
    # Each Output(component_id, property) identifies exactly one target.
    Output('asset-type', 'options'),
    Output('asset-type', 'disabled'),
    # Input: fire this callback whenever the 'value' of assetclasses-type
    # changes (i.e. whenever the user clicks a different radio button).
    Input('assetclasses-type', 'value'),
    # NOTE: do NOT add a `running=[...]` argument here. `running` is a
    # background-callback feature (it only applies when background=True). On a
    # regular synchronous callback, dash-renderer sets the "running" value
    # (disabled=True) but never applies the "runningOff" reset (disabled=False),
    # so asset-type.disabled would stay stuck at True and the dropdown would
    # never re-enable. This callback is a fast in-memory filter and already
    # returns the correct `disabled` value below, so no `running` is needed.
)
@log_time
def update_asset_class(asset_class):
    """Populate the asset search dropdown for the selected asset class.

    Filters the master catalogue to the chosen class and returns up to 200
    options. The dropdown is disabled if no class is chosen or the catalogue
    has not been loaded.

    Parameters
    ----------
    asset_class : str or None – the asset class selected by the radio button
                                (e.g. 'stocks', 'crypto'). None on first load.

    Returns
    -------
    (options, disabled)
      options  : list of {'label': str, 'value': str} dicts for the dropdown.
      disabled : bool – True when the dropdown should be greyed out.
    """
    options, disabled = [], True
    if asset_class and _config.df is not None:
        # Filter to the chosen class and limit to 200 rows for the initial
        # list; typing in the dropdown triggers update_asset_search which
        # further refines and ranks results.
        filtered = _config.df[_config.df['asset_class'] == asset_class].head(200)
        # Build a list of option dicts: each has 'label' (displayed text) and
        # 'value' (the value stored when the user selects this option).
        options = [
            {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
            for _, row in filtered.iterrows()
        ]
        # Enable the dropdown now that options exist.
        disabled = False
        _config.log.info("Asset class selected: %s", asset_class)
    # Returning two values corresponds to the two Output declarations above.
    return options, disabled


# ---------------------------------------------------------------------------
# Callback: live-search the asset dropdown as the user types
# ---------------------------------------------------------------------------

@callback(
    # allow_duplicate=True lets multiple callbacks share the same Output.
    # Without it, Dash raises an error if two callbacks write to the same
    # target. Both update_asset_class and update_asset_search write 'options'.
    Output('asset-type', 'options', allow_duplicate=True),
    # Input: fires every time the user types a character in the search box.
    Input('asset-type', 'search_value'),
    # State: read the current class and selected value WITHOUT re-firing.
    State('assetclasses-type', 'value'),
    State('asset-type', 'value'),
    # prevent_initial_call=True: do not fire on page load (nothing typed yet).
    prevent_initial_call=True,
)
@log_time
def update_asset_search(search_value, asset_class, current_value):
    """Return ranked search results for the query typed into the dropdown.

    Uses a numeric scoring system so the best matches appear first:
      0 – exact symbol match         (best)
      1 – symbol starts with query
      2 – name starts with query
      3 – symbol contains query
      4 – name contains query
     99 – no match at all            (excluded)

    Rows with score 99 are removed; the rest are sorted ascending so
    better matches appear at the top of the dropdown list.

    Parameters
    ----------
    search_value  : str or None – the text the user has typed.
    asset_class   : str or None – the currently selected asset class.
    current_value : str or None – the filename of the currently selected
                                  asset (if any), used to keep it visible
                                  in the options list even when it does not
                                  match the query.

    Returns
    -------
    list of {'label': str, 'value': str} dicts, at most 30 items.
    """
    if not asset_class or _config.df is None:
        return []
    filtered = _config.df[_config.df['asset_class'] == asset_class]

    if search_value:
        sl = search_value.lower()
        sym = filtered['symbol'].str.lower()
        name = filtered['name'].str.lower()

        # np.select(conditions, choices, default) evaluates each condition in
        # order (first match wins) and returns the corresponding choice value.
        # The result is a numpy array of scores, one per row in 'filtered'.
        score = np.select(
            [sym == sl,
             sym.str.startswith(sl, na=False),
             name.str.startswith(sl, na=False),
             sym.str.contains(sl, na=False),
             name.str.contains(sl, na=False)],
            [0, 1, 2, 3, 4],
            default=99,
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
        sel = _config.df[_config.df['filename'] == current_value]
        if not sel.empty:
            row = sel.iloc[0]
            options.append({
                'label': f"{row['symbol']} — {row['name']} ({row['interval']})",
                'value': current_value,
            })
    return options


# ---------------------------------------------------------------------------
# Callback: render the price + volume chart for the selected asset
# ---------------------------------------------------------------------------

@callback(
    Output('price-chart', 'figure'),        # the main chart figure
    Output('asset-headline', 'children'),   # the asset name/exchange heading
    Output('ohlcv-data', 'data'),           # serialised OHLCV stored for zoom sync
    Input('asset-type', 'value'),           # fires when the user picks an asset
)
@log_time
def update_chart(filename):
    """Load OHLCV data and build a candlestick + volume subplot chart.

    Reads the asset's parquet file from BASE_URL, clips the data to the last
    10 years, and constructs a two-panel Plotly figure (candlesticks on top,
    volume area chart on the bottom with a shared x-axis).

    The serialised High/Low/Volume columns are written to a dcc.Store so the
    sync_yaxis_on_xzoom callback can recalculate y-axis bounds on zoom without
    re-fetching the file.

    Parameters
    ----------
    filename : str or None – parquet filename of the selected asset
                             (e.g. 'aapl.parquet'). None when no asset is
                             selected yet.

    Returns
    -------
    (figure, headline_children, ohlcv_json)
      figure           : go.Figure with candlestick + volume traces.
      headline_children: list of [html.H2, html.P] with name and exchange.
      ohlcv_json       : JSON string (orient='split') of High/Low/Volume data.
    """
    # 'empty' is a convenience tuple to return when we cannot render a chart.
    empty = go.Figure(), "", None

    if not filename or not _config.base_url or _config.df is None:
        return empty

    try:
        # Look up the metadata row for the selected filename.
        # .iloc[0] picks the first (and only) matching row as a Series so
        # we can access columns by name: row['name'], row['exchange'], etc.
        row = _config.df[_config.df['filename'] == filename].iloc[0]

        # Build the asset heading: large name + smaller exchange info.
        headline = [
            html.H2(row['name'], style={'marginBottom': '2px'}),
            html.P(f"{row['exchange']} — {row['country']}", style={'marginTop': '0', 'color': 'gray'}),
        ]

        # Load the OHLCV parquet file for this asset.
        ohlcv = pd.read_parquet(f"{_config.base_url}/{filename}")

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
        # Green candles: Close > Open (price went up).
        # Red candles:   Close < Open (price went down).
        fig.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['Open'], high=ohlcv['High'],
            low=ohlcv['Low'], close=ohlcv['Close'],
            name='Price',
        ), row=1, col=1)

        # Add a volume trace to the bottom panel (row=2).
        # go.Scattergl is the WebGL-accelerated version of go.Scatter – it
        # renders much faster for large datasets (thousands of bars).
        # fill='tozeroy' fills the area between the line and y=0, giving a
        # filled area chart (visually similar to a bar chart for volume).
        fig.add_trace(go.Scattergl(
            x=ohlcv.index, y=ohlcv['Volume'],
            name='Volume', fill='tozeroy', line=dict(width=1),
        ), row=2, col=1)

        # Remove the range slider below the x-axis (redundant since we
        # already have a volume panel that shows temporal context).
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

        _config.log.info("Data loaded from %s", filename)

        # Store only High/Low/Volume in the dcc.Store component.
        # .to_json(orient='split') serialises as:
        #   {"columns": [...], "index": [...], "data": [[...], ...]}
        # This compact format is faster to deserialise than the default.
        store = ohlcv[['High', 'Low', 'Volume']].to_json(date_format='iso', orient='split')
        return fig, headline, store

    except Exception:
        _config.log.exception("Failed to load chart data for %s", filename)
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
    prevent_initial_call=True,
)
@log_time
def sync_yaxis_on_xzoom(relayout_data, ohlcv_json):
    """Rescale the price y-axis to fit the visible candles after an x-axis zoom.

    Plotly does not automatically tighten the y-axis when you zoom in on the
    x-axis of a candlestick chart (it keeps the original full range). This
    callback detects x-range changes and recalculates the y-axis bounds from
    the data that is actually visible so candles fill the panel nicely.

    Parameters
    ----------
    relayout_data : dict or None – Plotly's description of what changed in the
                                   chart layout (zoom range, autorange, etc.).
    ohlcv_json    : str or None  – JSON string of High/Low/Volume data
                                   previously written to dcc.Store by
                                   update_chart.

    Returns
    -------
    Patch or no_update
      Patch    – a partial figure update containing only the new y-axis ranges.
      no_update – signals Dash to leave the figure unchanged.
    """
    if not relayout_data or ohlcv_json is None:
        # Nothing useful happened; leave the chart unchanged.
        return no_update

    # If the user double-clicked to reset zoom, re-enable autorange on both
    # axes. Patch() creates a partial update object: only the changed keys
    # are sent to the browser, which is much cheaper than the full figure.
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
