# ---------------------------------------------------------------------------
# callbacks/backtesting.py – Backtesting tab callbacks
#
# This module contains all Dash callbacks and helper functions that power
# the Backtesting tab, where users build two asset baskets and compare how
# a monthly DCA (Dollar-Cost Averaging) strategy would have performed.
#
# The callbacks are grouped by feature:
#
#   Asset-class selectors   – bt_assetclass_a / bt_assetclass_b
#                             Populate each basket's asset dropdown when
#                             the user picks an asset class radio button.
#
#   Live search             – bt_search_a / bt_search_b
#                             Refine each basket's dropdown options as the
#                             user types (same ranked scoring as the Market
#                             Data tab).
#
#   Basket management       – manage_basket_a / manage_basket_b
#                             Handle "＋ add" and "✕ remove" actions for
#                             each basket's asset list.
#
#   Date-range slider       – update_date_range_slider
#                             Recomputes the intersection of all assets'
#                             available date ranges and configures the slider.
#
#   Date display            – update_date_display
#                             Shows the selected date range as a human-readable
#                             string while the user drags the slider.
#
#   Run backtest            – run_backtest_callback
#                             Executes the DCA simulation for both baskets,
#                             plots the portfolio value curves, and populates
#                             the metrics comparison table.
#
# Shared helper functions (_bt_assetclass_options, _bt_asset_search,
# _manage_basket, _build_slider_marks) contain the logic that is common to
# the A/B basket pairs so it is not duplicated.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# numpy: fast numerical array library. Used for np.select (vectorised
# conditional scoring) to rank search results efficiently.
import numpy as np

# pandas: tabular data library. Used for date arithmetic (date_range,
# Timestamp) and to pass data between the simulation and the chart.
import pandas as pd

# Plotly graph objects: the low-level chart primitives.
#   go.Figure   – container for all chart elements.
#   go.Scatter  – line chart trace used for portfolio value curves.
import plotly.graph_objects as go

# Dash callback utilities:
#   Input    – marks a callback argument as a trigger.
#   Output   – marks a return value as a target component property.
#   State    – read at fire-time but does NOT trigger the callback.
#   callback – standalone decorator that registers a Dash callback.
#   no_update – sentinel: tells Dash to leave an Output unchanged.
#   ALL      – pattern-match wildcard that targets ALL components whose
#              id dict shares a given 'type' key. Used to listen to every
#              remove button in a basket at once without knowing in advance
#              how many there will be.
from dash import (
    Input, Output, State, callback, no_update, ALL, MATCH,
    clientside_callback, ClientsideFunction,
)
from dash.exceptions import PreventUpdate

# dash.callback_context: provides runtime information about the callback
# that just fired (which Input triggered it, what its new value is, etc.).
# Only available inside a running callback; imported from the dash namespace
# so it can be patched in tests with patch('dash.callback_context', ...).
import dash

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

# config is imported as a module object so that df, base_url, and log are
# resolved at call time. This allows test patches on config_module to be
# visible to the callback functions at the moment they execute.
import src.config as _config

# log_time: decorator that measures and logs each callback's wall-clock
# execution time at DEBUG level.
from src.utils import log_time

# UI helper functions for rendering basket contents and the metrics table.
# These live in components.py because they build Dash component trees that
# are also needed elsewhere (e.g. layout.py builds _basket_ui panels).
from src.components import (
    _render_basket_list, _metrics_table, _build_strategy_params_ui, _transaction_section,
)

# The DCA simulation engine. run_backtest orchestrates data loading, the
# monthly investment simulation, and metric computation. get_common_date_range
# finds the date window that all selected assets share in common.
# Imported at module scope so tests can patch 'src.callbacks.backtesting.run_backtest'
# and 'src.callbacks.backtesting.get_common_date_range'.
from src.backtest import run_backtest, get_common_date_range, BacktestRun
from src.strategies.registry import get_strategy

# Trace/tab colour palette, assigned to runs by position.  The first two entries
# keep the established Basket A (blue) / Basket B (red) colours; further entries
# support future N-way comparisons without code changes.
_RUN_COLORS = ['#1a56db', '#c0392b', '#2e8b57', '#d97706', '#7c3aed', '#0891b2']

# Name of the transient marker trace appended to the chart when a transaction
# row is clicked.  Kept constant so each click replaces the previous marker.
_HIGHLIGHT_TRACE = '__row_highlight__'


# ---------------------------------------------------------------------------
# Strategy helper functions
# ---------------------------------------------------------------------------

def _get_strategy_instance(config: dict | None):
    """Instantiate the strategy class named in *config*, or return None.

    Parameters
    ----------
    config – dict with keys 'strategy' (str) and 'params' (dict), as stored
             in bt-strategy-config-store-{x}.  None or missing keys are safe.
    """
    if not config:
        return None
    name = config.get('strategy')
    if not name:
        return None
    try:
        return get_strategy(name)()
    except KeyError:
        return None


def _build_strategy_config(
    strategy_name: str | None,
    param_values: list,
    inputs_meta: list,
) -> dict:
    """Build a strategy config dict from the current strategy dropdown + param inputs.

    Parameters
    ----------
    strategy_name – selected strategy name from the dropdown.
    param_values  – list of current input values (from ALL pattern-match).
    inputs_meta   – callback_context.inputs_list entry for the ALL pattern;
                    each element has {'id': {'type': ..., 'index': key}, ...}.
    """
    if not strategy_name:
        return {'strategy': None, 'params': {}}
    try:
        strategy_cls = get_strategy(strategy_name)
    except KeyError:
        return {'strategy': strategy_name, 'params': {}}

    # Start from declared defaults so missing inputs never cause KeyError in run().
    params: dict = {p.key: p.default for p in strategy_cls.get_config_schema()}

    # Override with the values currently shown in the UI (may be fewer than
    # schema params if the UI is still updating after a strategy switch).
    for i, meta in enumerate(inputs_meta):
        key = meta['id']['index']
        val = param_values[i] if i < len(param_values) else None
        if val is not None:
            params[key] = val

    return {'strategy': strategy_name, 'params': params}


# ---------------------------------------------------------------------------
# Callbacks: asset-class selectors (one per basket)
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

    Parameters
    ----------
    asset_class : str or None – the chosen asset class (e.g. 'stocks').

    Returns
    -------
    ([], True)        – empty options + dropdown disabled (no class chosen yet).
    (options, False)  – populated options + dropdown enabled.
    """
    if not asset_class or _config.df is None:
        # No class selected yet: return empty options and keep the dropdown
        # greyed out (disabled=True).
        return [], True

    # Filter to the chosen class and cap at 200 rows for the initial list.
    # Typing in the dropdown triggers bt_search_a/b for finer results.
    filtered = _config.df[_config.df['asset_class'] == asset_class].head(200)
    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]
    # False = not disabled (i.e. the dropdown is now enabled).
    return options, False


# ---------------------------------------------------------------------------
# Callbacks: live search within each basket's dropdown
# ---------------------------------------------------------------------------

@callback(
    # allow_duplicate=True: both bt_assetclass_a and bt_search_a write to
    # bt-asset-a's 'options'. Dash requires explicit permission for that.
    Output('bt-asset-a', 'options', allow_duplicate=True),
    Input('bt-asset-a', 'search_value'),
    State('bt-assetclass-a', 'value'),
    State('bt-asset-a', 'value'),
    prevent_initial_call=True,
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
    prevent_initial_call=True,
)
@log_time
def bt_search_b(search_value, asset_class, current_value):
    """Refine Basket B's dropdown options as the user types a search query."""
    return _bt_asset_search(search_value, asset_class, current_value)


def _bt_asset_search(search_value, asset_class, current_value):
    """Shared search logic for both basket dropdowns.

    Uses the same 0–4 / 99 scoring algorithm as the Market Data tab's
    update_asset_search: exact symbol match scores 0 (best), no match
    scores 99 (excluded). Rows are sorted ascending so better matches
    appear at the top of the dropdown list.

    Parameters
    ----------
    search_value  : str or None – the text the user has typed.
    asset_class   : str or None – the currently selected asset class.
    current_value : str or None – the filename of the currently selected
                                  asset, kept in the options list even when
                                  it does not match the query string.

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

        # Score each row: 0 = exact symbol match (best), 99 = no match.
        # np.select evaluates conditions in order; first match wins.
        score = np.select(
            [sym == sl,
             sym.str.startswith(sl, na=False),
             name.str.startswith(sl, na=False),
             sym.str.contains(sl, na=False),
             name.str.contains(sl, na=False)],
            [0, 1, 2, 3, 4],
            default=99,
        )
        mask = score < 99
        filtered = (filtered[mask]
                    .assign(_score=score[mask])
                    .sort_values('_score')
                    .drop(columns='_score')
                    .head(30))
    else:
        # No query text: show the first 30 assets for this class.
        filtered = filtered.head(30)

    options = [
        {'label': f"{row['symbol']} — {row['name']} ({row['interval']})", 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]

    # Keep the currently selected asset in the options list even if it does
    # not match the current query string (prevents silent deselection on
    # each keystroke).
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
# Callbacks: basket management (add / remove assets)
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
    prevent_initial_call=True,
)
@log_time
def manage_basket_a(add_clicks, remove_clicks, selected_asset, basket_data):
    """Handle add/remove actions for Basket A."""
    return _manage_basket('a', remove_clicks, selected_asset, basket_data)


@callback(
    Output('bt-basket-store-b', 'data'),
    Output('bt-basket-list-b', 'children'),
    Input('bt-add-b', 'n_clicks'),
    Input({'type': 'bt-remove-b', 'index': ALL}, 'n_clicks'),
    State('bt-asset-b', 'value'),
    State('bt-basket-store-b', 'data'),
    prevent_initial_call=True,
)
@log_time
def manage_basket_b(add_clicks, remove_clicks, selected_asset, basket_data):
    """Handle add/remove actions for Basket B."""
    return _manage_basket('b', remove_clicks, selected_asset, basket_data)


def _manage_basket(basket_id, remove_clicks, selected_asset, basket_data):
    """Core logic for adding/removing an asset from a basket.

    Inspects dash.callback_context to determine whether the add button or a
    remove button triggered this callback, then mutates a copy of basket_data
    accordingly.

    Parameters
    ----------
    basket_id      : str  – 'a' or 'b'.
    remove_clicks  : list – n_clicks for each remove button (may be empty).
    selected_asset : str  – filename of the asset currently selected in the
                            dropdown (or None if nothing is selected).
    basket_data    : list – current list of asset dicts in the dcc.Store.
                            Each dict has keys 'filename', 'symbol', 'name'.

    Returns
    -------
    (updated_basket, updated_list_component)
      updated_basket         : list – new basket_data to persist in the Store.
      updated_list_component : Dash component – refreshed visible item list.
    or (no_update, no_update) when nothing should change.
    """
    # dash.callback_context provides runtime information about the callback
    # that just fired. Only available inside a callback function.
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
    # case where basket_data is None (Store initialised but never written).
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

    elif triggered_id == f'bt-add-{basket_id}' and selected_asset and _config.df is not None:
        # The add button was clicked and the dropdown has a selection.

        # Prevent duplicate entries: only add if the asset is not already in
        # the basket. any() returns True as soon as one match is found.
        if not any(item['filename'] == selected_asset for item in basket):
            meta = _config.df[_config.df['filename'] == selected_asset]
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
      ≤ 12 months  → every month labelled (step=1).
      13–36 months → every 12 months / yearly (step=12).
      > 36 months  → every 24 months / bi-yearly (step=24).

    The first position is always labelled so the user can always read the
    absolute left boundary of the available data.

    Parameters
    ----------
    date_range : DatetimeIndex – ordered monthly dates (from pd.date_range).

    Returns
    -------
    dict mapping integer slider position → short date string.
    """
    n = len(date_range)
    # Choose how many months to skip between each visible mark.
    step = 1 if n <= 12 else 12 if n <= 36 else 24
    # Use month abbreviation for short ranges, year only for longer ones.
    fmt = '%b' if n <= 12 else '%Y'
    marks = {}
    for i, d in enumerate(date_range):
        if i % step == 0:
            marks[i] = d.strftime(fmt)
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
    monthly date bounds, then intersects all ranges to find the overlap common
    to every selected asset. Updates the RangeSlider bounds and marks to
    reflect this common window, and stores the ordered list of dates so the
    run callback can look up exact Timestamps from slider integer positions.

    Parameters
    ----------
    basket_a : list of dicts or None – contents of Basket A's Store.
    basket_b : list of dicts or None – contents of Basket B's Store.

    Returns
    -------
    Seven values matching the seven Output declarations above.
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
        return (*_disabled[:6], 'Add assets to a basket to see the available date range.')

    if not _config.base_url or _config.df is None:
        return (*_disabled[:6], 'No data source configured.')

    # Compute the intersection of all asset date ranges across both baskets.
    # get_common_date_range returns (None, None) when there is no overlap.
    common_start, common_end = get_common_date_range(
        _config.base_url, filenames_a, filenames_b, _config.df,
    )

    if common_start is None or common_end is None:
        return (*_disabled[:6], 'No overlapping date range found across the selected assets.')

    # Build the ordered list of month-end dates within the common window.
    # pd.date_range with freq='ME' generates one date per calendar month-end.
    date_range = pd.date_range(common_start, common_end, freq='ME')
    n = len(date_range)

    # Serialise dates as ISO strings so they can be stored in dcc.Store (which
    # holds JSON). pd.Timestamp.isoformat() produces e.g.
    # '2020-01-31T00:00:00+00:00'.
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

    The slider reports integer indices; we look them up in the date_store
    list to get the actual Timestamps and format them for display.

    Parameters
    ----------
    slider_value : list or None – [start_index, end_index] from the slider.
    date_store   : list or None – ordered ISO date strings, one per slider step.

    Returns
    -------
    str – formatted date range string, e.g. "Selected: Jan 2020 – Dec 2024 (60 months)".
    or no_update if the store has not been populated yet.
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
# Callbacks: enable/disable the strategy dropdown based on basket contents
# ---------------------------------------------------------------------------

@callback(
    Output({'type': 'bt-strategy', 'basket': 'a'}, 'disabled'),
    Input('bt-basket-store-a', 'data'),
)
@log_time
def toggle_strategy_dropdown_a(basket_data: list | None) -> bool:
    """Disable the strategy dropdown for basket A when the basket is empty."""
    return not bool(basket_data)


@callback(
    Output({'type': 'bt-strategy', 'basket': 'b'}, 'disabled'),
    Input('bt-basket-store-b', 'data'),
)
@log_time
def toggle_strategy_dropdown_b(basket_data: list | None) -> bool:
    """Disable the strategy dropdown for basket B when the basket is empty."""
    return not bool(basket_data)


# ---------------------------------------------------------------------------
# Callback: render strategy parameter inputs when strategy changes (both baskets)
# ---------------------------------------------------------------------------

@callback(
    Output({'type': 'bt-strategy-params', 'basket': MATCH}, 'children'),
    Input({'type': 'bt-strategy', 'basket': MATCH}, 'value'),
    State('bt-basket-store-a', 'data'),
    State('bt-basket-store-b', 'data'),
    prevent_initial_call=True,
)
@log_time
def render_strategy_params(
    strategy_name: str | None,
    basket_a: list | None,
    basket_b: list | None,
) -> list:
    """Re-render param inputs whenever the user picks a different strategy.

    Uses MATCH so a single callback serves both basket A and basket B.
    basket_id is extracted from the triggering component's dict ID.
    Params are rendered disabled when the corresponding basket is empty.
    """
    basket_id = dash.callback_context.triggered_id['basket']
    basket_data = basket_a if basket_id == 'a' else basket_b
    disabled = not bool(basket_data)
    return _build_strategy_params_ui(strategy_name, basket_id, disabled=disabled)


# ---------------------------------------------------------------------------
# Callbacks: disable/enable param inputs when basket becomes empty/non-empty
# ---------------------------------------------------------------------------

@callback(
    Output({'type': 'bt-param-a', 'index': ALL}, 'disabled'),
    Input('bt-basket-store-a', 'data'),
)
@log_time
def disable_params_a(basket_data: list | None) -> list:
    """Disable all basket-A param inputs when the basket is empty."""
    disabled = not bool(basket_data)
    return [disabled] * len(dash.callback_context.outputs_list)


@callback(
    Output({'type': 'bt-param-b', 'index': ALL}, 'disabled'),
    Input('bt-basket-store-b', 'data'),
)
@log_time
def disable_params_b(basket_data: list | None) -> list:
    """Disable all basket-B param inputs when the basket is empty."""
    disabled = not bool(basket_data)
    return [disabled] * len(dash.callback_context.outputs_list)


# ---------------------------------------------------------------------------
# Callbacks: keep strategy config stores in sync with UI (one per basket)
# ---------------------------------------------------------------------------

@callback(
    Output('bt-strategy-config-store-a', 'data'),
    Input({'type': 'bt-strategy', 'basket': 'a'}, 'value'),
    Input({'type': 'bt-param-a', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
@log_time
def update_strategy_config_a(strategy_name: str | None, param_values: list) -> dict:
    """Sync strategy config store for basket A with the current UI state."""
    return _build_strategy_config(
        strategy_name,
        param_values,
        dash.callback_context.inputs_list[1],
    )


@callback(
    Output('bt-strategy-config-store-b', 'data'),
    Input({'type': 'bt-strategy', 'basket': 'b'}, 'value'),
    Input({'type': 'bt-param-b', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
@log_time
def update_strategy_config_b(strategy_name: str | None, param_values: list) -> dict:
    """Sync strategy config store for basket B with the current UI state."""
    return _build_strategy_config(
        strategy_name,
        param_values,
        dash.callback_context.inputs_list[1],
    )


# ---------------------------------------------------------------------------
# Callback: run the DCA backtest and render results
# ---------------------------------------------------------------------------

def _collect_runs(basket_a, basket_b, start_date, end_date, cfg_a, cfg_b):
    """Run the backtest for each basket and return a list of BacktestRun.

    This is the single seam between the (currently two-basket) input UI and the
    generic, run-list-driven result rendering.  To support more baskets or
    basket×strategy comparisons in the future, only this function changes; the
    chart and transaction tables already iterate the returned list.

    Each run pairs a basket with the strategy selected for it; an empty basket
    yields a run with portfolio=None (skipped by the renderers).
    """
    specs = [
        ('a', 'Basket A', basket_a, cfg_a),
        ('b', 'Basket B', basket_b, cfg_b),
    ]
    runs = []
    for i, (run_id, basket_label, basket, cfg) in enumerate(specs):
        filenames = [item['filename'] for item in (basket or [])]
        strategy = _get_strategy_instance(cfg)
        params = (cfg or {}).get('params') or {}
        strat_name = (cfg or {}).get('strategy') or 'DCA'
        portfolio, metrics, events = (
            run_backtest(_config.base_url, filenames, start_date, end_date, _config.df,
                         strategy=strategy, strategy_params=params)
            if filenames else (None, None, None)
        )
        runs.append(BacktestRun(
            run_id=run_id,
            label=f'{basket_label} · {strat_name}',
            color=_RUN_COLORS[i % len(_RUN_COLORS)],
            portfolio=portfolio,
            metrics=metrics,
            events=events,
        ))
    return runs


def _build_chart(active_runs, start_date, end_date):
    """Build the portfolio-value line chart, one trace per active run.

    Trace order follows *active_runs*, which the graph-click callback relies on
    to map a clicked point's curveNumber back to its run_id.
    """
    fig = go.Figure()
    for run in active_runs:
        # round(2) avoids floating-point noise in hover tooltips.
        fig.add_trace(go.Scatter(
            x=run.portfolio.index,
            y=run.portfolio.round(2),
            name=run.label,
            line=dict(color=run.color, width=2),
        ))
    fig.update_layout(
        title='Portfolio Value',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (€)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    return fig


def _build_events_store(active_runs):
    """Build the bt-events-store payload used by the click interactions.

    Structure
    ---------
    {'order': [run_id, …],                 # matches chart trace order
     'rows':  {run_id: [{'date': iso, 'value': float}, …]}}
    """
    order = [run.run_id for run in active_runs]
    rows = {}
    for run in active_runs:
        rows[run.run_id] = [
            {'date': ev['date'].isoformat(), 'value': round(ev['value_post_trade'], 2)}
            for ev in (run.events or [])
        ]
    return {'order': order, 'rows': rows}


@callback(
    Output('bt-chart', 'figure'),         # the portfolio value line chart
    Output('bt-chart', 'style'),          # show/hide the chart container
    Output('bt-metrics', 'children'),     # the metrics comparison table
    Output('bt-tx-section', 'children'),  # the per-run transaction tab section
    Output('bt-events-store', 'data'),    # event metadata for click interactions
    Output('bt-status', 'children'),      # status / error message text
    Input('bt-run', 'n_clicks'),          # fires when the Run button is clicked
    State('bt-basket-store-a', 'data'),          # basket A contents (read, not trigger)
    State('bt-basket-store-b', 'data'),          # basket B contents
    State('bt-date-range', 'value'),             # [start_index, end_index] into date_store
    State('bt-date-store', 'data'),              # ISO date strings, one per slider step
    State('bt-strategy-config-store-a', 'data'),  # selected strategy + params for basket A
    State('bt-strategy-config-store-b', 'data'),  # selected strategy + params for basket B
    prevent_initial_call=True,  # do not run at page load (no data yet)
)
@log_time
def run_backtest_callback(n_clicks, basket_a, basket_b, slider_value, date_store,
                          strategy_config_a, strategy_config_b):
    """Execute the backtest for every run and update the UI.

    Steps:
      1. Validate that at least one basket has assets and dates are available.
      2. Convert slider indices to actual Timestamps.
      3. Build the run list via _collect_runs() (one run per basket+strategy).
      4. Plot every successful run on a single line chart.
      5. Build the metrics table and the per-run transaction tab section.

    Returns
    -------
    (figure, chart_style, metrics_children, tx_section, events_store, status).
    """
    empty_chart = go.Figure()
    hidden = {'width': '100%', 'display': 'none'}   # hide the chart div
    visible = {'width': '100%', 'display': 'block'}  # show the chart div

    # Require at least one basket to have assets before running.
    if not basket_a and not basket_b:
        return empty_chart, hidden, '', [], {}, 'Please fill at least one basket.'

    if not _config.base_url or _config.df is None:
        return empty_chart, hidden, '', [], {}, 'No data source available.'

    # Guard: the date slider must have been populated by update_date_range_slider
    # before the user can run. If it has not (e.g. all files failed to load),
    # we cannot resolve the slider positions to actual Timestamps.
    if not date_store or not slider_value or len(date_store) < 2:
        return empty_chart, hidden, '', [], {}, 'No date range available. Add assets first.'

    # Convert the slider's integer positions back to pandas Timestamps.
    start_date = pd.Timestamp(date_store[slider_value[0]])
    end_date = pd.Timestamp(date_store[slider_value[1]])

    # Run every basket+strategy combination.
    runs = _collect_runs(basket_a, basket_b, start_date, end_date,
                         strategy_config_a, strategy_config_b)
    active = [r for r in runs if r.portfolio is not None]

    # If every backtest failed (e.g. all parquet files missing), abort.
    if not active:
        return empty_chart, hidden, '', [], {}, 'No data available for the selected period.'

    fig = _build_chart(active, start_date, end_date)
    tx_section = _transaction_section(runs)
    events_store = _build_events_store(active)

    # Metrics table keeps its A/B columns for now (runs[0]/runs[1] are baskets
    # A and B); generalising it to N runs is a separate, later change.
    metrics_div = _metrics_table(runs[0].metrics, runs[1].metrics)

    months_shown = max(len(r.portfolio) for r in active)
    d0_label = start_date.strftime('%b %Y')
    d1_label = end_date.strftime('%b %Y')
    status = (
        f'Backtest complete – {d0_label} to {d1_label} ({months_shown} months). '
        + ', '.join(r.label for r in active) + '.'
    )

    _config.log.info("Backtest completed: %d months, %d run(s)", months_shown, len(active))

    return fig, visible, metrics_div, tx_section, events_store, status


# ---------------------------------------------------------------------------
# Callback: row click → highlight the corresponding point on the chart
# ---------------------------------------------------------------------------

@callback(
    Output('bt-chart', 'figure', allow_duplicate=True),
    Input({'type': 'bt-tx-row', 'run': ALL, 'index': ALL}, 'n_clicks'),
    State('bt-chart', 'figure'),
    State('bt-events-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def highlight_chart_point(_clicks, fig, store):
    """Mark the clicked transaction's date/value on the chart.

    Reads the triggering row's (run, index) from the pattern-matching id, looks
    up its date and value in the event store, and appends a single marker trace
    (replacing any previous one).  The marker is always the LAST trace so the
    run↔curveNumber mapping used by the graph-click callback stays intact.
    """
    triggered = dash.callback_context.triggered
    # Ignore the initial registration burst where every n_clicks is None/0.
    if not triggered or not triggered[0]['value']:
        raise PreventUpdate
    if not fig or not store:
        raise PreventUpdate

    row_id = dash.callback_context.triggered_id
    run_id = row_id['run']
    index = row_id['index']
    rows = (store.get('rows') or {}).get(run_id) or []
    if index >= len(rows):
        raise PreventUpdate
    row = rows[index]

    # Drop any previous highlight marker, then append a fresh one.
    fig['data'] = [t for t in fig['data'] if t.get('name') != _HIGHLIGHT_TRACE]
    fig['data'].append({
        'type': 'scatter',
        'x': [row['date']],
        'y': [row['value']],
        'mode': 'markers',
        'marker': {'size': 13, 'color': '#111', 'symbol': 'circle-open',
                   'line': {'width': 2, 'color': '#111'}},
        'name': _HIGHLIGHT_TRACE,
        'showlegend': False,
        'hoverinfo': 'skip',
    })
    return fig


# ---------------------------------------------------------------------------
# Clientside callbacks: DOM scrolling (buttons + graph-click → table row)
# ---------------------------------------------------------------------------

# Scroll-to-top / scroll-to-bottom buttons.  Pure DOM work, so it runs in the
# browser; the dummy Store output is required because every Dash callback must
# declare at least one Output.
clientside_callback(
    ClientsideFunction(namespace='transactions', function_name='scrollButtons'),
    Output('bt-tx-scroll-dummy', 'data'),
    Input({'type': 'bt-tx-top', 'run': ALL}, 'n_clicks'),
    Input({'type': 'bt-tx-bottom', 'run': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)

# Graph point click → activate the matching run's tab and scroll its table to
# (and highlight) the nearest event row.
clientside_callback(
    ClientsideFunction(namespace='transactions', function_name='graphClickToRow'),
    Output('bt-tx-tabs', 'active_tab'),
    Input('bt-chart', 'clickData'),
    State('bt-events-store', 'data'),
    prevent_initial_call=True,
)
