# ---------------------------------------------------------------------------
# callbacks/backtesting.py – Backtesting callbacks
#
# This module contains all Dash callbacks and helper functions that power
# the main page.  Users build ONE shared asset list; each asset's two
# membership checkboxes (Basket A / Basket B) decide which basket(s) it belongs
# to, and the two strategy panels compare how their chosen strategies would have
# performed over a selected date range.
#
# The callbacks are grouped by feature:
#
#   Asset-class selector    – bt_assetclass
#                             Populates the shared asset dropdown when the user
#                             picks an asset class radio button.
#
#   Live search             – bt_search
#                             Refines the dropdown options as the user types.
#
#   Basket management       – manage_basket
#                             Handles "＋ add" and "✕ remove" actions on the
#                             shared asset list (master list + membership).
#
#   Weights / membership    – sync_weights / sync_membership
#                             Persist per-asset weights and the A/B checkboxes
#                             and refresh the live per-basket allocation %s.
#
#   Basket derivation       – derive_baskets
#                             Splits the master list into the per-basket stores
#                             (bt-basket-store-a/b) every consumer below reads.
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
#                             Executes the simulation for both baskets,
#                             plots the portfolio value curves, and populates
#                             the metrics comparison table.
#
# Shared helper functions (_bt_assetclass_options, _bt_asset_search,
# _manage_basket, _sync_weights, _sync_membership, _derive_baskets,
# _build_slider_marks) hold the reusable logic and are unit-tested directly.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

from typing import Any

# time: high-resolution timer (perf_counter) to log how long the server spends
# building the figure + tables — distinct from the client-side render that
# happens in the browser after this callback returns (invisible to @log_time).
import time

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
from dash import Input, Output, State, callback, no_update, ALL, MATCH, dcc

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
    _render_basket_list, _metrics_table, _build_strategy_params_ui,
    _strategy_desc_panel, _order_rows, _order_table_component, _asset_currency_map,
    _basket_membership_pcts,
)

# The DCA simulation engine. run_backtest orchestrates data loading, the
# monthly investment simulation, and metric computation. get_common_date_range
# finds the date window that all selected assets share in common.
# Imported at module scope so tests can patch 'src.callbacks.backtesting.run_backtest'
# and 'src.callbacks.backtesting.get_common_date_range'.
from src.backtest import run_backtest, get_common_date_range
from src.strategies.registry import get_strategy


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
    Output('bt-asset', 'options'),
    Output('bt-asset', 'disabled'),
    Input('bt-assetclass', 'value'),
)
@log_time
def bt_assetclass(asset_class):
    """Populate the shared asset dropdown when the user picks an asset class."""
    return _bt_assetclass_options(asset_class)


def _asset_option_label(row) -> str:
    """Build a search-dropdown option label for one catalogue row.

    Shows "<symbol> — <name> (<interval> · <currency>)"; the currency tag is
    dropped for assets whose catalogue currency is blank/unknown (e.g. Indices)
    or absent (legacy catalogues without a ``currency`` column), so the user sees
    each asset's trading currency directly in the search results.
    """
    base = f"{row['symbol']} — {row['name']} ({row['interval']}"
    ccy = str(row.get('currency') or '').strip() if 'currency' in row.index else ''
    if ccy and ccy not in ('0', 'nan', 'None'):
        return f"{base} · {ccy})"
    return f"{base})"


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
        {'label': _asset_option_label(row), 'value': row['filename']}
        for _, row in filtered.iterrows()
    ]
    # False = not disabled (i.e. the dropdown is now enabled).
    return options, False


# ---------------------------------------------------------------------------
# Callbacks: live search within each basket's dropdown
# ---------------------------------------------------------------------------

@callback(
    # allow_duplicate=True: both bt_assetclass and bt_search write to
    # bt-asset's 'options'. Dash requires explicit permission for that.
    Output('bt-asset', 'options', allow_duplicate=True),
    Input('bt-asset', 'search_value'),
    State('bt-assetclass', 'value'),
    State('bt-asset', 'value'),
    prevent_initial_call=True,
)
@log_time
def bt_search(search_value, asset_class, current_value):
    """Refine the shared dropdown options as the user types a search query."""
    return _bt_asset_search(search_value, asset_class, current_value)


def _bt_asset_search(search_value, asset_class, current_value):
    """Shared search logic for both basket dropdowns.

    Uses a 0–4 / 99 scoring algorithm: exact symbol match scores 0 (best),
    no match scores 99 (excluded). Rows are sorted ascending so better
    matches appear at the top of the dropdown list.

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
        {'label': _asset_option_label(row), 'value': row['filename']}
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
                'label': _asset_option_label(row),
                'value': current_value,
            })
    return options


# ---------------------------------------------------------------------------
# Callbacks: basket management (add / remove assets)
# ---------------------------------------------------------------------------

@callback(
    Output('bt-basket-store', 'data'),        # updated master asset list
    Output('bt-membership-store', 'data'),    # updated per-asset A/B membership
    # Pruned on remove (a removed asset's weight is dropped); allow_duplicate
    # because sync_weights also writes this store on every weight edit.
    Output('bt-weights-store', 'data', allow_duplicate=True),
    Output('bt-basket-list', 'children'),     # refreshed visible list
    Input('bt-add', 'n_clicks'),              # add button clicked
    # ALL pattern: fires when *any* remove button is clicked.  The Input value
    # is a list of n_clicks, one per matching button.
    Input({'type': 'bt-remove', 'index': ALL}, 'n_clicks'),
    State('bt-asset', 'value'),               # currently selected asset filename
    State('bt-basket-store', 'data'),         # current master list (list of dicts)
    State('bt-weights-store', 'data'),        # current shared weights {filename: weight}
    State('bt-membership-store', 'data'),     # current membership {filename: {a, b}}
    prevent_initial_call=True,
)
@log_time
def manage_basket(add_clicks, remove_clicks, selected_asset, basket_data, weights, membership):
    """Handle add/remove actions on the shared asset list."""
    return _manage_basket(remove_clicks, selected_asset, basket_data, weights, membership)


def _manage_basket(remove_clicks, selected_asset, basket_data, weights=None, membership=None):
    """Core logic for adding/removing an asset from the shared list.

    Inspects dash.callback_context to determine whether the add button or a
    remove button triggered this callback, then mutates copies of the master
    list, the membership map, and the weights map accordingly.

    Parameters
    ----------
    remove_clicks  : list – n_clicks for each remove button (may be empty).
    selected_asset : str  – filename of the asset currently selected in the
                            dropdown (or None if nothing is selected).
    basket_data    : list – current master list of asset dicts (each with keys
                            'filename', 'symbol', 'name', 'currency').
    weights        : dict – current shared per-asset weights ({filename: weight});
                            an entry is dropped when its asset is removed.
    membership     : dict – current per-asset A/B membership
                            ({filename: {'a': bool, 'b': bool}}); a newly added
                            asset defaults to a member of BOTH baskets.

    Returns
    -------
    (basket, membership, weights, list_component) to persist / display,
    or (no_update, no_update, no_update, no_update) when nothing should change.
    """
    # dash.callback_context provides runtime information about the callback
    # that just fired. Only available inside a callback function.
    ctx = dash.callback_context

    # ctx.triggered is a list of dicts describing every Input that changed.
    # If it is empty the callback was fired spuriously; do nothing.
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    # ctx.triggered_id is the id of the single Input that actually caused the
    # callback to fire. For a plain button it is a string like 'bt-add'.  For a
    # pattern-matching button it is a dict like
    #   {'type': 'bt-remove', 'index': 'aapl.parquet'}.
    triggered_id = ctx.triggered_id

    # ctx.triggered[0]['value'] is the new property value that changed. For
    # n_clicks this is the new click count. .get('value', 0) or 0 handles
    # the case where Dash emits None for newly rendered components whose
    # n_clicks initialises to 0 – those should NOT trigger a removal.
    triggered_value = ctx.triggered[0].get('value', 0) or 0

    # Copy the stores so we can mutate them safely. 'or …' handles the case
    # where a store is None (initialised but never written).
    basket = list(basket_data or [])
    membership = dict(membership or {})
    weights = dict(weights or {})

    if isinstance(triggered_id, dict) and triggered_id.get('type') == 'bt-remove':
        # A remove button was clicked. The dict-id carries 'index' = filename.

        # Guard: newly rendered remove buttons fire the ALL-pattern callback
        # with n_clicks=0 when added to the layout. Skipping those prevents
        # a phantom removal every time a new asset is added to the list.
        if triggered_value > 0:
            filename = triggered_id['index']
            # Keep all items except the one whose filename matches, and drop its
            # membership + weight so the stores do not accumulate stale entries.
            basket = [item for item in basket if item['filename'] != filename]
            membership.pop(filename, None)
            weights.pop(filename, None)

    elif triggered_id == 'bt-add' and selected_asset and _config.df is not None:
        # The add button was clicked and the dropdown has a selection.

        # Prevent duplicate entries: only add if the asset is not already in
        # the list. any() returns True as soon as one match is found.
        if not any(item['filename'] == selected_asset for item in basket):
            meta = _config.df[_config.df['filename'] == selected_asset]
            if not meta.empty:
                row = meta.iloc[0]
                # Append a minimal dict: filename, symbol, display name and the
                # trading currency (blank when absent/unknown) so the list can
                # tag each asset with its currency.
                ccy = ''
                if 'currency' in row.index and pd.notna(row.get('currency')):
                    ccy = str(row['currency']).strip()
                basket.append({'filename': selected_asset, 'symbol': row['symbol'],
                               'name': row['name'], 'currency': ccy})
                # A newly added asset starts as a member of BOTH baskets so it is
                # immediately comparable under both strategies; the user unticks a
                # checkbox to exclude it from one basket.
                membership[selected_asset] = {'a': True, 'b': True}

    else:
        # The callback fired for some other reason (e.g. the remove button
        # list was updated with new buttons that all report n_clicks=0, or
        # the add button was clicked without an asset selected). Do nothing.
        return no_update, no_update, no_update, no_update

    # Return the new master list, membership and weights (written to their
    # dcc.Stores) plus the refreshed visible list component.  The weights /
    # membership maps keep each surviving asset's user-set values across the
    # add/remove; a freshly added asset defaults to weight 1.0 and both baskets.
    return basket, membership, weights, _render_basket_list(basket, weights, membership)


# ---------------------------------------------------------------------------
# Callbacks: per-asset weights (persist edits + live allocation percentages)
# ---------------------------------------------------------------------------

def _sync_weights(values, inputs_meta, pct_a_meta, pct_b_meta, membership):
    """Persist the shared weight inputs and recompute the per-basket allocation %.

    Builds the ``{filename: weight}`` map from the live weight inputs (a cleared or
    invalid field falls back to the default weight 1.0; negatives are clamped to
    0), then renders each asset's share-of-basket percentage for **both** baskets,
    each aligned to the order of its percentage-label outputs.  A single weight is
    shared by both baskets; only the renormalisation subset (each basket's members)
    differs, so the A and B percentages can differ.

    Parameters
    ----------
    values      – current value of every weight input (ALL pattern order).
    inputs_meta – callback_context.inputs_list entry for the weight inputs; each
                  element's ``id['index']`` is the asset's filename.
    pct_a_meta  – outputs_list entry for the Basket A percentage labels.
    pct_b_meta  – outputs_list entry for the Basket B percentage labels.
    membership  – the {filename: {'a', 'b'}} membership map (State), to know which
                  assets each basket's allocation renormalises over.

    Returns
    -------
    (weights_store, pct_a_children, pct_b_children)
      weights_store  – the {filename: weight} dict to persist.
      pct_a_children – Basket A percentage strings, one per A label output.
      pct_b_children – Basket B percentage strings, one per B label output.
    """
    weights: dict = {}
    for i, meta in enumerate(inputs_meta):
        filename = meta['id']['index']
        val = values[i] if i < len(values) else None
        try:
            weights[filename] = 1.0 if val is None else max(float(val), 0.0)
        except (TypeError, ValueError):
            weights[filename] = 1.0

    # _basket_membership_pcts only reads each item's 'filename', so a minimal
    # list built from the weight-input metas is enough to compute the shares.
    basket = [{'filename': meta['id']['index']} for meta in inputs_meta]
    pct_a, pct_b = _basket_membership_pcts(basket, weights, membership)
    a_children = [pct_a.get(meta['id']['index'], '—') for meta in pct_a_meta]
    b_children = [pct_b.get(meta['id']['index'], '—') for meta in pct_b_meta]
    return weights, a_children, b_children


@callback(
    # allow_duplicate: the weights store is also written by manage_basket (which
    # prunes a removed asset); the percentage labels are also written by
    # sync_membership, so both those outputs need allow_duplicate too.
    Output('bt-weights-store', 'data', allow_duplicate=True),
    Output({'type': 'bt-weight-pct-a', 'index': ALL}, 'children', allow_duplicate=True),
    Output({'type': 'bt-weight-pct-b', 'index': ALL}, 'children', allow_duplicate=True),
    Input({'type': 'bt-weight', 'index': ALL}, 'value'),
    State('bt-membership-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def sync_weights(weight_values, membership):
    """Persist shared weight edits and refresh both baskets' live allocation %s."""
    ctx = dash.callback_context
    # Guard: when the last weight input disappears (list emptied / replaced), the
    # ALL pattern fires with no inputs; writing {} would wipe the stored weights
    # for nothing, so leave the store untouched.
    if not ctx.inputs_list[0]:
        return no_update, [], []
    return _sync_weights(weight_values, ctx.inputs_list[0],
                         ctx.outputs_list[1], ctx.outputs_list[2], membership)


# ---------------------------------------------------------------------------
# Callbacks: per-asset A/B membership (which basket(s) each asset belongs to)
# ---------------------------------------------------------------------------

def _sync_membership(member_a, member_b, a_meta, b_meta, basket_data, weights,
                     membership_state, pct_a_meta, pct_b_meta):
    """Build the membership map from the A/B checkboxes and refresh the %s.

    Reads the two checkbox groups (Basket A / Basket B), assembles the
    ``{filename: {'a': bool, 'b': bool}}`` map, and recomputes each basket's
    allocation percentages under the *new* membership.  The store output is
    suppressed (``no_update``) when the assembled map already equals the stored
    one, so a checkbox re-render right after manage_basket does not needlessly
    re-fire derive_baskets (and the date slider).

    Returns
    -------
    (membership_or_no_update, pct_a_children, pct_b_children)
    """
    membership: dict = {}
    for i, meta in enumerate(a_meta):
        filename = meta['id']['index']
        membership.setdefault(filename, {})['a'] = bool(member_a[i]) if i < len(member_a) else True
    for i, meta in enumerate(b_meta):
        filename = meta['id']['index']
        membership.setdefault(filename, {})['b'] = bool(member_b[i]) if i < len(member_b) else True

    # Recompute the per-basket allocation %s with the freshly toggled membership.
    basket = basket_data or [{'filename': meta['id']['index']} for meta in a_meta]
    pct_a, pct_b = _basket_membership_pcts(basket, weights, membership)
    a_children = [pct_a.get(meta['id']['index'], '—') for meta in pct_a_meta]
    b_children = [pct_b.get(meta['id']['index'], '—') for meta in pct_b_meta]

    # No real change (e.g. the checkboxes were just re-rendered by manage_basket,
    # which already wrote this exact map) → keep the store to avoid a redundant
    # derive/date-slider round-trip.
    store_out = no_update if membership == (membership_state or {}) else membership
    return store_out, a_children, b_children


@callback(
    # allow_duplicate: the membership store is also written by manage_basket (new
    # asset defaults), and the percentage labels are also written by sync_weights.
    Output('bt-membership-store', 'data', allow_duplicate=True),
    Output({'type': 'bt-weight-pct-a', 'index': ALL}, 'children', allow_duplicate=True),
    Output({'type': 'bt-weight-pct-b', 'index': ALL}, 'children', allow_duplicate=True),
    Input({'type': 'bt-member-a', 'index': ALL}, 'value'),
    Input({'type': 'bt-member-b', 'index': ALL}, 'value'),
    State('bt-basket-store', 'data'),
    State('bt-weights-store', 'data'),
    State('bt-membership-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def sync_membership(member_a, member_b, basket_data, weights, membership_state):
    """Persist the A/B membership checkboxes and refresh both baskets' %s."""
    ctx = dash.callback_context
    a_meta = ctx.inputs_list[0]
    b_meta = ctx.inputs_list[1]
    # Guard: no matching checkboxes (list emptied / replaced) → writing {} would
    # wipe the membership; leave it untouched.
    if not a_meta and not b_meta:
        return no_update, [], []
    return _sync_membership(member_a, member_b, a_meta, b_meta, basket_data, weights,
                            membership_state, ctx.outputs_list[1], ctx.outputs_list[2])


# ---------------------------------------------------------------------------
# Callbacks: derive the per-basket stores from the master list + membership
# ---------------------------------------------------------------------------

def _derive_baskets(master, membership):
    """Split the master asset list into Basket A and Basket B by membership.

    An asset belongs to a basket when its membership flag for that basket is
    truthy; a **missing** entry (or flag) counts as a member of both baskets, so
    a freshly added asset is already in both even before sync_membership persists
    its default — keeping derive_baskets consistent with the add default.

    Parameters
    ----------
    master     – the shared asset list ([{filename, symbol, name, currency}, …]).
    membership – the {filename: {'a': bool, 'b': bool}} map.

    Returns
    -------
    (basket_a, basket_b) – the two filtered asset lists.
    """
    master = master or []
    membership = membership or {}
    basket_a = [item for item in master if membership.get(item['filename'], {}).get('a', True)]
    basket_b = [item for item in master if membership.get(item['filename'], {}).get('b', True)]
    return basket_a, basket_b


@callback(
    # Sole writer of the two per-basket stores every downstream callback reads.
    Output('bt-basket-store-a', 'data'),
    Output('bt-basket-store-b', 'data'),
    Input('bt-basket-store', 'data'),        # the master asset list
    Input('bt-membership-store', 'data'),    # per-asset A/B membership
)
@log_time
def derive_baskets(master, membership):
    """Fan the master list out into Basket A / Basket B by their membership flags."""
    return _derive_baskets(master, membership)


def _symbol_weights(basket_items: list | None, weights_store: dict | None) -> dict | None:
    """Map each basket asset's symbol to its relative weight for the simulation.

    Looks each asset's filename up in *weights_store* (default 1.0 when absent or
    invalid; negatives clamped to 0).  Returns ``None`` — meaning "equal weight" to
    the engines — when the basket is empty or every weight is non-positive, so the
    common untouched case behaves exactly as before weighting existed.

    The engines key weights by **symbol** (the price-frame columns), so this
    translates the filename-keyed store into the symbol space.
    """
    store = weights_store or {}
    out: dict[str, float] = {}
    for item in basket_items or []:
        raw = store.get(item['filename'], 1.0)
        try:
            weight = max(float(raw), 0.0)
        except (TypeError, ValueError):
            weight = 1.0
        out[str(item['symbol'])] = weight
    if not out or sum(out.values()) <= 0.0:
        return None
    return out


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
    if slider_value is None or not date_store:
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
    Output({'type': 'bt-strategy-desc', 'basket': MATCH}, 'children'),
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
) -> tuple[list, list] | Any:
    """Re-render param inputs and the description panel when the strategy changes.

    Uses MATCH so a single callback serves both basket A and basket B.
    basket_id is extracted from the triggering component's dict ID.
    Params are rendered disabled when the corresponding basket is empty; the
    rich-text description panel is refilled for the newly selected strategy.
    """
    tid = dash.callback_context.triggered_id
    if not isinstance(tid, dict):
        return no_update
    basket_id = tid['basket']
    basket_data = basket_a if basket_id == 'a' else basket_b
    disabled = not bool(basket_data)
    return (
        _build_strategy_params_ui(strategy_name, basket_id, disabled=disabled),
        _strategy_desc_panel(strategy_name),
    )


# ---------------------------------------------------------------------------
# Callback: toggle the strategy description panel open/closed (both baskets)
# ---------------------------------------------------------------------------

@callback(
    Output({'type': 'bt-strategy-desc-collapse', 'basket': MATCH}, 'is_open'),
    Input({'type': 'bt-strategy-desc-toggle', 'basket': MATCH}, 'n_clicks'),
    State({'type': 'bt-strategy-desc-collapse', 'basket': MATCH}, 'is_open'),
    prevent_initial_call=True,
)
@log_time
def toggle_strategy_desc(n_clicks: int | None, is_open: bool) -> bool:
    """Flip the description panel's open state each time the ⓘ button is clicked.

    Uses MATCH so a single callback serves both basket A and basket B.
    """
    return not is_open


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
# Helper: thin a daily curve for plotting (client render is the bottleneck)
# ---------------------------------------------------------------------------

# Max points drawn per chart trace.  A multi-year *daily* curve has thousands
# of points, but the chart is only ~1-2k pixels wide, so drawing every point
# just bloats the JSON payload and the browser's SVG render (the dominant cost
# of a run) with no visible difference.  Only the plotted curve is thinned —
# every metric is still computed on the full-resolution series.
_MAX_PLOT_POINTS = 2000


def _downsample_for_plot(series: pd.Series, max_points: int = _MAX_PLOT_POINTS) -> pd.Series:
    """Reduce *series* to at most *max_points* points for plotting, keeping shape.

    Returns the series unchanged when it is already small enough.  Otherwise it
    keeps every k-th point (k chosen so the result is ~max_points) and always
    appends the final point so the curve still ends on the true last value.
    """
    n = len(series)
    if n <= max_points:
        return series
    step = (n // max_points) + 1
    thinned = series.iloc[::step]
    if thinned.index[-1] != series.index[-1]:
        thinned = pd.concat([thinned, series.iloc[[-1]]])
    return thinned


# ---------------------------------------------------------------------------
# Callback: run the DCA backtest and render results
# ---------------------------------------------------------------------------

@callback(
    Output('bt-chart', 'figure'),       # the portfolio value line chart
    Output('bt-chart', 'style'),        # show/hide the chart container
    Output('bt-metrics', 'children'),   # the metrics comparison table
    Output('bt-status', 'children'),    # status / error message text
    # Both baskets' order tables, as native-HTML strings keyed 'a'/'b'.  A
    # separate callback (render_order_table) renders the active tab's table into
    # the always-visible bt-orders-content div, so the table never lives inside a
    # display:none Bootstrap tab pane (which Safari fails to repaint on update).
    Output('bt-orders-store', 'data'),
    Input('bt-run', 'n_clicks'),        # fires when the Run button is clicked
    State('bt-basket-store-a', 'data'),          # basket A contents (read, not trigger)
    State('bt-basket-store-b', 'data'),          # basket B contents
    State('bt-date-range', 'value'),             # [start_index, end_index] into date_store
    State('bt-date-store', 'data'),              # ISO date strings, one per slider step
    State('bt-strategy-config-store-a', 'data'),  # selected strategy + params for basket A
    State('bt-strategy-config-store-b', 'data'),  # selected strategy + params for basket B
    State('bt-base-currency', 'value'),           # reporting currency for both baskets
    State('bt-weights-store', 'data'),            # shared per-asset weights (both baskets)
    prevent_initial_call=True,  # do not run at page load (no data yet)
)
@log_time
def run_backtest_callback(n_clicks, basket_a, basket_b, slider_value, date_store,
                          strategy_config_a, strategy_config_b, base_currency,
                          weights=None):
    """Execute the DCA simulation for both baskets and update the UI.

    Steps:
      1. Validate that at least one basket has assets and dates are available.
      2. Convert slider indices to actual Timestamps.
      3. Run the backtest for each non-empty basket via run_backtest().
      4. Plot both portfolios on a single line chart.
      5. Build the side-by-side metrics comparison table.
      6. Stash both baskets' per-order table HTML in the orders store (the active
         one is rendered into the page by render_order_table).

    Parameters
    ----------
    n_clicks          : int  – click count on the Run button (unused; the click
                               itself is the trigger, not its count).
    basket_a          : list – asset dicts for Basket A (may be None or empty).
    basket_b          : list – asset dicts for Basket B (may be None or empty).
    slider_value      : list – [start_index, end_index] into date_store.
    date_store        : list – ISO date strings for each slider position.
    strategy_config_a : dict – {'strategy': name, 'params': {...}} for basket A.
    strategy_config_b : dict – {'strategy': name, 'params': {...}} for basket B.
    base_currency     : str  – reporting currency both baskets are converted into
                               (every asset's prices, metrics and order log).
    weights           : dict – the shared per-asset weights ({filename: weight});
                               each basket's subset is translated to symbol-keyed
                               weights and forwarded to its simulation (None/all-zero
                               → equal weight).  One weight applies to both baskets.

    Returns
    -------
    (figure, chart_style, metrics_children, status_text, orders_store)
      figure          : go.Figure with A and/or B portfolio value traces.
      chart_style     : dict – {'display': 'block'} or {'display': 'none'}.
      metrics_children: Dash component – the comparison table or empty string.
      status_text     : str – summary or error message shown below the button.
      orders_store    : dict {'a': html|None, 'b': html|None} – each basket's
                        order-table HTML; render_order_table shows the active one.
    """
    empty_chart = go.Figure()
    empty_orders: dict = {}                          # empty orders-store payload
    hidden = {'width': '100%', 'display': 'none'}   # hide the chart div
    visible = {'width': '100%', 'display': 'block'}  # show the chart div

    # Require at least one basket to have assets before running.
    if not basket_a and not basket_b:
        return empty_chart, hidden, '', 'Please fill at least one basket.', empty_orders

    if not _config.base_url or _config.df is None:
        return empty_chart, hidden, '', 'No data source available.', empty_orders

    # Guard: the date slider must have been populated by update_date_range_slider
    # before the user can run. If it has not (e.g. all files failed to load),
    # we cannot resolve the slider positions to actual Timestamps.
    if date_store is None or slider_value is None or len(date_store) < 2:
        return (empty_chart, hidden, '', 'No date range available. Add assets first.',
                empty_orders)

    # Convert the slider's integer positions back to pandas Timestamps.
    # slider_value is [i0, i1]; date_store[i0] is an ISO string like
    # '2020-01-31T00:00:00+00:00'. pd.Timestamp() parses it correctly.
    start_date = pd.Timestamp(date_store[slider_value[0]])
    end_date = pd.Timestamp(date_store[slider_value[1]])

    # Extract filenames from each basket's list of asset dicts.
    filenames_a = [item['filename'] for item in (basket_a or [])]
    filenames_b = [item['filename'] for item in (basket_b or [])]

    # Fall back to the configured default when the dropdown value is somehow
    # missing (e.g. a stale client state), so conversion always has a target.
    base_currency = base_currency or _config.default_base_currency

    # Resolve strategy instances from the per-basket config stores.  If a
    # store is absent or its 'strategy' key is missing, _get_strategy_instance
    # returns None and run_backtest falls back to the built-in DCA code path.
    strategy_a = _get_strategy_instance(strategy_config_a)
    strategy_b = _get_strategy_instance(strategy_config_b)
    params_a = (strategy_config_a or {}).get('params') or {}
    params_b = (strategy_config_b or {}).get('params') or {}

    # Translate the shared (filename-keyed) weights into the symbol-keyed form the
    # engines expect, once per basket over that basket's assets; None means equal
    # weight (the untouched default).  The same weights map serves both baskets —
    # _symbol_weights filters it down to each basket's own symbols.
    weights_sym_a = _symbol_weights(basket_a, weights)
    weights_sym_b = _symbol_weights(basket_b, weights)

    # Run the simulation for each basket, but skip the call entirely if the
    # basket has no assets (saves an unnecessary function call).
    # run_backtest returns (portfolio_series, metrics_dict, order_log) on
    # success, or (None, None, None) if no data was available for the period.
    portfolio_a, metrics_a, orders_a = (
        run_backtest(_config.base_url, filenames_a, start_date, end_date, _config.df,
                     strategy=strategy_a, strategy_params=params_a,
                     base_currency=base_currency, weights=weights_sym_a)
        if filenames_a else (None, None, None)
    )
    portfolio_b, metrics_b, orders_b = (
        run_backtest(_config.base_url, filenames_b, start_date, end_date, _config.df,
                     strategy=strategy_b, strategy_params=params_b,
                     base_currency=base_currency, weights=weights_sym_b)
        if filenames_b else (None, None, None)
    )

    # If both backtests failed (e.g. all parquet files missing), abort.
    if portfolio_a is None and portfolio_b is None:
        return (empty_chart, hidden, '', 'No data available for the selected period.',
                empty_orders)

    # --- diagnostics: how much is about to be serialised to the browser? ---
    # The figure point count (= portfolio length) is the prime suspect for a
    # slow *client-side* render: that happens in the browser AFTER this callback
    # returns, so it never appears in this callback's @log_time total.
    n_pts_a = len(portfolio_a) if portfolio_a is not None else 0
    n_pts_b = len(portfolio_b) if portfolio_b is not None else 0
    n_ord_a = len(orders_a) if orders_a else 0
    n_ord_b = len(orders_b) if orders_b else 0
    _config.log.debug('[perf] render input: A=%d points / %d orders, B=%d points / %d orders',
                      n_pts_a, n_ord_a, n_pts_b, n_ord_b)
    _t_render = time.perf_counter()

    # Trace / metric-column labels.  Both baskets may hold different asset subsets
    # AND run different strategies, so label by BOTH the basket letter and the
    # strategy ('A: Buy & Hold' / 'B: DCA') — unambiguous whether the two baskets
    # share assets (then the strategy distinguishes them) or not.
    name_a = (strategy_config_a or {}).get('strategy') or 'DCA'
    name_b = (strategy_config_b or {}).get('strategy') or 'DCA'
    label_a, label_b = f'A: {name_a}', f'B: {name_b}'

    # Build the portfolio value chart.
    fig = go.Figure()

    if portfolio_a is not None:
        # go.Scatter draws a line chart. round(2) avoids floating-point noise
        # in hover tooltips (e.g. 1000.0000000002 → 1000.0).  The curve is
        # thinned to ~_MAX_PLOT_POINTS first (see _downsample_for_plot).
        plot_a = _downsample_for_plot(portfolio_a)
        fig.add_trace(go.Scatter(
            x=plot_a.index,       # x-axis: daily dates
            y=plot_a.round(2),    # y-axis: portfolio value in the base currency
            name=label_a,
            line=dict(color='#1a56db', width=2),  # blue line, 2px thick
        ))

    if portfolio_b is not None:
        plot_b = _downsample_for_plot(portfolio_b)
        fig.add_trace(go.Scatter(
            x=plot_b.index,
            y=plot_b.round(2),
            name=label_b,
            line=dict(color='#c0392b', width=2),  # red line
        ))

    # Length of the selected window in months. Derived from the date range
    # (not len(portfolio), which is now a daily count) so the status line still
    # reads in months even though the curves are daily. Computed as an integer
    # month index to avoid the tz-dropping warning that to_period() would emit.
    n_months = (
        (end_date.year * 12 + end_date.month)
        - (start_date.year * 12 + start_date.month) + 1
    )

    d0_label = start_date.strftime('%b %Y')
    d1_label = end_date.strftime('%b %Y')
    fig.update_layout(
        title='Portfolio Value',
        xaxis_title='Date',
        yaxis_title=f'Portfolio Value ({base_currency})',
        # hovermode='x unified': a single tooltip shows values for ALL traces
        # at the hovered x-position instead of separate tooltips per trace.
        hovermode='x unified',
        # Horizontal legend above the chart to save vertical space.
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        # Tight margins to maximise plot area; t=48 leaves room for the title.
        margin=dict(l=8, r=8, t=48, b=8),
    )

    # Build the side-by-side metrics table and assemble the status message.
    metrics_div = _metrics_table(metrics_a, metrics_b, label_a, label_b)

    # Format each basket's order log into display rows (None for an empty/failed
    # basket) and stash both in the store.  render_order_table renders the active
    # one into the page; download_orders exports it as CSV / Excel.  The per-asset
    # currency map lets _order_rows label each asset's trading-currency price
    # column (and decide whether a separate base-currency column is needed).
    orders_store = {
        'a': _order_rows(orders_a, base_currency, _asset_currency_map(filenames_a, _config.df)),
        'b': _order_rows(orders_b, base_currency, _asset_currency_map(filenames_b, _config.df)),
    }

    # Server-side cost of assembling the figure + metric/order tables. If this is
    # small but the user still waits seconds, the time is the browser rendering
    # the payload sized by the point/order counts logged above.
    _config.log.debug('[perf] build figure+tables: %.1fms',
                      (time.perf_counter() - _t_render) * 1000)

    status = (
        f'Backtest complete – {d0_label} to {d1_label} ({n_months} months). '
        f'Strategy A: {name_a}, Strategy B: {name_b}.'
    )

    _config.log.info("Backtest completed: %d months, A=%s (%s), B=%s (%s)",
                     n_months, len(filenames_a), name_a, len(filenames_b), name_b)

    # Return five values matching the five Output declarations above.
    return fig, visible, metrics_div, status, orders_store


# ---------------------------------------------------------------------------
# Callback: reset results when any backtest input changes
# ---------------------------------------------------------------------------

@callback(
    Output('bt-chart', 'figure', allow_duplicate=True),
    Output('bt-chart', 'style', allow_duplicate=True),
    Output('bt-metrics', 'children', allow_duplicate=True),
    Output('bt-status', 'children', allow_duplicate=True),
    Output('bt-orders-store', 'data', allow_duplicate=True),
    Input('bt-basket-store-a', 'data'),
    Input('bt-basket-store-b', 'data'),
    Input('bt-base-currency', 'value'),
    Input('bt-strategy-config-store-a', 'data'),
    Input('bt-strategy-config-store-b', 'data'),
    Input('bt-weights-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def reset_results_on_input_change(basket_a, basket_b, base_currency, strategy_a, strategy_b,
                                  weights=None):
    """Clear chart, KPIs, and order tables when any backtest input changes.

    Whenever the user modifies the asset list, an asset's basket membership, the
    base currency, a strategy, or a per-asset weight before (or after) running a
    backtest, the previous results no longer match the current configuration and
    would be misleading. This callback resets all result areas to the initial
    page-load state: an empty visible chart and a metrics table filled with —
    placeholders (matching the state established by the layout on first render).

    Membership changes flow in through the derived bt-basket-store-a/b inputs, so
    ticking/unticking a basket checkbox also clears stale results.
    """
    placeholder_metrics = {k: '—' for k in (
        'Total Return', 'CAGR', 'Sharpe Ratio', 'Max. Drawdown',
        'Volatility (p.a.)', 'Calmar Ratio', 'Invested', 'End Value', 'Profit/Loss',
    )}
    return go.Figure(), {'width': '100%', 'display': 'block'}, _metrics_table(placeholder_metrics, None), '', {}


# ---------------------------------------------------------------------------
# Callback: render the active basket's order table into an always-visible div
# ---------------------------------------------------------------------------

@callback(
    Output('bt-orders-content', 'children'),
    Input('bt-orders-tabs', 'active_tab'),   # 'a' or 'b'
    Input('bt-orders-store', 'data'),        # {'a': html|None, 'b': html|None}
)
@log_time
def render_order_table(active_tab, orders_store):
    """Render the selected basket's order table into the always-visible content div.

    The order tables deliberately do NOT live inside the dbc.Tabs panes: a
    Bootstrap tab pane is display:none until activated, and Safari does not
    repaint content a callback injects into the active pane until a tab switch
    forces a reflow.  Instead the tabs only choose the active basket; this
    callback (re-)renders that basket's stored HTML into a single always-visible
    div whenever the tab or the stored data changes — which paints reliably,
    exactly like the chart and metrics.
    """
    store = orders_store or {}
    return _order_table_component(store.get(active_tab or 'a'))


# ---------------------------------------------------------------------------
# Callback: download the active basket's order table as CSV / Excel
# ---------------------------------------------------------------------------

@callback(
    Output('bt-orders-dl-data', 'data'),
    Input('bt-dl-csv', 'n_clicks'),    # 'CSV' menu item
    Input('bt-dl-xlsx', 'n_clicks'),   # 'Excel' menu item
    State('bt-orders-tabs', 'active_tab'),
    State('bt-orders-store', 'data'),
    prevent_initial_call=True,
)
@log_time
def download_orders(n_csv, n_xlsx, active_tab, orders_store):
    """Send the active basket's order table to the browser as CSV or Excel.

    The order rows already live (display-formatted) in bt-orders-store, so no
    recomputation is needed; the requested format is read from the triggering
    menu item.  An empty/absent basket downloads nothing.
    """
    tab = active_tab or 'a'
    rows = (orders_store or {}).get(tab)
    if not rows:
        return no_update

    # Columns in the same left-to-right order as the on-screen table: the fixed
    # _ORDER_COLUMNS followed by the dynamic per-asset value columns.  Derived
    # from the stored rows' keys so the export matches the rendered table exactly.
    labels = list(rows[0].keys())
    df = pd.DataFrame(rows, columns=labels)
    basket = 'A' if tab == 'a' else 'B'

    if dash.callback_context.triggered_id == 'bt-dl-xlsx':
        return dcc.send_data_frame(
            df.to_excel, f'orders_basket_{basket}.xlsx', index=False,
            sheet_name=f'Basket {basket}')
    return dcc.send_data_frame(df.to_csv, f'orders_basket_{basket}.csv', index=False)
