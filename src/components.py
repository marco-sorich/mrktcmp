# ---------------------------------------------------------------------------
# components.py – Reusable UI component builders
#
# This module contains functions that construct pieces of the Dash component
# tree that are needed in more than one place, or that are complex enough to
# deserve their own function. Keeping them here rather than inline in
# layout.py or in the callbacks makes each piece easier to read and test in
# isolation.
#
# All functions return Dash/HTML component objects. Dash components are plain
# Python objects; Dash serialises them to JSON and React renders them in the
# browser. Changing a component's properties in a callback causes only that
# component to re-render, not the entire page.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dash component imports
# ---------------------------------------------------------------------------

# html: wrappers for every standard HTML tag (html.Div, html.P, html.Span,
#       html.Button, html.H3, html.Table, html.Tr, html.Th, html.Td …).
# dcc:  "Dash Core Components" – interactive widgets like Dropdown,
#       RadioItems, and Store that go beyond plain HTML.
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

# config is imported as a module object so that assetsClasses is resolved at
# call time rather than at import time. This matters during testing, where
# config.assetsClasses may be replaced or the module may not yet be fully
# initialised when components.py is first imported.
import src.config as _config

# OrderRow is the finalized order-log row type produced by build_order_log;
# imported here only so _order_table can be type-annotated with it.
from src.backtest import OrderRow

# Shared style dicts imported from styles.py so all components stay visually
# consistent without duplicating the same dict literals.
from src.styles import _BASKET_ITEM_STYLE, _BTN_SMALL, _METRIC_TABLE_STYLE


# ---------------------------------------------------------------------------
# Helper: build the UI block for one asset basket
# ---------------------------------------------------------------------------

def _get_strategy_options() -> list[dict]:
    """Build dcc.Dropdown option dicts for every registered strategy.

    Each option shows a Bootstrap icon alongside the strategy name.
    Lazy-imported so component module load order does not matter.
    """
    from src.strategies.registry import get_all_strategy_info
    return [
        {
            'label': html.Span(
                [html.I(className=f"bi {info['icon']} me-2"), info['name']],
                style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'},
            ),
            'value': info['name'],
        }
        for info in get_all_strategy_info()
    ]


def _default_strategy_config() -> dict:
    """Return the config store initial value for the first registered strategy.

    Used to pre-populate bt-strategy-config-store-{x} at layout time so the
    main backtest callback always has a valid strategy + params dict to read.
    """
    from src.strategies.registry import list_strategies, get_strategy
    strategies = list_strategies()
    if not strategies:
        return {'strategy': None, 'params': {}}
    name = strategies[0]
    strategy_cls = get_strategy(name)
    params = {p.key: p.default for p in strategy_cls.get_config_schema()}
    return {'strategy': name, 'params': params}


def _build_strategy_params_ui(
    strategy_name: str | None,
    basket_id: str,
    disabled: bool = False,
) -> list:
    """Build input widgets for every ConfigParam of *strategy_name*.

    Called at layout time (initial render) and by the strategy-selector
    callback whenever the user switches strategies.

    Parameters
    ----------
    strategy_name – registered strategy name (e.g. 'DCA'), or None.
    basket_id     – 'a' or 'b'; used to build per-basket param input IDs.
    disabled      – when True all controls are rendered in the disabled state
                    (used when the basket is empty).

    Returns
    -------
    List of Dash components (one labelled input per ConfigParam), or [].
    """
    if not strategy_name:
        return []
    from src.strategies.registry import get_strategy
    try:
        strategy_cls = get_strategy(strategy_name)
    except KeyError:
        return []

    widgets = []
    for p in strategy_cls.get_config_schema():
        input_id = {'type': f'bt-param-{basket_id}', 'index': p.key}
        if p.type in ('int', 'float'):
            control = dbc.Input(
                id=input_id,
                type='number',
                value=p.default,
                min=p.min_value,
                max=p.max_value,
                step=1 if p.type == 'int' else 'any',
                debounce=True,
                size='sm',
                disabled=disabled,
            )
        else:  # 'select'
            control = dcc.Dropdown(
                id=input_id,
                options=[{'label': o, 'value': o} for o in p.options],
                value=str(p.default),
                clearable=False,
                disabled=disabled,
                style={'fontSize': '13px'},
            )
        widgets.append(
            html.Div([
                html.Label(
                    p.label,
                    style={'fontSize': '12px', 'color': '#555', 'marginBottom': '2px', 'display': 'block'},
                ),
                control,
            ], style={'marginBottom': '6px'})
        )
    return widgets


def _basket_ui(basket_id):
    """Return the complete HTML/component tree for a single basket panel.

    Each basket panel contains:
      • A heading ("Basket A" or "Basket B").
      • Asset-class radio buttons that filter the search dropdown.
      • A searchable asset dropdown + an add (＋) button side by side.
      • A list area that displays the assets currently in the basket.
      • An invisible dcc.Store that persists the basket's contents between
        callbacks (browser-side JSON storage, no server round-trip).

    Parameters
    ----------
    basket_id : str – either 'a' or 'b'. Used to build unique component IDs
                      (e.g. 'bt-assetclass-a', 'bt-add-b') so Dash can
                      distinguish the two baskets' components.

    Returns
    -------
    html.Div containing all controls for one basket.
    """
    # Map basket_id to a human-readable label for the heading.
    label = 'A' if basket_id == 'a' else 'B'

    # Outer panel style: flex:1 lets this panel grow equally with its sibling
    # basket panel in their shared flex row. minWidth:0 prevents a flex child
    # from overflowing when it contains long text (a common flexbox gotcha).
    return html.Div([

        # Section heading displayed above the basket controls.
        html.H3(f'Basket {label}', style={'marginBottom': '8px'}),

        # dcc.RadioItems renders a group of radio buttons (mutually exclusive
        # choices). Here it lists the asset classes loaded at startup so the
        # user can narrow the search dropdown to one class (e.g. 'stocks').
        # inline=True puts the radio buttons on a single horizontal line.
        dcc.RadioItems(
            _config.assetsClasses,            # list of option labels/values
            id=f'bt-assetclass-{basket_id}',  # unique component ID for callbacks
            inline=True,
            style={'marginBottom': '8px'},
        ),

        # Flex row: display:flex places the dropdown and add-button side by
        # side. gap:6px spaces them apart. alignItems:center aligns them
        # vertically. The dropdown has flex:1 so it grows to fill the row.
        html.Div([
            # dcc.Dropdown renders a searchable select input.
            # placeholder – the greyed-out hint text shown when empty.
            # disabled=True – greyed out until the user picks an asset class.
            # style={'flex': 1} – in a flex container, flex:1 means "grow to
            #                     fill all available horizontal space".
            dcc.Dropdown(
                id=f'bt-asset-{basket_id}',
                placeholder='Search asset…',
                disabled=True,
                style={'flex': 1},
            ),
            # The add button. n_clicks=0 initialises the click counter so Dash
            # has a starting value to compare against on the first click.
            # ** unpacks _BTN_SMALL and the extra keys override on top.
            html.Button(
                '＋',
                id=f'bt-add-{basket_id}',
                n_clicks=0,
                style={**_BTN_SMALL, 'fontSize': '16px', 'padding': '2px 12px'},
            ),
        ], style={'display': 'flex', 'gap': '6px', 'alignItems': 'center', 'marginBottom': '8px'}),

        # Container for the list of assets currently in the basket.
        # Updated by the manage_basket_x callback whenever the user adds or
        # removes assets. minHeight ensures the panel does not collapse to
        # zero height when the basket is empty.
        html.Div(id=f'bt-basket-list-{basket_id}', style={'minHeight': '32px'}),

        # --------------- Strategy selector ----------------------------------
        # A thin divider separates the asset list from the strategy section.
        html.Hr(style={'margin': '10px 0', 'borderColor': '#eee'}),

        html.Label(
            'Strategy',
            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px', 'display': 'block'},
        ),

        # Dropdown listing every registered strategy with its Bootstrap icon.
        # Dict ID enables a single MATCH callback to serve both baskets.
        # Starts disabled; the toggle_strategy_dropdown_x callbacks enable it
        # once at least one asset has been added to the basket.
        dcc.Dropdown(
            id={'type': 'bt-strategy', 'basket': basket_id},
            options=_get_strategy_options(),
            value=_default_strategy_config()['strategy'],
            clearable=False,
            disabled=True,
            style={'marginBottom': '8px'},
        ),

        # Container for strategy-specific parameter inputs.  Populated by the
        # render_strategy_params callback when the user picks a strategy.
        html.Div(
            id={'type': 'bt-strategy-params', 'basket': basket_id},
            children=_build_strategy_params_ui(
                _default_strategy_config()['strategy'], basket_id, disabled=True
            ),
        ),

        # Invisible store that holds the currently selected strategy name and
        # its resolved parameter values.  Read by the main backtest callback.
        dcc.Store(id=f'bt-strategy-config-store-{basket_id}', data=_default_strategy_config()),

        # dcc.Store is an invisible component that holds JSON data in the
        # browser's memory for the duration of the session. We use it to
        # persist the list of assets in each basket between callbacks.
        # data=[] initialises it with an empty list.
        dcc.Store(id=f'bt-basket-store-{basket_id}', data=[]),

    ], style={'flex': 1, 'minWidth': 0})


# ---------------------------------------------------------------------------
# Helper: render the visible list of assets currently in a basket
# ---------------------------------------------------------------------------

def _render_basket_list(basket_data, basket_id):
    """Build the component tree for the basket's item list.

    Each item shows the asset's symbol and name alongside a remove (✕) button.
    Called by _manage_basket every time an asset is added or removed.

    Parameters
    ----------
    basket_data : list of dicts, each with keys 'filename', 'symbol', 'name'.
    basket_id   : str – 'a' or 'b'. Used to build pattern-matching IDs for
                        the remove buttons so the manage_basket callback knows
                        which basket a removal belongs to.

    Returns
    -------
    html.P (empty placeholder) or html.Div (list of rows with remove buttons).
    """
    # If the basket is empty, show a placeholder message in light grey italic.
    if not basket_data:
        return html.P('No assets', style={'color': '#aaa', 'fontStyle': 'italic', 'margin': '4px 0'})

    # Build one row per asset using a Python list comprehension.
    # A list comprehension [ expr for item in iterable ] is a concise way to
    # build a list by applying expr to each element of an iterable.
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

    The table has three columns: Metric | Basket A | Basket B. Basket A values
    appear in blue, Basket B values in red. If one basket has no results (e.g.
    it was left empty), its column shows '—' for every metric.

    Parameters
    ----------
    metrics_a : dict or None – metric_name → formatted string for basket A.
                               Example: {'Total Return': '+25.3%', 'CAGR': '8.1%'}
    metrics_b : dict or None – same structure for basket B.

    Returns
    -------
    html.P (if both are None/empty) or html.Table with header + data rows.
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
        # the basket dict is None or the key is absent (basket not run yet).
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
# Helper: render the per-order transaction table
# ---------------------------------------------------------------------------

# Column specification for the order table: (OrderRow key, header label,
# value formatter).  The order here is the left-to-right column order in the
# rendered table.  Each formatter receives the already-non-None value (None is
# rendered as an em-dash by _order_table before the formatter runs):
#   • currency columns use thousands separators and no decimals (e.g. 12,500);
#     profit/loss adds an explicit sign so gains/losses read at a glance;
#   • percentage columns scale by 100 and show one decimal (e.g. 66.7%);
#     P&L and period return add an explicit sign.
# Mirrors the formatting conventions of compute_metrics() in backtest.py.
_ORDER_COLUMNS = [
    ('date', 'Date', lambda v: v.strftime('%Y-%m-%d')),
    ('side', 'Buy/Sell', str),
    ('value_before', 'Value before', lambda v: f'{v:,.0f}'),
    ('inflow', 'Inflow', lambda v: f'{v:,.0f}'),
    ('assets_after', 'Assets value', lambda v: f'{v:,.0f}'),
    ('cash_after', 'Cash value', lambda v: f'{v:,.0f}'),
    ('value_after', 'Value after', lambda v: f'{v:,.0f}'),
    ('net_deposits', 'Net deposits', lambda v: f'{v:,.0f}'),
    ('pnl_abs', 'P&L (€)', lambda v: f'{v:+,.0f}'),
    ('pnl_pct', 'P&L (%)', lambda v: f'{v * 100:+.1f}%'),
    ('equity_exposure', 'Equity exposure', lambda v: f'{v * 100:.1f}%'),
    ('cash_quote', 'Cash quota', lambda v: f'{v * 100:.1f}%'),
    ('period_return', 'Period return', lambda v: f'{v * 100:+.1f}%'),
]


def _order_table(orders: list[OrderRow] | None) -> "dash_table.DataTable | html.P":
    """Build the per-order transaction table for one basket.

    Rendered as a single *virtualized* ``dash_table.DataTable`` rather than a
    grid of html.Tr/html.Td components.  A long order log (hundreds of monthly
    contributions over a multi-year window) otherwise costs *seconds* of
    client-side rendering: dash-renderer instantiates every one of the
    rows × columns cells as its own React component.  The DataTable is a single
    component that ships its rows as a compact data prop and only renders the
    rows currently in view, while keeping the header and the first (Date) column
    fixed and fitting the 80vh results area.

    Each cell is pre-formatted to a string with the _ORDER_COLUMNS formatters
    (None → em-dash) so the displayed text is identical to the previous table.

    Parameters
    ----------
    orders – list of OrderRow dicts (from build_order_log), or None/empty when
             the basket produced no result.

    Returns
    -------
    html.P placeholder when there are no orders, otherwise a dash_table.DataTable.
    """
    # Empty / missing → a plain placeholder, mirroring _metrics_table.
    if not orders:
        return html.P('No orders.', style={'color': '#aaa'})

    # One dict per order: every cell pre-formatted to a string (None → em-dash)
    # so the DataTable simply displays text identical to the old html.Table.
    data = [
        {key: ('—' if row[key] is None else fmt(row[key]))
         for key, _label, fmt in _ORDER_COLUMNS}
        for row in orders
    ]
    columns = [{'name': label, 'id': key} for key, label, _fmt in _ORDER_COLUMNS]

    return dash_table.DataTable(
        data=data,
        columns=columns,
        # virtualization renders only the rows near the viewport; fixed_rows /
        # fixed_columns keep the header and the Date column pinned while scrolling.
        virtualization=True,
        fixed_rows={'headers': True},
        fixed_columns={'headers': True, 'data': 1},
        page_action='none',
        style_table={'height': '80vh', 'overflowY': 'auto',
                     'overflowX': 'auto', 'minWidth': '100%'},
        # Deterministic per-column widths: fixed_columns mis-sizes (and so the
        # frozen Date column would overlap the next one) unless cell widths are
        # explicit.  120px fits every header without wrapping.
        style_cell={'fontSize': '13px', 'padding': '4px 10px', 'textAlign': 'right',
                    'whiteSpace': 'nowrap', 'fontFamily': 'inherit',
                    'border': 'none', 'borderBottom': '1px solid #eee',
                    'minWidth': '130px', 'width': '130px', 'maxWidth': '130px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0', 'border': 'none'},
        # Text columns (Date, Buy/Sell) read better left-aligned; numbers stay right.
        style_cell_conditional=[
            {'if': {'column_id': 'date'}, 'textAlign': 'left'},
            {'if': {'column_id': 'side'}, 'textAlign': 'left'},
        ],
    )
