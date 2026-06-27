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
from dash import html, dcc
import dash_bootstrap_components as dbc

# escape: HTML-escape cell text (e.g. the '&' in the 'P&L (€)' header) before it
# goes into the raw-HTML order table rendered via dcc.Markdown.  Aliased so it
# does not shadow Dash's `html` component module imported above.
from html import escape as _html_escape

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


# Currency codes that must never be offered as a *reporting* currency: blank /
# placeholder values, and GBp (pence) which is a quote sub-unit, not a currency
# anyone reports a portfolio in.
_NON_BASE_CURRENCIES = {'', '0', 'nan', 'None', 'GBp'}

# Currency codes that carry no usable information (blank / placeholder).  Unlike
# _NON_BASE_CURRENCIES this keeps GBp, which *is* a valid trading currency to
# label an asset's price column with (it just cannot be a reporting currency).
_BLANK_CURRENCIES = {'', '0', 'nan', 'None'}


def _base_currency_options() -> list[str]:
    """List the currencies offerable as the portfolio's reporting (base) currency.

    Derived from the catalogue's ``currency`` asset-class rows (the FX pairs):
    their quote currencies are exactly the set we can convert *into*.  Blank /
    placeholder codes and GBp (pence) are excluded; the configured default is
    always included so the dropdown never starts on an absent value.  Falls back
    to ``[default]`` when no catalogue is loaded.

    Returns
    -------
    Sorted list of currency-code strings.
    """
    default = _config.default_base_currency
    codes = {default}
    df = _config.df
    if df is not None and 'currency' in df.columns and 'asset_class' in df.columns:
        fx = df[df['asset_class'] == 'currency']
        for code in fx['currency'].dropna().astype(str).str.strip().unique():
            if code not in _NON_BASE_CURRENCIES:
                codes.add(code)
    return sorted(codes)


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


def _strategy_desc_panel(strategy_name: str | None) -> list:
    """Build the rich-text (Markdown) description panel for *strategy_name*.

    Renders the strategy's get_long_description() — a multi-paragraph Markdown
    write-up — inside a dcc.Markdown.  Returns [] when no/unknown strategy is
    given so the collapsible info panel stays empty.

    Called at layout time (initial render) and by the strategy-selector callback
    whenever the user switches strategies.

    Parameters
    ----------
    strategy_name – registered strategy name (e.g. 'DCA'), or None.

    Returns
    -------
    Single-element list with a dcc.Markdown, or [].
    """
    if not strategy_name:
        return []
    from src.strategies.registry import get_strategy
    try:
        strategy_cls = get_strategy(strategy_name)
    except KeyError:
        return []
    return [dcc.Markdown(strategy_cls.get_long_description())]


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

        # Label row: the "Strategy" caption plus a small info (ⓘ) toggle button
        # that expands/collapses the rich-text description panel below.
        html.Div([
            html.Label(
                'Strategy',
                style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '0', 'display': 'block'},
            ),
            html.Button(
                html.I(className='bi bi-info-circle'),
                id={'type': 'bt-strategy-desc-toggle', 'basket': basket_id},
                n_clicks=0,
                title='Show strategy description',
                className='strategy-desc-toggle',
                style={
                    'border': 'none', 'background': 'none', 'cursor': 'pointer',
                    'padding': '0', 'color': '#0d6efd', 'fontSize': '14px', 'lineHeight': '1',
                },
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '4px'}),

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

        # Collapsible rich-text description of the selected strategy.  Hidden by
        # default; toggled open by the ⓘ button above and re-filled on strategy
        # change by the render_strategy_params callback.
        dbc.Collapse(
            html.Div(
                id={'type': 'bt-strategy-desc', 'basket': basket_id},
                children=_strategy_desc_panel(_default_strategy_config()['strategy']),
                className='strategy-desc',
            ),
            id={'type': 'bt-strategy-desc-collapse', 'basket': basket_id},
            is_open=False,
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

        # Persists the per-asset relative weights as {filename: weight}.  Kept
        # separate from the basket store so editing a weight does NOT retrigger the
        # date-range slider (which resets the selected window).  Written by the
        # sync_weights callback; read by _manage_basket (to keep weights through
        # add/remove) and by the run callback (to weight the simulation).
        dcc.Store(id=f'bt-weights-store-{basket_id}', data={}),

    ], style={'flex': 1, 'minWidth': 0})


# ---------------------------------------------------------------------------
# Helper: render the visible list of assets currently in a basket
# ---------------------------------------------------------------------------

def _weight_percentages(weight_by_key: dict) -> dict:
    """Map each key to its share-of-total weight as a percentage string ('40%').

    The per-asset weights the user types are *relative* numbers; what matters for
    the allocation is each one's share of the basket's total, so the UI shows that
    normalised percentage next to every weight input.  Negative inputs are clamped
    to 0; when the weights sum to zero (e.g. every asset zeroed) the share is
    undefined and rendered as an em-dash for every key.

    Parameters
    ----------
    weight_by_key – mapping of any key (here: a basket asset's filename) → its raw
                    relative weight.

    Returns
    -------
    dict with the same keys, each mapped to a formatted percentage string.
    """
    total = sum(max(float(w), 0.0) for w in weight_by_key.values())
    if total <= 0.0:
        return {k: '—' for k in weight_by_key}
    return {k: f'{max(float(w), 0.0) / total * 100:.0f}%' for k, w in weight_by_key.items()}


def _basket_item_label(item: dict) -> str:
    """Build one basket row's text: "<symbol> — <name>" plus a currency tag.

    The trading currency (stored on the item when the asset was added) is appended
    in parentheses — e.g. "AAPL — Apple Inc. (USD)" — so a user scanning a
    mixed-currency basket can see each asset's currency at a glance.  Blank /
    unknown currencies (e.g. Indices) get no tag.
    """
    label = f"{item['symbol']} — {item['name']}"
    ccy = str(item.get('currency') or '').strip()
    if ccy and ccy not in _BLANK_CURRENCIES:
        label += f" ({ccy})"
    return label


def _render_basket_list(basket_data, basket_id, weights=None):
    """Build the component tree for the basket's item list.

    Each item shows the asset's symbol and name, an editable **weight** input with
    the resulting allocation percentage beside it, and a remove (✕) button.
    Called by _manage_basket every time an asset is added or removed.

    The weights are *relative* numbers (default 1.0, so an untouched basket is
    equal-weighted and adding an asset automatically re-weights all of them); the
    percentage shown is each weight's share of the basket total (see
    ``_weight_percentages``).  Editing a weight is handled by the per-basket
    sync_weights callback, which persists the value and refreshes the percentage
    in place without re-rendering this list (so the input keeps focus).

    Parameters
    ----------
    basket_data : list of dicts, each with keys 'filename', 'symbol', 'name'.
    basket_id   : str – 'a' or 'b'. Used to build pattern-matching IDs for the
                        per-row weight inputs and remove buttons so the callbacks
                        know which basket a control belongs to.
    weights     : dict or None – filename → relative weight, used to pre-fill each
                        row's weight input and percentage (missing → default 1.0).

    Returns
    -------
    html.P (empty placeholder) or html.Div (list of rows with weight inputs and
    remove buttons).
    """
    # If the basket is empty, show a placeholder message in light grey italic.
    if not basket_data:
        return html.P('No assets', style={'color': '#aaa', 'fontStyle': 'italic', 'margin': '4px 0'})

    # Resolve each row's weight (default 1.0) and its share-of-total percentage so
    # the inputs and labels start out consistent with the stored weights.
    weights = weights or {}
    weight_by_file = {
        item['filename']: max(float(weights.get(item['filename'], 1.0)), 0.0)
        for item in basket_data
    }
    pct_by_file = _weight_percentages(weight_by_file)

    # Build one row per asset using a Python list comprehension.
    # A list comprehension [ expr for item in iterable ] is a concise way to
    # build a list by applying expr to each element of an iterable.
    return html.Div([
        html.Div([
            # Asset label: "AAPL — Apple Inc (USD)" — the trailing currency tag
            # (omitted for assets with a blank/unknown currency) tells the user
            # which currency the asset trades in, since baskets may mix currencies.
            # flex '1 1 140px' lets it grow but claim at least ~140px before the
            # controls wrap below it; wordBreak wraps long names instead of
            # clipping them, so the asset text stays readable on a narrow phone.
            html.Span(_basket_item_label(item),
                      style={'flex': '1 1 140px', 'minWidth': '120px',
                             'wordBreak': 'break-word'}),

            # Right-hand controls: the relative-weight input, its resulting
            # allocation percentage, and the remove button, grouped together.
            # flexShrink 0 keeps them at their natural size so they never crush the
            # label; when the row is too narrow they wrap onto their own line.
            html.Div([
                # Editable relative weight (pattern-matching ID carries the
                # filename).  debounce=True commits on Enter/blur so the sync
                # callback does not fire on every keystroke.
                dbc.Input(
                    id={'type': f'bt-weight-{basket_id}', 'index': item['filename']},
                    type='number',
                    value=weight_by_file[item['filename']],
                    min=0,
                    step='any',
                    debounce=True,
                    size='sm',
                    style={'width': '58px', 'textAlign': 'right'},
                ),
                # Live allocation percentage (= this weight's share of the total),
                # updated in place by the sync_weights callback.
                html.Span(
                    pct_by_file[item['filename']],
                    id={'type': f'bt-weight-pct-{basket_id}', 'index': item['filename']},
                    style={'fontSize': '12px', 'color': '#666',
                           'minWidth': '38px', 'textAlign': 'right'},
                ),

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
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px',
                      'flexShrink': 0, 'marginLeft': 'auto'}),
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
    ('inflow', 'Inflow', lambda v: f'{v:,.0f}'),
    ('assets_after', 'Assets value', lambda v: f'{v:,.0f}'),
    ('cash_after', 'Cash value', lambda v: f'{v:,.0f}'),
    ('value_after', 'Portfolio value', lambda v: f'{v:,.0f}'),
    ('bh_value', 'B&H value', lambda v: f'{v:,.0f}'),
    ('net_deposits', 'Net deposits', lambda v: f'{v:,.0f}'),
    # The base-currency code is appended to this header at render time by
    # _order_rows (e.g. 'P&L (USD)'), so the table reflects the selected
    # reporting currency instead of a hard-coded one.
    ('pnl_abs', 'P&L', lambda v: f'{v:+,.0f}'),
    ('pnl_pct', 'P&L (%)', lambda v: f'{v * 100:+.1f}%'),
    ('equity_exposure', 'Equity exposure', lambda v: f'{v * 100:.1f}%'),
    ('cash_quote', 'Cash quota', lambda v: f'{v * 100:.1f}%'),
    ('period_return', 'Period return', lambda v: f'{v * 100:+.1f}%'),
]


def _order_asset_columns(orders: list[OrderRow]) -> list[str]:
    """Collect the basket's asset symbols (each drives a value + price column).

    Walks every order's ``asset_values`` dict and returns the symbols in
    first-seen order, so the per-asset columns appear in the basket's natural
    asset order and the set is stable across rows.  Empty when no order carries
    a per-asset breakdown (e.g. legacy events / test stubs without it).
    """
    cols: list[str] = []
    for row in orders:
        for symbol in (row.get('asset_values') or {}):
            if symbol not in cols:
                cols.append(symbol)
    return cols


def _order_fx_columns(orders: list[OrderRow]) -> list[str]:
    """Collect the ``{LOCAL}{BASE}=X`` FX pairs used to convert the basket.

    Walks every order's ``fx_rates`` dict and returns the pair symbols in
    first-seen order, so the order table can add one rate column per currency
    pair actually used.  Empty when no conversion happened (single-currency
    basket, base-currency assets, or legacy/test rows without FX context).
    """
    cols: list[str] = []
    for row in orders:
        for pair in (row.get('fx_rates') or {}):
            if pair not in cols:
                cols.append(pair)
    return cols


def _asset_currency_map(filenames: list[str], df_meta) -> dict[str, str]:
    """Map each basket asset's symbol to its catalogue trading currency.

    Used by the run callback to tell ``_order_rows`` which currency each asset
    trades in, so the order table can label its trading-currency price column.
    Returns ``''`` for assets with a blank/unknown currency, and an empty map
    when the catalogue carries no ``currency`` column (legacy/test fixtures).
    """
    out: dict[str, str] = {}
    if df_meta is None or 'currency' not in getattr(df_meta, 'columns', []):
        return out
    for filename in filenames:
        meta = df_meta[df_meta['filename'] == filename]
        if not meta.empty:
            cur = meta.iloc[0]['currency']
            out[str(meta.iloc[0]['symbol'])] = '' if cur is None else str(cur).strip()
    return out


def _order_rows(
    orders: list[OrderRow] | None,
    base_currency: str = 'EUR',
    asset_currency: "dict[str, str] | None" = None,
) -> "list[dict[str, str]] | None":
    """Format an order log into display rows: a list of {column label: text}.

    Each cell uses its _ORDER_COLUMNS formatter (None → em-dash), followed by the
    dynamic per-asset columns and, last, one column per FX pair used to convert
    the basket.  For each basket asset there is a '<symbol> value' column (its
    worth in the reporting currency, from ``asset_values``) and its price
    column(s): when the asset trades in a currency *other* than the reporting one
    (looked up in *asset_currency*) both the trading-currency quote
    '<symbol> price (<local>)' (from ``asset_prices_local``) and the converted
    '<symbol> price (<base>)' (from ``asset_prices``) are shown; otherwise a single
    price column is shown ('<symbol> price (<base>)' when the currency is known and
    equals the base, or a plain '<symbol> price' when it is unknown).  Finally a
    '<{LOCAL}{BASE}=X>' column per affected pair shows the rate each trade was
    converted at (from ``fx_rates``).  The P&L header gets *base_currency* appended
    (e.g. 'P&L (USD)') so the table names the selected reporting currency.  The
    result is plain JSON (all strings, keyed by the human column label) so it can
    live in a dcc.Store and feed **both** the rendered table
    (_order_table_component) and the CSV / Excel download (download_orders) from
    one source.  Returns None when there are no orders.
    """
    if not orders:
        return None
    asset_currency = asset_currency or {}
    # The per-asset columns are dynamic, appended after the fixed _ORDER_COLUMNS in
    # the basket's asset order (value then price), and the FX-pair rate columns
    # come last so the conversion inputs sit together at the right of the table.
    asset_cols = _order_asset_columns(orders)
    fx_cols = _order_fx_columns(orders)
    rows: list[dict[str, str]] = []
    for row in orders:
        # The absolute P&L column names the reporting currency dynamically; all
        # other fixed columns keep their static label.
        cells = {(f'P&L ({base_currency})' if key == 'pnl_abs' else label):
                 ('—' if row[key] is None else fmt(row[key]))
                 for key, label, fmt in _ORDER_COLUMNS}
        values = row.get('asset_values') or {}
        base_prices = row.get('asset_prices') or {}
        local_prices = row.get('asset_prices_local') or {}
        fx_rates = row.get('fx_rates') or {}
        for symbol in asset_cols:
            v = values.get(symbol)
            cells[f'{symbol} value'] = '—' if v is None else f'{v:,.0f}'
            base_p = base_prices.get(symbol)
            base_cell = '—' if base_p is None else f'{base_p:,.2f}'
            local = str(asset_currency.get(symbol, '')).strip()
            if local and local not in _BLANK_CURRENCIES and local != base_currency:
                # Asset trades in another currency → show both quotes side by side.
                local_p = local_prices.get(symbol)
                cells[f'{symbol} price ({local})'] = '—' if local_p is None else f'{local_p:,.2f}'
                cells[f'{symbol} price ({base_currency})'] = base_cell
            elif local and local not in _BLANK_CURRENCIES:
                # Currency known and already the base currency → one labelled column.
                cells[f'{symbol} price ({base_currency})'] = base_cell
            else:
                # Unknown/blank currency (e.g. an Index) → an unlabelled price column.
                cells[f'{symbol} price'] = base_cell
        for pair in fx_cols:
            r = fx_rates.get(pair)
            cells[pair] = '—' if r is None else f'{r:,.4f}'
        rows.append(cells)
    return rows


def _order_table_component(rows: "list[dict[str, str]] | None") -> "dcc.Markdown | html.P":
    """Render formatted order rows (from _order_rows, typically via a dcc.Store)
    as a native HTML table inside a single ``dcc.Markdown``.

    Emitting native HTML (one component) avoids the *seconds* of client-side
    rendering that a grid of html.Tr/html.Td costs — dash-renderer would
    instantiate every one of the rows × columns cells as its own React
    component — and, being plain HTML, it needs no JS layout measurement, so it
    paints first-time even inside the tabs UI.  dangerously_allow_html renders
    the raw <table>; the content is machine-generated and HTML-escaped, so it is
    safe.  None/empty rows yield the 'No orders.' placeholder.
    """
    if not rows:
        return html.P('No orders.', style={'color': '#aaa'})

    # Derive the column order from the row dicts (insertion order = the fixed
    # _ORDER_COLUMNS followed by the dynamic per-asset columns) so the table
    # automatically includes each basket's per-asset value columns.
    labels = list(rows[0].keys())
    head = ''.join(f'<th>{_html_escape(label)}</th>' for label in labels)
    body = ''.join(
        '<tr>' + ''.join(
            f'<td>{_html_escape(str(row.get(label, "")))}</td>' for label in labels
        ) + '</tr>'
        for row in rows
    )
    markup = (
        '<div class="order-table-wrapper">'
        f'<table class="order-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )
    return dcc.Markdown(markup, dangerously_allow_html=True)
