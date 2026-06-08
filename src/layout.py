# ---------------------------------------------------------------------------
# layout.py – Application layout factory
#
# Dash's app.layout defines the *entire* page structure as a tree of
# component objects. Dash serialises this tree to JSON, which React renders
# as HTML in the browser. The layout is set once at startup; callbacks then
# dynamically update individual component properties (such as a figure, a
# style dict, or an options list) without re-rendering the whole page.
#
# Splitting the layout into its own module keeps app.py thin and makes the
# page structure easy to find and modify without having to scroll past
# callback definitions.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dash component imports
# ---------------------------------------------------------------------------

# html: wrappers for every standard HTML element (html.Div, html.H1, etc.).
# dcc:  "Dash Core Components" – interactive widgets such as Dropdown,
#       Graph, RangeSlider, RadioItems, and Store.
from dash import html, dcc

# dbc: Dash Bootstrap Components – used here for the Tabs that switch between
# the two baskets' order tables.
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

# _basket_ui builds the full component subtree for a single basket panel
# (heading, radio buttons, dropdown, add button, item list, and Store).
from src.components import _basket_ui


# ---------------------------------------------------------------------------
# Layout factory
# ---------------------------------------------------------------------------

def create_layout():
    """Build and return the complete Dash component tree for the application.

    Called once during application startup in app.py:
        app.layout = create_layout()

    The page lets users build two asset baskets and compare how a chosen
    backtesting strategy (e.g. DCA, Risk-Off Signale) would have performed
    over a user-selected date range.

    Returns
    -------
    html.Div – the root component of the page.
    """
    # Outer container: maxWidth=100% fills the viewport. padding=0 8px
    # adds small side margins. boxSizing=border-box includes padding in the
    # declared width so content does not overflow horizontally.
    return html.Div([

        # Page heading – main brand name with a subtitle line below.
        html.Header([
            html.H1([
                html.I(
                    className="bi bi-graph-up-arrow",
                    style={'marginRight': '10px', 'fontSize': '2rem', 'verticalAlign': 'middle'}
                    ),
                "mrktcmp",
            ], style={
                'margin': '0',
                'fontSize': '2.4rem',
                'fontWeight': '700',
                'letterSpacing': '0.08em',
                'lineHeight': '1',
            }),
            html.P("compare markets", style={
                'margin': '4px 0 0 2px',
                'fontSize': '0.85rem',
                'fontWeight': '400',
                'letterSpacing': '0.18em',
                'color': '#888',
                'textTransform': 'uppercase',
            }),
        ], style={'marginBottom': '16px', 'borderBottom': '1px solid #e0e0e0', 'paddingBottom': '12px'}),

        # Two basket panels side by side using a flex row.
        # gap: 8px adds horizontal space between the panels.
        # marginTop: 12px adds vertical breathing room below the header.
        html.Div([
            _basket_ui('a'),                       # Basket A controls
            html.Div(style={'width': '24px'}),     # visual divider spacer
            _basket_ui('b'),                       # Basket B controls
        ], style={'display': 'flex', 'gap': '8px', 'marginTop': '12px'}),

        # Date-range section: slider + human-readable label + store.
        html.Div([
            html.Label('Analysis period:', style={'fontWeight': 'bold', 'marginBottom': '4px'}),

            # dcc.RangeSlider has two draggable handles so the user
            # can select both a start and an end date.
            # min / max / value are integers (indices into the date
            # list stored in bt-date-store). They are set dynamically
            # by update_date_range_slider when assets are added.
            # allowCross=False prevents the left handle from passing
            # the right handle (keeps the range always valid).
            # disabled=True at startup because no baskets are filled.
            html.Div(
                dcc.RangeSlider(
                    id='bt-date-range',
                    min=0, max=1, step=1, value=[0, 1],
                    marks={},
                    allowCross=False,
                    updatemode='drag',
                    disabled=True,
                    allow_direct_input=False,
                ),
                style={'width': '98%', 'margin': '0 auto'},
            ),

            # Human-readable date range display, e.g. "Jan 2020 – Dec 2024".
            # Updated by update_date_range_slider (on basket change)
            # and update_date_display (on every slider drag).
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

        # Run button – clicking this fires run_backtest_callback.
        html.Button(
            '▶ Start Backtest',
            id='bt-run',
            n_clicks=0,
            style={'padding': '8px 20px', 'fontSize': '14px', 'cursor': 'pointer', 'marginBottom': '16px'},
        ),

        # Status line – shows messages like "Backtest complete – 5.0 years".
        # Updated by run_backtest_callback; starts empty.
        html.Div(id='bt-status', style={'color': '#888', 'fontSize': '13px', 'marginBottom': '8px'}),

        # Chart and metrics side by side on wide screens (via layout.css),
        # stacked on narrow screens. CSS classes are defined in
        # src/assets/layout.css which Dash loads automatically.
        html.Div([
            html.Div(
                dcc.Graph(id='bt-chart', style={'width': '100%', 'display': 'none'}),
                className='bt-chart-wrapper',
            ),
            html.Div(id='bt-metrics', className='bt-metrics-wrapper'),
        ], className='results-container'),

        # Order (transaction) tables – one per basket, switchable via tabs and
        # placed below the chart + metrics. A distinct id ('bt-orders-tabs', not
        # 'main-tabs') avoids inheriting the orphaned #main-tabs icon rules in
        # layout.css. dbc.Tabs keeps both panes in the DOM, so the run callback
        # can populate both basket ids in one go. Each pane starts with an empty
        # placeholder rendered by _order_table(None).
        html.Div([
            html.H3('Orders', style={'marginBottom': '8px'}),
            dbc.Tabs(
                id='bt-orders-tabs',
                children=[
                    dbc.Tab(html.Div(id='bt-orders-a'), label='Basket A'),
                    dbc.Tab(html.Div(id='bt-orders-b'), label='Basket B'),
                ],
            ),
        ], style={'marginTop': '20px'}),

        # Hidden result store (reserved for future drill-down
        # interactions, e.g. clicking a month to inspect it).
        dcc.Store(id='bt-result-store', data={}),

    ], style={'maxWidth': '100%', 'padding': '0 8px', 'boxSizing': 'border-box'})
