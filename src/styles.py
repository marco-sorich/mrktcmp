# ---------------------------------------------------------------------------
# styles.py – Shared CSS-in-Python style dictionaries
#
# In Dash, component styles are plain Python dicts using camelCase CSS
# property names (e.g. CSS 'background-color' → Python 'backgroundColor').
# Defining them once here as module-level constants avoids repetition across
# components.py and layout.py, and makes future visual changes a one-line
# edit in a single file.
# ---------------------------------------------------------------------------

# _BASKET_ITEM_STYLE styles each row in the asset basket list.
#   display: flex         – activates Flexbox layout so children sit side by
#                           side on a horizontal line.
#   alignItems: center    – vertically centres children inside the flex row.
#   justifyContent:       – pushes the label to the left edge and the remove
#     space-between         button to the right edge.
#   padding / marginBottom – inner spacing and gap between rows.
#   background / borderRadius / fontSize – cosmetic look.
_BASKET_ITEM_STYLE = {
    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
    'padding': '4px 8px', 'marginBottom': '2px', 'background': '#f5f5f5',
    'borderRadius': '4px', 'fontSize': '13px',
}

# _BTN_SMALL applies to the compact "＋" (add) and "✕" (remove) buttons.
#   padding / fontSize – make the button compact.
#   cursor: pointer     – shows the hand cursor on hover (UX convention for
#                         clickable elements that are not standard <a> links).
#   border / borderRadius / background – minimal border with rounded corners.
_BTN_SMALL = {
    'padding': '2px 8px', 'fontSize': '12px', 'cursor': 'pointer',
    'border': '1px solid #ccc', 'borderRadius': '3px', 'background': 'white',
}

# _METRIC_TABLE_STYLE styles the performance metrics comparison table.
#   borderCollapse: collapse – removes the double border between adjacent
#                             cells (normally HTML tables have a gap).
#   width: 100%             – stretch the table to fill its container.
#   fontSize: 13px          – slightly smaller than body text for compactness.
_METRIC_TABLE_STYLE = {'borderCollapse': 'collapse', 'width': '100%', 'fontSize': '13px'}

# Approximate pixel height of one transaction-table row, used to size the
# scroll viewport to ~30 rows (see _TX_SCROLL_STYLE).  A small constant rather
# than a magic number sprinkled across the code.
_TX_ROW_PX = 28

# _TX_SCROLL_STYLE wraps a transaction table in a fixed-height, vertically
# scrollable box.  maxHeight is sized to ~30 data rows plus the sticky header
# so the table never grows taller than the requested 30-line viewport.
_TX_SCROLL_STYLE = {
    'maxHeight': f'{_TX_ROW_PX * 31}px',  # ~30 rows + header
    'overflowY': 'auto',
    # Let the wide ledger scroll sideways on narrow screens; the Date column is
    # frozen (sticky) via the .tx-col-date CSS class so rows stay identifiable.
    'overflowX': 'auto',
    'border': '1px solid #eee',
    'borderRadius': '4px',
}

# _TX_BTN_STYLE styles the "scroll to top / bottom" buttons above each table.
# Mirrors _BTN_SMALL but with a touch more horizontal padding for the labels.
_TX_BTN_STYLE = {
    'padding': '2px 10px', 'fontSize': '12px', 'cursor': 'pointer',
    'border': '1px solid #ccc', 'borderRadius': '3px', 'background': 'white',
    'marginRight': '6px',
}
