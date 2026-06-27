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
#   flexWrap: wrap        – on a container too narrow to hold the asset label and
#                           the weight/remove controls side by side (e.g. a phone),
#                           the controls wrap onto a second line instead of
#                           squeezing the asset text down to a few pixels.
#   gap                   – spacing that also applies between the wrapped rows.
_BASKET_ITEM_STYLE = {
    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
    'flexWrap': 'wrap', 'gap': '6px',
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
