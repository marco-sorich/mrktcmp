---
name: verify
description: Launch and drive the mrktcmp Dash app end-to-end against generated local parquet data, using Playwright with the pre-installed Chromium.
---

# Verify mrktcmp changes in the running app

## Data source (no network needed)

`BASE_URL` accepts a local directory. Generate a minimal catalogue + assets:

```python
# master.parquet needs: asset_class, symbol, interval, name, exchange, country,
# category, first_date, last_date, filename (+ optional currency).
# Each asset file: OHLCV columns, tz-aware (UTC) DatetimeIndex named 'Date'.
# Span >= 3 calendar months or compute_metrics() rejects the window.
```

## Launch

```bash
BASE_URL=/path/to/data LOG_LEVEL=INFO python src/app.py   # serves on :8050
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8050/   # 200 when up
```

## Drive (Playwright, Python)

- `pip install playwright` only — Chromium is pre-installed; launch with
  `p.chromium.launch(executable_path='/opt/pw-browsers/chromium', headless=True)`.
- **Dash 4.x `dcc.Dropdown` is Radix-based**: the options render in a portal
  *outside* the component. Click the dropdown (`#bt-asset`), `keyboard.type()`
  the filter, then click `page.locator('[role="option"]', has_text=...)`.
  The old react-select selectors (`.Select-control`, `.VirtualizedSelectOption`)
  do not exist.
- Dict component ids appear in the DOM as sorted compact JSON; build selectors
  with `json.dumps(id, separators=(',', ':'), sort_keys=True)` wrapped in
  `[id='...']` (single quotes — double quotes break querySelector).
- **`dbc.Checkbox` puts the id on the `<input>` itself**, so the dict-id selector
  IS the checkbox — call `.is_checked()` / `.uncheck()` on it directly (no
  ` input` descendant).
- Wait for a run via `#bt-status` containing `complete`, then read
  `#bt-metrics`, `#bt-orders-content`, and the chart legend from page content.

## Flows worth driving

- One shared asset list: add assets (asset-class radio → `#bt-asset` search
  dropdown → ＋), edit per-row weights, check the live per-basket allocation %
  labels (`{'type':'bt-weight-pct-a'|'bt-weight-pct-b','index':fn}`).
- Per-asset A/B membership: each row has two `dbc.Checkbox`es
  (`{'type':'bt-member-a'|'bt-member-b','index':fn}`), both checked by default.
  Uncheck one → that asset leaves that basket (its % becomes `—`, the basket
  renormalises over its remaining members); the derived `bt-basket-store-a/b`
  and the date slider react.
- Run backtest and check the chart legend + metric column labels are
  `A: <strategy>` / `B: <strategy>`, and that each order tab lists exactly its
  basket's members.
