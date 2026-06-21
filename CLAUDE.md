# CLAUDE.md

This is a Plotly/Dash based webpage for comparing baskets, containing financial asset, via backtesting.

## Commands

```bash
# Setup: install editable package with dev tools (from pyproject.toml)
pip install -e ".[dev]"

# Development server (hot-reload via DASH_DEBUG=true in .env)
python src/app.py
# or
gunicorn --reload src.app:server

# Production
gunicorn src.app:server

# Linting
flake8 --max-complexity=10 --max-line-length=127 ./src

# Type check (requires mypy 2.1.0 — pinned in pyproject.toml; older 1.x
# releases silently miss errors that CI's mypy 2.1.0 reports)
mypy --explicit-package-bases --ignore-missing-imports src/backtest.py src/strategies/
mypy --explicit-package-bases --ignore-missing-imports tests/test_backtest.py tests/test_strategies.py

# Tests
pytest

# Single test file
pytest tests/test_backtest.py
```

## Environment

```bash
# URL or local path to parquet files
BASE_URL=https://your-data-host/path

# set log level: CRITICAL, ERROR, WARNING, INFO, DEBUG
LOG_LEVEL=INFO

# (de-)activate dev tools of Plotly/Dash
DASH_DEBUG=false
```

`BASE_URL` must point to a directory containing `master.parquet` (asset catalogue) and one parquet file per asset (OHLCV data).

## Dependency Management

All dependencies (runtime and development) are pinned to exact versions in `pyproject.toml` for reproducibility. The build system (`setuptools` + `setuptools-scm`) manages the package metadata, versioning from git tags, and entry points. Installation uses `pip install -e ".[dev]"` (editable install with dev tools); no separate `requirements*.txt` files exist.

**Dependency Updates**: Dependabot (GitHub Action, configured in `.github/dependabot.yml`) scans `pyproject.toml` weekly and creates pull requests with new versions. CI runs on these PRs; tests validate compatibility. Review and merge PRs to keep dependencies current.

## Development

* Implement type checking for all functions which are not Dash callbacks
* Add comprhensive code commenting
* Keep the tests updated
* Keep the comments updated
* Keep `CLAUDE.md` updated
* Keep `README.md` updated
* Perform linting after each change
* Perform type checking after each change
* Perform tests after each change

## Architecture

**Entry point and module loading order:**
`app.py` → imports `src.callbacks` (side-effect: registers all Dash callbacks) → imports `src.layout` → `src.config` is always imported first as a module-level singleton.

**`src/config.py` — startup singleton**
Loads `.env`, configures logging, fetches `master.parquet` from `BASE_URL` into the module-level global `df` (a DataFrame). Resolves `app_version` from setuptools-scm (git tags, dev versions, or fallback); the version displays in the page footer. Exposes `base_url`, `assetsClasses`, `df`, `log`, and `app_version` to the rest of the app. All other modules import this as `import src.config as _config`. Version resolution chain: (1) installed distribution metadata, (2) generated `src/_version.py` from setuptools-scm, (3) live git query, (4) fallback to `'unknown'`. For releases, tag with `git tag v<VERSION>` (e.g., `git tag v0.1.0`); setuptools-scm will then report that exact version.

**`src/backtest.py` — pure simulation engines (no Dash dependencies)**
Three strategies share this module; all parquet I/O happens here. All engines run on **daily** close prices (`load_daily_closes()`), windowed to the selected months by `_window_by_month()`, and produce a daily portfolio-value curve. All engines are **vectorised** (numpy array maths, no per-day Python loop) for speed: DCA cumulative-sums a (days × assets) purchase matrix; Buy & Hold and Risk-Off snapshot holdings on the (few) trade days — reusing `_rebalance_to_target()` — then value all days in one pass. The daily valuation matches `_portfolio_value()` exactly (Σ units × price over non-NaN prices, NaN-priced assets keep their units). `_portfolio_value()` and `_rebalance_to_target()` remain the per-trade primitives reused by the plugins' order-event generators.
- Buy & Hold: `run_backtest()` → `load_daily_closes()` → `_window_by_month()` → `simulate_lumpsum()` → `compute_metrics()`. `simulate_lumpsum()` deploys the whole lump sum once — equal-weight on the first buyable trading day (via `_rebalance_to_target(..., inv_frac=1.0)`) — then holds those units unchanged and values daily.
- DCA: `run_backtest()` → `load_daily_closes()` → `_window_by_month()` → `simulate_dca()` → `compute_metrics()`. `simulate_dca()` contributes once per month — on each month's last trading day (`_is_month_end_trading_day()`) — but values daily.
- Risk-Off: `load_daily_closes()` (full daily history for look-back) → `build_equal_weight_index()` → `compute_riskoff_signals()` (three booleans `_sma_trend_signal`/`_ytd_return_signal`/`_first_n_days_signal`, summed to a 0..3 positive-signal count, ÷3 = daily target fraction) → `simulate_riskoff()`. Signals are evaluated daily and `simulate_riskoff()` buys/sells to the new target (via `_rebalance_to_target()`) **only on the day the target changes**, holding (and letting the fraction drift) in between.
- `compute_metrics()` is shared by all (daily returns annualised with √252; CAGR from the calendar span; for the lump-sum path `total_invested` is the initial investment). It returns **9** metric keys (no Best/Worst Month) and rejects windows spanning fewer than three calendar months. `get_common_date_range()` finds the overlapping history across both baskets (the date slider stays month-granular).
- Order log (generic): `build_order_log(events, initial_capital)` turns a strategy's raw `OrderEvent`s into finalized `OrderRow`s (13 fixed columns — the raw trade plus derived value-after, running net deposits, P&L €/%, equity exposure, cash quota, period return — plus an `asset_values` dict carrying each asset's post-trade worth, which the UI renders as one extra value column per basket asset). The per-asset breakdown is produced by `_asset_values(holdings, prices)` (units × price per asset column, NaN-priced → 0, summing to `assets_after`) inside each plugin's events generator. This is the **only** order-log code here and is strategy-agnostic; each strategy emits its own `OrderEvent`s inside its plugin. `run_backtest()` now returns a 3-tuple `(series, metrics, order_log)`; its built-in fallback path (strategy=None, not used by the UI) runs Buy & Hold via `simulate_lumpsum()` and returns `order_log=None`.

**`src/strategies/` — strategy plugin system**
`base.py` defines the `BacktestStrategy` ABC and the `ConfigParam` dataclass (GUI-rendered, self-validating params). `registry.py` holds the `@register` decorator and `get_strategy()`/`list_strategies()`/`get_all_strategy_info()`. Plugins are thin orchestrators over `backtest.py`: `lumpsum.py` (Buy & Hold), `dca.py` (DCA) and `riskoff.py` (Risk-Off Signale). `__init__.py` imports every plugin module so importing the package registers all strategies; **import order is the dropdown order**, so `lumpsum` is imported first to make Buy & Hold the default GUI selection (`components._default_strategy_config()` picks `list_strategies()[0]`). Each `run()` returns `(pd.Series | None, dict | None, list[OrderRow] | None)` — the same 9 metric keys as `compute_metrics()` plus the strategy's order log. A plugin builds that log from its **own** strategy-specific events generator (`_lumpsum_order_events()` / `_dca_order_events()` / `_riskoff_order_events()`, which reuse the shared `_is_month_end_trading_day`/`_portfolio_value`/`_rebalance_to_target` primitives) fed to the generic `build_order_log()`, so adding a strategy and its order log touches only that plugin file — never `backtest.py`.

**`src/callbacks/`**
- `backtesting.py`: basket management, date range slider, `run_backtest` orchestration. Its run callback has **5 outputs** — chart, metrics table, status, chart style, and `bt-orders-store.data` (`{'a': rows|None, 'b': rows|None}`, each basket's display rows from `_order_rows()`); a separate `render_order_table` callback renders the **active** basket's stored rows into the always-visible `bt-orders-content` div (the tabs only select; the table never lives in a `display:none` pane), and `download_orders` exports the active basket's rows as CSV / Excel via `dcc.send_data_frame` → `dcc.Download`. The daily value curve is thinned to ~2000 points by `_downsample_for_plot()` before plotting (the chart is only ~1-2k px wide; metrics use the full series) to keep the JSON payload and browser SVG render small.
- `__init__.py` imports `backtesting` so that importing the package registers all callbacks

**UI building blocks**
- `layout.py`: top-level layout only, calls `_basket_ui()` from `components.py`; below the chart/metrics it places an "Orders" heading row (title left, a `dbc.DropdownMenu` download-icon button right → CSV / Excel items `bt-dl-csv`/`bt-dl-xlsx`, plus a hidden `dcc.Download`), a `dbc.Tabs` (id `bt-orders-tabs`, `tab_id` `a`/`b`) that **only selects** the active basket, a `dcc.Store` (`bt-orders-store`), and a single **always-visible** `bt-orders-content` div. The order table is **not** put inside the tab panes: a Bootstrap tab pane is `display:none` until activated and Safari won't repaint callback-injected content in the active pane until a tab switch (the table stayed blank until toggling B→A). `render_order_table` re-renders the active basket's stored rows into the always-visible div, which paints first-time like the chart/metrics.
- `components.py`: `_basket_ui`, `_render_basket_list`, `_metrics_table`, and the order-table pair driven by the `_ORDER_COLUMNS` spec **plus** dynamic per-asset value columns: `_order_rows(orders)` formats the log into JSON display rows (`{column label: text}`, the fixed `_ORDER_COLUMNS` followed by one column per basket asset from `_order_asset_columns()`, stored in `bt-orders-store` and shared by the table and the CSV/Excel download), and `_order_table_component(rows)` derives its column order from the row dict keys (so the per-asset columns appear automatically) and renders them as a **native HTML `<table class="order-table">`** inside a single `dcc.Markdown(dangerously_allow_html=True)` (not thousands of `html.Tr`/`html.Td` — those cost *seconds* of dash-renderer per-component render; native HTML also needs no JS layout measurement, so it paints first-time inside the tabs UI), or the 'No orders.' placeholder.
- `styles.py`: shared inline style dicts. The metrics table is an `html.Table`; the order table is native HTML styled by the `.order-table` / `.order-table-wrapper` rules in `assets/layout.css` (80vh `max-height` scroll box, sticky header + sticky first column via `position: sticky`).

**Data flow**
- On startup: `master.parquet` → `config.df` (asset catalogue with symbol, name, exchange, asset_class, filename, interval)
- On user interaction: per-asset parquet files (`{BASE_URL}/{filename}`) fetched on demand; each file has OHLCV columns with a datetime index
- Basket state is held in `dcc.Store` components client-side; the backtest callback reads them as lists of `{filename, symbol, name}` dicts

**`@log_time` decorator** (`utils.py`) wraps every callback to emit DEBUG-level timing.
