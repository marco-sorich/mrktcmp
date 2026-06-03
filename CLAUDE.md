# CLAUDE.md

This is a Plotly/Dash based webpage for comparing baskets, containing financial asset, via backtesting.

## Commands

```bash
# Development server (hot-reload via DASH_DEBUG=true in .env)
python src/app.py
# or
gunicorn --reload src.app:server

# Production
gunicorn src.app:server

# Linting
flake8 --max-complexity=10 --max-line-length=127

# Type check (requires mypy 2.1.0 — pinned in requirements_dev.txt; older 1.x
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
Loads `.env`, configures logging, fetches `master.parquet` from `BASE_URL` into the module-level global `df` (a DataFrame). Exposes `base_url`, `assetsClasses`, `df`, and `log` to the rest of the app. All other modules import this as `import src.config as _config`.

**`src/backtest.py` — pure simulation engines (no Dash dependencies)**
Two strategies share this module; all parquet I/O happens here.
- DCA: `run_backtest()` → `load_monthly_closes()` → `simulate_dca()` → `compute_metrics()`.
- Risk-Off: `load_daily_closes()` (full daily history for look-back) → `build_equal_weight_index()` → `compute_riskoff_signals()` (three booleans `_sma_trend_signal`/`_ytd_return_signal`/`_first_n_days_signal`, summed to a 0..3 positive-signal count) → `simulate_riskoff()` (lump sum rebalanced between basket and cash via `_rebalance_to_target()`).
- `compute_metrics()` is shared by both (for the lump-sum path `total_invested` is the initial investment). `get_common_date_range()` finds the overlapping history across both baskets.
- **Event ledger:** `simulate_dca()` and `simulate_riskoff()` also return a list of per-event dicts (date, `value_pre_trade`/`value_post_trade`, `cash`, `external_flow`, and signed per-asset `legs`). `run_backtest()` passes these through `_enrich_events()`, which adds derived KPIs (`cum_invested`, `pnl`/`pnl_pct`, `equity_pct`/`cash_pct`, `period_return_pct`). `run_backtest()` and every strategy `run()` therefore return a **3-tuple** `(portfolio, metrics, events)`.
- **`BacktestRun` dataclass:** the generic comparable result unit (`run_id`, `label`, `color`, `portfolio`, `metrics`, `events`). A "run" is a (basket, strategy) pair; the chart and transaction tables iterate a `list[BacktestRun]`, so adding more baskets or basket×strategy comparisons needs no rendering change.

**`src/strategies/` — strategy plugin system**
`base.py` defines the `BacktestStrategy` ABC and the `ConfigParam` dataclass (GUI-rendered, self-validating params). `registry.py` holds the `@register` decorator and `get_strategy()`/`list_strategies()`/`get_all_strategy_info()`. Plugins are thin orchestrators over `backtest.py`: `dca.py` (DCA) and `riskoff.py` (Risk-Off Signale). `__init__.py` imports every plugin module so importing the package registers all strategies. Each `run()` returns `(pd.Series | None, dict | None, list[dict] | None)` — portfolio, the same 11 metric keys as `compute_metrics()`, and the raw event ledger (or `None` if the strategy has none; `run_backtest()` enriches it).

**`src/callbacks/`**
- `chart.py`: Market Data tab — asset class filter, search, candlestick+volume chart, y-axis sync on zoom
- `backtesting.py`: Backtesting tab — basket management, date range slider, `run_backtest` orchestration
- `__init__.py` imports both modules so that importing the package registers all callbacks

**UI building blocks**
- `layout.py`: top-level layout only, calls `_basket_ui()` from `components.py`. Holds an empty `bt-tx-section` container that the backtest callback fills with the transaction tabs.
- `components.py`: `_basket_ui`, `_render_basket_list`, `_metrics_table`, plus `_transaction_table` / `_transaction_section` (a `dbc.Tabs`, one `dbc.Tab` of transactions per run)
- `styles.py`: shared inline style dicts
- `assets/transactions.js`: clientside callbacks for the transaction tables — scroll-to-top/bottom buttons and graph-click → activate the run's tab and scroll/highlight the nearest event row. Row-click → chart marker is a server callback (`highlight_chart_point`, `Output('bt-chart','figure', allow_duplicate=True)`). Dynamically-created IDs require `suppress_callback_exceptions=True` (set in `app.py`).

**Data flow**
- On startup: `master.parquet` → `config.df` (asset catalogue with symbol, name, exchange, asset_class, filename, interval)
- On user interaction: per-asset parquet files (`{BASE_URL}/{filename}`) fetched on demand; each file has OHLCV columns with a datetime index
- Basket state is held in `dcc.Store` components client-side; the backtest callback reads them as lists of `{filename, symbol, name}` dicts
- The y-axis zoom callback stores serialised OHLCV JSON in a `dcc.Store` to avoid re-fetching

**`@log_time` decorator** (`utils.py`) wraps every callback to emit DEBUG-level timing.
