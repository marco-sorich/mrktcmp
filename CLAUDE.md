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

**`src/strategies/` — strategy plugin system**
`base.py` defines the `BacktestStrategy` ABC and the `ConfigParam` dataclass (GUI-rendered, self-validating params). `registry.py` holds the `@register` decorator and `get_strategy()`/`list_strategies()`/`get_all_strategy_info()`. Plugins are thin orchestrators over `backtest.py`: `dca.py` (DCA) and `riskoff.py` (Risk-Off Signale). `__init__.py` imports every plugin module so importing the package registers all strategies. Each `run()` returns `(pd.Series | None, dict | None)` with the same 11 metric keys as `compute_metrics()`.

**`src/callbacks/`**
- `backtesting.py`: basket management, date range slider, `run_backtest` orchestration
- `__init__.py` imports `backtesting` so that importing the package registers all callbacks

**UI building blocks**
- `layout.py`: top-level layout only, calls `_basket_ui()` from `components.py`
- `components.py`: `_basket_ui`, `_render_basket_list`, `_metrics_table`
- `styles.py`: shared inline style dicts

**Data flow**
- On startup: `master.parquet` → `config.df` (asset catalogue with symbol, name, exchange, asset_class, filename, interval)
- On user interaction: per-asset parquet files (`{BASE_URL}/{filename}`) fetched on demand; each file has OHLCV columns with a datetime index
- Basket state is held in `dcc.Store` components client-side; the backtest callback reads them as lists of `{filename, symbol, name}` dicts

**`@log_time` decorator** (`utils.py`) wraps every callback to emit DEBUG-level timing.
