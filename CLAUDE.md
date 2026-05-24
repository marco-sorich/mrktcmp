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

# Type check
mypy src/backtest.py
mypy tests/test_backtest.py

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

**`src/backtest.py` — pure DCA engine**
No Dash dependencies. Call chain: `run_backtest()` → `load_monthly_closes()` → `simulate_dca()` → `compute_metrics()`. Separately, `get_common_date_range()` finds the overlapping history across both baskets. All parquet I/O happens here.

**`src/callbacks/`**
- `chart.py`: Market Data tab — asset class filter, search, candlestick+volume chart, y-axis sync on zoom
- `backtesting.py`: Backtesting tab — basket management, date range slider, `run_backtest` orchestration
- `__init__.py` imports both modules so that importing the package registers all callbacks

**UI building blocks**
- `layout.py`: top-level layout only, calls `_basket_ui()` from `components.py`
- `components.py`: `_basket_ui`, `_render_basket_list`, `_metrics_table`
- `styles.py`: shared inline style dicts

**Data flow**
- On startup: `master.parquet` → `config.df` (asset catalogue with symbol, name, exchange, asset_class, filename, interval)
- On user interaction: per-asset parquet files (`{BASE_URL}/{filename}`) fetched on demand; each file has OHLCV columns with a datetime index
- Basket state is held in `dcc.Store` components client-side; the backtest callback reads them as lists of `{filename, symbol, name}` dicts
- The y-axis zoom callback stores serialised OHLCV JSON in a `dcc.Store` to avoid re-fetching

**`@log_time` decorator** (`utils.py`) wraps every callback to emit DEBUG-level timing.
