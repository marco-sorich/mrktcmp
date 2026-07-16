# Project Context for Vibe

**Project**: Plotly/Dash web app for comparing financial asset baskets via backtesting.

---

## Quick Start

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Development server (hot-reload)
python src/app.py
# or
gunicorn --reload src.app:server

# Production
gunicorn --config gunicorn.conf.py src.app:server
```

---

## Key Files

- **`src/app.py`** - Entry point
- **`src/config.py`** - Singleton: loads `.env`, configures logging, fetches `master.parquet`
- **`src/backtest.py`** - Core simulation engines (Buy & Hold, DCA, Risk-Off)
- **`src/strategies/`** - Strategy plugins
- **`src/callbacks/`** - Dash callbacks
- **`src/components.py`** - UI components
- **`src/layout.py`** - Top-level layout

---

## Environment Variables

```bash
BASE_URL=https://your-data-host/path    # Parquet files location (must contain master.parquet)
LOG_LEVEL=INFO                         # CRITICAL, ERROR, WARNING, INFO, DEBUG
DASH_DEBUG=true                        # Enable dev tools
BASE_CURRENCY=EUR                      # Default reporting currency
```

`BASE_URL` must point to a directory containing `master.parquet` (asset catalogue) and one parquet file per asset (OHLCV data). The catalogue's `currency` column records each asset's quote currency.

---

## Architecture Notes

- All parquet I/O funnels through `_read_close_series()` with **process-wide caching** (each file read once per process)
- Currency normalization happens in `load_daily_closes()` via FX pairs (`{LOCAL}{BASE}=X`)
- Three strategies: **Buy & Hold** (lump sum), **DCA** (monthly contributions), **Risk-Off** (signal-based)
- `compute_metrics()` returns 9 standard metric keys (annualized returns, CAGR, etc.)
- All engines run on **daily** close prices and are vectorized (numpy array math, no per-day loops)

### Strategy Details
- **Buy & Hold**: Invests lump sum once, equal-weight across all assets when all are first priced
- **DCA**: Contributes monthly on each month's last trading day, values daily
- **Risk-Off**: Signal-based (SMA trend, YTD return, first-N-days), gates to 0 until all assets priced

---

## Development Workflow

```bash
# Type checking (requires mypy 2.1.0)
mypy --explicit-package-bases --ignore-missing-imports src/backtest.py src/strategies/
mypy --explicit-package-bases --ignore-missing-imports tests/test_backtest.py tests/test_strategies.py

# Linting
flake8 --max-complexity=10 --max-line-length=127 ./src

# Tests
pytest
pytest tests/test_backtest.py
```

- All dependencies pinned in `pyproject.toml`
- Dependabot creates weekly PRs for updates
- Import order in `src/strategies/__init__.py` determines dropdown order

---

## Data Flow

- **Startup**: `master.parquet` → `config.df` (asset catalogue)
- **On demand**: Per-asset parquet files fetched from `{BASE_URL}/{filename}`
- **Basket state**: Held in `dcc.Store` components client-side

---

## Vibe-Specific Notes

- Prefer vectorized numpy operations over Python loops
- Cache parquet reads via `_read_close_series()` - it's already optimized
- Use `config.df` for asset catalogue access
- Strategy plugins: implement `BacktestStrategy` ABC from `src/strategies/base.py`
- Callbacks: registered via imports in `src/callbacks/__init__.py`
