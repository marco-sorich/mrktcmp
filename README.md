# mrktcmp — Markets Compare

A Plotly Dash web application for backtesting across user-defined asset baskets.

## Features

- Build two asset baskets (Basket A and Basket B) from any combination of assets
- **Multi-currency support**: pick a reporting (base) currency from a dropdown (default EUR); every asset's prices are
  converted from its quote currency into that base currency using daily FX rates before the backtest, so baskets mixing
  USD/GBP/EUR/… assets are compared on a common basis (the standard *unhedged* approach). Currency pairs can themselves
  be added to a basket as assets. Each asset's trading currency is shown in the search dropdown and basket list, and the
  order tables show each trade's price in both the trading and the reporting currency plus the FX rate used
- Date range slider auto-calculated from the overlapping history of all selected assets
- Pluggable backtesting strategies — each basket can use a different strategy with its own configurable parameters; a
  click on the ⓘ icon next to the strategy selector expands a rich-text description explaining the selected strategy and its parameters
- **Buy & Hold** (default): a single lump sum (10,000, in the reporting currency) is invested in full on the first
  trading day, split equally across available assets, and then held unchanged until the end — one order, no rebalancing
- **DCA**: a fixed amount (1,000, in the reporting currency) contributed per basket every month (on each month's last trading day),
  split equally across available assets; the portfolio is valued on every trading day
- **Risk-Off Signale**: a one-off lump sum is held as cash and tactically shifted between the basket and cash.
  Three market signals are evaluated on the basket as a whole *every day* — 200-day trend, year-to-date return, and the
  January barometer (first 10 trading days of the year) — and the number of *positive* signals sets the target invested
  fraction in thirds (3 → 100 %, 2 → 66 %, 1 → 33 %, 0 → all cash). The basket is bought/sold to the new target on the
  day the signal changes and then held (drifting with the market) until the next change
- All strategies value the portfolio on a **daily** basis, so the chart is a dense daily curve and the risk metrics are
  daily-correct
- Side-by-side performance metrics: Total Return, CAGR, Sharpe Ratio, Max. Drawdown, Volatility, Calmar Ratio, and more
- Per-order transaction tables (one per basket, switchable via tabs) listing every buy/sell with its date, side,
  pre-/post-trade value, inflow, asset/cash split, running net deposits, P&L (absolute in the reporting currency, and %),
  equity exposure, cash quota, and period return. Then, per basket asset, a value column (units × price, in the reporting
  currency) and its price column(s): for an asset that trades in a non-reporting currency **both** the trading-currency
  close and the converted reporting-currency close are shown side by side; for an asset already in the reporting currency
  a single price column is shown. Finally one column per currency pair (`{LOCAL}{BASE}=X`) used in the basket, giving the
  exact FX rate each trade was converted at. The table fills ~80 % of the viewport height with a
  sticky header row and sticky first column for easy scanning of long logs, and a download button (next to the
  "Orders" heading) exports the active basket's table (per-asset value/price + FX-pair columns included) as **CSV** or
  **Excel (.xlsx)**

## Requirements

- Python 3.14+
- A data source (parquet files) accessible via `BASE_URL` — a `master.parquet` catalogue plus one parquet file per asset

## Data Source: Parquet File Format

The app reads two kinds of Parquet files from the directory pointed to by `BASE_URL`
(either a URL prefix such as `https://host/data` or a local filesystem path such as `/mnt/data`).

### `master.parquet` — asset catalogue

One row per available asset. Loaded once at startup; used to populate asset search dropdowns.

| Column | Type | Description |
|---|---|---|
| `symbol` | `str` | Short ticker or identifier shown in the UI (e.g. `AAPL`) |
| `name` | `str` | Human-readable asset name (e.g. `Apple Inc.`) |
| `asset_class` | `str` | Category shown as a filter tab (e.g. `stocks`, `crypto`, `etf`) |
| `filename` | `str` | Name of the per-asset parquet file (e.g. `aapl.parquet`) |
| `exchange` | `str` | Exchange name used for sorting and display (e.g. `NASDAQ`) |
| `country` | `str` | *(optional)* Country code or name |
| `interval` | `str` | *(optional)* Data cadence hint (e.g. `1d`) |
| `currency` | `str` | *(optional)* The currency the asset's prices are quoted in (e.g. `USD`, `EUR`, `GBp`). Drives FX conversion into the reporting currency. Blank/`0`/unknown → left unconverted |

To enable multi-currency comparison, the catalogue may also include **FX-pair rows** with `asset_class == 'currency'`:
one row per downloadable currency pair, with `symbol` like `USDEUR=X`, `name` like `USD/EUR`, `currency` set to the
pair's *quote* currency (`EUR`), and a `filename` pointing at that pair's OHLCV file. Conversion multiplies an asset's
local-currency close by the `{LOCAL}{BASE}=X` pair's daily close (units of base per unit of local).

The catalogue is sorted at startup by `asset_class → symbol → exchange`; the order controls
how assets appear in search results.

### Per-asset parquet files — OHLCV price data

One file per asset, named by the `filename` column in the catalogue. It contains OHLCV columns (`Open`, `High`, `Low`, `Volume`).

| Requirement | Detail |
|---|---|
| **Index** | `DatetimeIndex` — timezone-aware or naive (naive is localized to UTC at load time) |
| **`Open` column** | Daily opening price as a numeric type; `NaN` or non-positive values are treated as missing (no trade placed) |
| **`High` column** | Daily high price as a numeric type; `NaN` or non-positive values are treated as missing (no trade placed) |
| **`Low` column** | Daily low price as a numeric type; `NaN` or non-positive values are treated as missing (no trade placed) |
| **`Close` column** | Daily closing price as a numeric type; `NaN` or non-positive values are treated as missing (no trade placed) |
| **Cadence** | **Daily** resolution. The backtesting engines are designed for daily data; monthly or intraday files will load but produce incorrect results |
| **History** | As much history as available — the Risk-Off strategy needs at least 200 trading days of warm-up before the backtest window starts for its SMA signal |

Example minimal structure (pseudo-code):

```
DatetimeIndex (daily, UTC)   Close
2020-01-02                   300.35
2020-01-03                   298.10
2020-01-06                   301.75
…
```

Files are fetched on demand when a user adds an asset to a basket; only the `Close` column is read
for the date-range slider, and the full file is read when a backtest is run.

## Setup

1. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   (Dependencies and build metadata are defined in `pyproject.toml`. No separate `requirements*.txt` files exist.)
   
   ...or...
   
   Install the package with dependencies on production server:
   ```bash
   pip install .
   ```
   (Dependencies and build metadata are defined in `pyproject.toml`. No separate `requirements*.txt` files exist.)
   
   ...or...

2. Set environment variables:
   ```bash
   # URL or local path to parquet files
   BASE_URL=https://your-data-host/path

   # optional: set log level to CRITICAL, ERROR, WARNING, INFO, DEBUG (default: INFO)
   LOG_LEVEL=INFO

   # optional: (de-)activate dev tools of Plotly/Dash (default: false)
   DASH_DEBUG=false

   # optional: default reporting (base) currency pre-selected in the GUI (default: EUR)
   BASE_CURRENCY=EUR
   ```

## Running

**Development:**
```
gunicorn --reload src.app:server
```

**Production (gunicorn):**
```
gunicorn src.app:server
```

Open `http://127.0.0.1:8050/` in your browser.


## QA Checks

All checks require development dependencies, which are installed via `pip install -e ".[dev]"` (this includes **mypy 2.1.0**, the version CI runs — older 1.x releases silently miss some errors):

```bash
# Linting
flake8 --max-complexity=10 --max-line-length=127 ./src

# Type checking (mypy 2.1.0)
mypy --explicit-package-bases --ignore-missing-imports src/backtest.py src/strategies/
mypy --explicit-package-bases --ignore-missing-imports tests/test_backtest.py tests/test_strategies.py

# Testing
pytest
```

## Versioning

The application version is automatically derived from git tags via `setuptools-scm`. The version displays in the page footer. To create a release:

```bash
git tag -a v<VERSION> -m "Release <VERSION>"  # e.g., v0.1.0
git push origin v<VERSION>
```

Without tags, the version will be a development version (e.g., `0.1.dev3+g<hash>`).

## Project Structure

```
src/
  app.py              # Dash application entry point
  backtest.py         # pure simulation engines, all on daily close prices:
                      #   Buy & Hold (load_daily_closes, _window_by_month, simulate_lumpsum),
                      #   DCA (load_daily_closes, _window_by_month, simulate_dca) and
                      #   Risk-Off (load_daily_closes, build_equal_weight_index,
                      #   compute_riskoff_signals, simulate_riskoff); shared compute_metrics
  strategies/
    __init__.py       # imports all plugins to trigger registration at startup
                      #   (import order = dropdown order; lumpsum first = default)
    base.py           # BacktestStrategy ABC and ConfigParam dataclass
    registry.py       # @register decorator, get_strategy(), list_strategies()
    lumpsum.py        # Buy & Hold strategy plugin (one single initial investment)
    dca.py            # Dollar-Cost Averaging strategy plugin
    riskoff.py        # Risk-Off signal strategy plugin (lump sum + tactical cash)
  callbacks/
    backtesting.py    # backtesting callbacks
  config.py           # startup singleton: logging, env vars, master.parquet
  components.py       # reusable UI component builders
  layout.py           # top-level Dash layout
  styles.py           # shared inline style dicts
  utils.py            # @log_time decorator
tests/
  test_app.py
  test_backtest.py
  test_strategies.py  # plugin system: ConfigParam, registry, BuyHold/DCA/RiskOff, backward compat
```

## Adding a New Backtesting Strategy

1. Create `src/strategies/my_strategy.py` and implement `BacktestStrategy`:

   ```python
   from src.strategies.base import BacktestStrategy, ConfigParam
   from src.strategies.registry import register

   @register
   class MyStrategy(BacktestStrategy):
       @classmethod
       def get_name(cls) -> str:
           return "My Strategy"

       @classmethod
       def get_description(cls) -> str:
           return "One-sentence description shown in the GUI."

       @classmethod
       def get_config_schema(cls) -> list[ConfigParam]:
           return [
               ConfigParam(
                   key='my_param',
                   label='My Parameter',
                   type='float',          # 'int', 'float', or 'select'
                   default=42.0,          # always required — GUI pre-fills this value
                   min_value=1.0,
                   max_value=1000.0,
               ),
           ]

       def run(self, base_url, filenames, start_date, end_date, df_meta, params):
           # resolve_params merges caller-supplied values with schema defaults
           # so that params={} (no user input) always works correctly.
           resolved = self.resolve_params(params)
           my_param = float(resolved['my_param'])
           # ... compute portfolio (pd.Series) and metrics (dict[str, str]) ...
           # Build this strategy's order log: emit your own list of
           # backtest.OrderEvent (the strategy-specific part — when you trade,
           # the side, the inflow, the asset/cash split) and hand it to the
           # generic builder, which derives all the shared columns.
           from src.backtest import build_order_log
           order_log = build_order_log(my_order_events, initial_capital=0.0)
           return portfolio, metrics, order_log
   ```

2. Register the plugin by adding one import line to `src/strategies/__init__.py`:

   ```python
   import src.strategies.my_strategy  # noqa: F401
   ```

3. Add tests in `tests/test_strategies.py` following the existing `TestDCAStrategy` pattern.

**Plugin contract:**
- `run()` must return `(pd.Series | None, dict[str, str] | None, list[OrderRow] | None)`.
- The metrics dict must contain exactly the same 9 keys as `compute_metrics()` in `backtest.py`.
- The order log is built **inside the plugin** from your own `OrderEvent`s passed to the generic
  `build_order_log()` in `backtest.py` — that generic builder never needs to change for a new strategy.
- Every `ConfigParam` must have a `default` value so the user is never forced to enter anything.
- For `int`/`float` params, `min_value` and `max_value` are required (the GUI uses them to prevent invalid input).
