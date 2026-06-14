# mrktcmp — Markets Compare

A Plotly Dash web application for backtesting across user-defined asset baskets.

## Features

- Build two asset baskets (Basket A and Basket B) from any combination of assets
- Date range slider auto-calculated from the overlapping history of all selected assets
- Pluggable backtesting strategies — each basket can use a different strategy with its own configurable parameters
- **DCA** (default): a fixed amount (1,000 €) contributed per basket every month (on each month's last trading day),
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
  pre-/post-trade value, inflow, asset/cash split, running net deposits, P&L (€ and %), equity exposure, cash quota,
  and period return. The table fills ~80 % of the viewport height with a sticky header row and sticky first column for
  easy scanning of long logs, and a download button (next to the "Orders" heading) exports the active basket's
  table as **CSV** or **Excel (.xlsx)**

## Requirements

- Python 3.9+
- A data source (parquet files) accessible via `BASE_URL` — a `master.parquet` catalogue plus one parquet file per asset

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   # URL or local path to parquet files
   BASE_URL=https://your-data-host/path

   # optional: set log level to CRITICAL, ERROR, WARNING, INFO, DEBUG (default: INFO)
   LOG_LEVEL=INFO

   # optional: (de-)activate dev tools of Plotly/Dash (default: false)
   DASH_DEBUG=false
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

Install the development dependencies first (this also pins **mypy 2.1.0**, the
version CI runs — older 1.x releases silently miss some errors):

```bash
pip install -r requirements_dev.txt
```

```bash
# Linting
flake8 --max-complexity=10 --max-line-length=127

# Type checking (mypy 2.1.0)
mypy --explicit-package-bases --ignore-missing-imports src/backtest.py src/strategies/
mypy --explicit-package-bases --ignore-missing-imports tests/test_backtest.py tests/test_strategies.py

# Testing
pytest
```

## Project Structure

```
src/
  app.py              # Dash application entry point
  backtest.py         # pure simulation engines, all on daily close prices:
                      #   DCA (load_daily_closes, _window_by_month, simulate_dca) and
                      #   Risk-Off (load_daily_closes, build_equal_weight_index,
                      #   compute_riskoff_signals, simulate_riskoff); shared compute_metrics
  strategies/
    __init__.py       # imports all plugins to trigger registration at startup
    base.py           # BacktestStrategy ABC and ConfigParam dataclass
    registry.py       # @register decorator, get_strategy(), list_strategies()
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
  test_strategies.py  # plugin system: ConfigParam, registry, DCAStrategy, backward compat
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
