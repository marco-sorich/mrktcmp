# mrktcmp — Markets Compare

A Plotly Dash web application for market data visualisation and backtesting across user-defined asset baskets.

## Features

**Market Data tab**
- Searchable asset selector filtered by asset class (stocks, crypto, ETFs, …)
- Candlestick + volume subplot chart with a shared x-axis
- Y-axis auto-rescales to visible candles when zooming or panning

**Backtesting tab**
- Build two asset baskets (Basket A and Basket B) from any combination of assets
- Date range slider auto-calculated from the overlapping history of all selected assets
- Pluggable backtesting strategies — each basket can use a different strategy with its own configurable parameters
- Monthly DCA simulation (default): 1,000 € invested per basket per month, split equally across available assets
- Side-by-side performance metrics: Total Return, CAGR, Sharpe Ratio, Max. Drawdown, Volatility, Calmar Ratio, and more

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

```bash
# Linting
flake8 --max-complexity=10 --max-line-length=127

# Type checking
mypy --explicit-package-bases --ignore-missing-imports src/backtest.py src/strategies/
mypy --explicit-package-bases --ignore-missing-imports tests/test_backtest.py tests/test_strategies.py

# Testing
pytest
```

## Project Structure

```
src/
  app.py              # Dash application entry point
  backtest.py         # DCA simulation engine: load_monthly_closes, simulate_dca, compute_metrics
  strategies/
    __init__.py       # imports all plugins to trigger registration at startup
    base.py           # BacktestStrategy ABC and ConfigParam dataclass
    registry.py       # @register decorator, get_strategy(), list_strategies()
    dca.py            # Dollar-Cost Averaging strategy plugin
  callbacks/
    backtesting.py    # Backtesting tab callbacks
    chart.py          # Market Data tab callbacks
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
           # params always contains all keys from get_config_schema() with
           # either user-supplied values or the declared defaults.
           my_param = float(params['my_param'])
           # ... compute portfolio (pd.Series) and metrics (dict[str, str]) ...
           return portfolio, metrics
   ```

2. Register the plugin by adding one import line to `src/strategies/__init__.py`:

   ```python
   import src.strategies.my_strategy  # noqa: F401
   ```

3. Add tests in `tests/test_strategies.py` following the existing `TestDCAStrategy` pattern.

**Plugin contract:**
- `run()` must return `(pd.Series | None, dict[str, str] | None)`.
- The metrics dict must contain exactly the same 11 keys as `compute_metrics()` in `backtest.py`.
- Every `ConfigParam` must have a `default` value so the user is never forced to enter anything.
- For `int`/`float` params, `min_value` and `max_value` are required (the GUI uses them to prevent invalid input).
