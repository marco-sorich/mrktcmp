# mrktcmp — Markets Compare

A Plotly Dash web application for market data visualisation and DCA (Dollar-Cost Averaging) backtesting across user-defined asset baskets.

## Features

**Market Data tab**
- Searchable asset selector filtered by asset class (stocks, crypto, ETFs, …)
- Candlestick + volume subplot chart with a shared x-axis
- Y-axis auto-rescales to visible candles when zooming or panning

**Backtesting tab**
- Build two asset baskets (Basket A and Basket B) from any combination of assets
- Date range slider auto-calculated from the overlapping history of all selected assets
- Monthly DCA simulation: 1,000 € invested per basket per month, split equally across available assets
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

## Running Tests

```
pytest
```

## Project Structure

```
src/
  app.py        # Dash application, layout, and all callbacks
  backtest.py   # DCA simulation engine (load, simulate, compute metrics)
tests/
  test_app.py
  test_backtest.py
```
