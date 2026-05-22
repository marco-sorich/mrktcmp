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

2. Create a `.env` file (or set environment variables):
   ```
   BASE_URL=https://your-data-host/path   # root URL or local path to parquet files
   LOG_LEVEL=INFO                          # optional, default INFO
   DASH_DEBUG=false                        # set true for hot-reload in development
   ```

## Running

**Development:**
```
python src/app.py
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
