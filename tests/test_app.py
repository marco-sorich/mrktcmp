import os
import pytest
import pandas as pd
from unittest.mock import patch
import plotly.graph_objects as go

# Ensure no BASE_URL during import so df=None and no network calls are made
os.environ.pop("BASE_URL", None)

import src.app as app_module
from src.app import update_asset_type_options, update_chart

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_DF = pd.DataFrame(
    {
        "asset_class": ["stocks", "stocks", "crypto"],
        "symbol": ["AAPL", "GOOGL", "BTC"],
        "interval": ["1d", "1d", "1d"],
        "name": ["Apple Inc", "Alphabet Inc", "Bitcoin"],
        "exchange": ["NASDAQ", "NASDAQ", "Binance"],
        "country": ["US", "US", "Global"],
        "category": ["Tech", "Tech", "Crypto"],
        "first_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "last_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "filename": ["aapl.parquet", "googl.parquet", "btc.parquet"],
    }
)

SAMPLE_OHLCV = pd.DataFrame(
    {
        "Open": [150.0, 155.0, 160.0],
        "High": [155.0, 160.0, 165.0],
        "Low": [148.0, 153.0, 158.0],
        "Close": [153.0, 158.0, 163.0],
        "Volume": [1_000_000, 1_100_000, 1_200_000],
    },
    index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
)
SAMPLE_OHLCV.index.name = "Date"

BASE_URL = "http://example.com"


# ---------------------------------------------------------------------------
# update_asset_type_options
# ---------------------------------------------------------------------------


class TestUpdateAssetTypeOptions:
    def test_no_asset_class_returns_disabled_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_type_options(None, None, None)
        assert options == []
        assert disabled is True

    def test_df_none_returns_disabled_empty(self):
        with patch.object(app_module, "df", None):
            options, disabled = update_asset_type_options("stocks", None, None)
        assert options == []
        assert disabled is True

    def test_filters_by_asset_class(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_type_options("stocks", None, None)
        assert disabled is False
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values
        assert "btc.parquet" not in values

    def test_crypto_class_returns_only_crypto(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_type_options("crypto", None, None)
        assert disabled is False
        assert len(options) == 1
        assert options[0]["value"] == "btc.parquet"

    def test_search_by_symbol_case_insensitive(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, _ = update_asset_type_options("stocks", "aapl", None)
        assert len(options) == 1
        assert options[0]["value"] == "aapl.parquet"

    def test_search_by_name_partial_match(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, _ = update_asset_type_options("stocks", "alphabet", None)
        assert len(options) == 1
        assert options[0]["value"] == "googl.parquet"

    def test_search_with_no_matches_returns_empty_options(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_type_options("stocks", "ZZZZZZ", None)
        assert options == []
        assert disabled is False

    def test_option_label_format(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, _ = update_asset_type_options("crypto", None, None)
        assert options[0]["label"] == "BTC — Bitcoin (1d)"

    def test_current_value_appended_when_not_in_search_results(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            # search narrows to GOOGL, but current selection is AAPL
            options, _ = update_asset_type_options("stocks", "GOOGL", "aapl.parquet")
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values

    def test_current_value_not_duplicated_when_already_visible(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, _ = update_asset_type_options("stocks", None, "aapl.parquet")
        values = [o["value"] for o in options]
        assert values.count("aapl.parquet") == 1

    def test_unknown_current_value_not_appended(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, _ = update_asset_type_options("stocks", "GOOGL", "nonexistent.parquet")
        values = [o["value"] for o in options]
        assert "nonexistent.parquet" not in values

    def test_results_capped_at_thirty(self):
        large_df = pd.DataFrame(
            {
                "asset_class": ["stocks"] * 40,
                "symbol": [f"SYM{i}" for i in range(40)],
                "interval": ["1d"] * 40,
                "name": [f"Company {i}" for i in range(40)],
                "exchange": ["NYSE"] * 40,
                "country": ["US"] * 40,
                "category": ["Tech"] * 40,
                "first_date": ["2020-01-01"] * 40,
                "last_date": ["2024-01-01"] * 40,
                "filename": [f"sym{i}.parquet" for i in range(40)],
            }
        )
        with patch.object(app_module, "df", large_df):
            options, _ = update_asset_type_options("stocks", None, None)
        assert len(options) == 30

    def test_search_results_capped_at_thirty(self):
        large_df = pd.DataFrame(
            {
                "asset_class": ["stocks"] * 40,
                "symbol": [f"ABC{i}" for i in range(40)],
                "interval": ["1d"] * 40,
                "name": [f"Company {i}" for i in range(40)],
                "exchange": ["NYSE"] * 40,
                "country": ["US"] * 40,
                "category": ["Tech"] * 40,
                "first_date": ["2020-01-01"] * 40,
                "last_date": ["2024-01-01"] * 40,
                "filename": [f"sym{i}.parquet" for i in range(40)],
            }
        )
        with patch.object(app_module, "df", large_df):
            options, _ = update_asset_type_options("stocks", "ABC", None)
        assert len(options) == 30


# ---------------------------------------------------------------------------
# update_chart
# ---------------------------------------------------------------------------


class TestUpdateChart:
    def test_no_filename_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL):
            fig, headline = update_chart(None)
        assert headline == ""

    def test_no_base_url_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", None):
            fig, headline = update_chart("aapl.parquet")
        assert headline == ""

    def test_df_none_returns_empty(self):
        with patch.object(app_module, "df", None), \
             patch.object(app_module, "base_url", BASE_URL):
            fig, headline = update_chart("aapl.parquet")
        assert headline == ""

    def test_valid_input_returns_figure(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            fig, headline = update_chart("aapl.parquet")
        assert isinstance(fig, go.Figure)

    def test_chart_contains_candlestick_trace(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            fig, _ = update_chart("aapl.parquet")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Candlestick" in trace_types

    def test_chart_contains_volume_scatter_trace(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            fig, _ = update_chart("aapl.parquet")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter" in trace_types

    def test_headline_shows_asset_name(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            _, headline = update_chart("aapl.parquet")
        assert any("Apple Inc" in str(c) for c in headline)

    def test_headline_shows_exchange_and_country(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=SAMPLE_OHLCV):
            _, headline = update_chart("aapl.parquet")
        combined = " ".join(str(c) for c in headline)
        assert "NASDAQ" in combined
        assert "US" in combined

    def test_parquet_read_error_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", side_effect=Exception("network error")):
            fig, headline = update_chart("aapl.parquet")
        assert headline == ""

    def test_unknown_filename_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL):
            # filename not in df → .iloc[0] raises IndexError → caught → empty
            fig, headline = update_chart("nonexistent.parquet")
        assert headline == ""


# ---------------------------------------------------------------------------
# App-level structure
# ---------------------------------------------------------------------------


class TestAppStructure:
    def test_server_is_flask_instance(self):
        import flask
        assert isinstance(app_module.server, flask.Flask)

    def test_layout_is_not_none(self):
        assert app_module.app.layout is not None

    def test_df_is_none_when_no_base_url(self):
        assert app_module.df is None

    def test_asset_classes_empty_when_no_base_url(self):
        assert app_module.assetsClasses == []
