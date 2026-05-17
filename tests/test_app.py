import os
import pytest
import pandas as pd
from unittest.mock import patch
import plotly.graph_objects as go

# Ensure no BASE_URL during import so df=None and no network calls are made
os.environ.pop("BASE_URL", None)

import src.app as app_module
from src.app import update_asset_class, update_asset_search, update_chart

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
# update_asset_class
# ---------------------------------------------------------------------------


class TestUpdateAssetClass:
    def test_no_asset_class_returns_disabled_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_class(None)
        assert options == []
        assert disabled is True

    def test_df_none_returns_disabled_empty(self):
        with patch.object(app_module, "df", None):
            options, disabled = update_asset_class("stocks")
        assert options == []
        assert disabled is True

    def test_filters_by_asset_class(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_class("stocks")
        assert disabled is False
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values
        assert "btc.parquet" not in values

    def test_crypto_class_returns_only_crypto(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, disabled = update_asset_class("crypto")
        assert disabled is False
        assert len(options) == 1
        assert options[0]["value"] == "btc.parquet"

    def test_option_label_format(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options, _ = update_asset_class("crypto")
        assert options[0]["label"] == "BTC — Bitcoin (1d)"

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
            options, _ = update_asset_class("stocks")
        assert len(options) == 30


# ---------------------------------------------------------------------------
# update_asset_search
# ---------------------------------------------------------------------------


class TestUpdateAssetSearch:
    def test_no_asset_class_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search(None, None, None)
        assert options == []

    def test_df_none_returns_empty(self):
        with patch.object(app_module, "df", None):
            options = update_asset_search(None, "stocks", None)
        assert options == []

    def test_no_search_returns_top_results_for_class(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search(None, "stocks", None)
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values
        assert "btc.parquet" not in values

    def test_search_by_symbol_case_insensitive(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("aapl", "stocks", None)
        assert len(options) == 1
        assert options[0]["value"] == "aapl.parquet"

    def test_search_by_name_partial_match(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("alphabet", "stocks", None)
        assert len(options) == 1
        assert options[0]["value"] == "googl.parquet"

    def test_search_with_no_matches_returns_empty(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("ZZZZZZ", "stocks", None)
        assert options == []

    def test_current_value_appended_when_not_in_search_results(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("GOOGL", "stocks", "aapl.parquet")
        values = [o["value"] for o in options]
        assert "aapl.parquet" in values
        assert "googl.parquet" in values

    def test_current_value_not_duplicated_when_already_visible(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search(None, "stocks", "aapl.parquet")
        values = [o["value"] for o in options]
        assert values.count("aapl.parquet") == 1

    def test_unknown_current_value_not_appended(self):
        with patch.object(app_module, "df", SAMPLE_DF):
            options = update_asset_search("GOOGL", "stocks", "nonexistent.parquet")
        values = [o["value"] for o in options]
        assert "nonexistent.parquet" not in values

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
            options = update_asset_search("ABC", "stocks", None)
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
        assert "Scattergl" in trace_types

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

    def test_data_older_than_10_years_is_filtered(self):
        now = pd.Timestamp.now()
        mixed_ohlcv = pd.DataFrame(
            {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
             "Close": [1.0, 2.0], "Volume": [100, 200]},
            index=[now - pd.DateOffset(years=11), now - pd.DateOffset(days=1)],
        )
        mixed_ohlcv.index.name = "Date"
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=mixed_ohlcv):
            fig, _ = update_chart("aapl.parquet")
        assert len(fig.data[0].x) == 1

    def test_data_within_10_years_is_kept(self):
        now = pd.Timestamp.now()
        recent_ohlcv = pd.DataFrame(
            {"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0], "Low": [1.0, 2.0, 3.0],
             "Close": [1.0, 2.0, 3.0], "Volume": [100, 200, 300]},
            index=[now - pd.DateOffset(years=9), now - pd.DateOffset(years=5), now - pd.DateOffset(days=1)],
        )
        recent_ohlcv.index.name = "Date"
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=recent_ohlcv):
            fig, _ = update_chart("aapl.parquet")
        assert len(fig.data[0].x) == 3

    def test_timezone_aware_index_does_not_error(self):
        now = pd.Timestamp.now(tz="Europe/Berlin")
        tz_ohlcv = pd.DataFrame(
            {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
             "Close": [1.0, 2.0], "Volume": [100, 200]},
            index=[now - pd.DateOffset(years=11), now - pd.DateOffset(days=1)],
        )
        tz_ohlcv.index.name = "Date"
        with patch.object(app_module, "df", SAMPLE_DF), \
             patch.object(app_module, "base_url", BASE_URL), \
             patch("src.app.pd.read_parquet", return_value=tz_ohlcv):
            fig, headline = update_chart("aapl.parquet")
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 1


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
