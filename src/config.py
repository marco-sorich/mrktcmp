# ---------------------------------------------------------------------------
# config.py – Central configuration, logging setup, and master data loading
#
# This module is imported early by every other module in the application.
# It has three responsibilities:
#   1. Configure the Python logging system so all modules share one consistent
#      log format and destination.
#   2. Read runtime settings from environment variables (LOG_LEVEL, BASE_URL,
#      DASH_DEBUG) and from a .env file loaded via python-dotenv.
#   3. Fetch the master asset catalogue (master.parquet) from BASE_URL and
#      expose it as the module-level variables `df` and `assetsClasses` so
#      every callback can read them without re-loading the file on each request.
#
# Other modules access the shared state like this:
#
#   import src.config as _config
#   if _config.df is not None:
#       ...
#
# Accessing through the module object (rather than a bare `from src.config
# import df`) is deliberate: it lets the test suite temporarily replace the
# value with `patch.object(config_module, "df", ...)` and have that change
# seen by all code that reads `_config.df` at call time.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

# logging: Python's built-in structured log system. Preferred over print()
# because it supports log levels (DEBUG/INFO/WARNING/ERROR/CRITICAL) and lets
# you control output format and destination in one place.
import logging

# os: operating-system interface. Used here to read environment variables
# with os.getenv (returns None if the variable is absent, unlike os.environ
# which raises KeyError).
import os

# sys: Python runtime interface. Used to attach log handlers to sys.stdout
# and sys.stderr so log messages flow to the correct output stream.
import sys

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# pandas: tabular data library. DataFrame = a 2-D table with labelled rows
# and columns. Used here to load the master asset catalogue from a Parquet
# file and to filter/sort it.
import pandas as pd

# dotenv: reads key=value pairs from a .env file and injects them into the
# process environment (os.environ). override=True means values in the file
# take precedence over any already-set environment variables.
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env before reading any environment variables
# ---------------------------------------------------------------------------

# load_dotenv must be called before the first os.getenv so that variables
# defined in .env are available when we read LOG_LEVEL and BASE_URL below.
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Read the desired log level from the LOG_LEVEL environment variable.
# os.getenv returns None if the variable is not set; the second argument
# is the default. getattr(logging, 'INFO', logging.DEBUG) converts the
# string 'INFO' to the integer constant logging.INFO (= 20). If the string
# is unrecognised, it falls back to logging.DEBUG (= 10) so we still see
# all output.
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.DEBUG)

# Create a handler that writes to standard output (the terminal's normal
# output stream). This is what you see in the console when you run the app.
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(_log_level)

# Add a filter so that only DEBUG and INFO messages go to stdout.
# WARNING, ERROR, and CRITICAL messages are more serious and are sent to
# stderr (the error output stream) separately below. This matters in
# production: process supervisors like gunicorn and systemd capture stdout
# and stderr separately, so operators can filter error logs without noise.
# A filter is a callable that returns True to keep the record, False to drop.
# r.levelno < logging.WARNING means: keep only records below WARNING level.
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

# Create a second handler for stderr (standard error). Only WARNING and above
# are routed here. This ensures serious messages always reach operators even
# if stdout is piped away or suppressed.
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

# Apply both handlers globally. logging.basicConfig configures the root
# logger (the parent of all named loggers). level sets the minimum level
# that will pass through the root logger before reaching any handler.
logging.basicConfig(level=_log_level, handlers=[_stdout_handler, _stderr_handler])

# Create a named logger for this module. __name__ resolves to 'src.config'
# at runtime. Using a named logger makes it easy to identify which module
# produced each log message.
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load master asset catalogue from BASE_URL
# ---------------------------------------------------------------------------

# BASE_URL is an environment variable that points to the root URL (or path)
# where the parquet data files are stored. Example:
#   BASE_URL=https://example.com/data   → files at BASE_URL/master.parquet
#   BASE_URL=/mnt/data                  → files at /mnt/data/master.parquet
base_url = os.getenv("BASE_URL")

# These two variables hold the loaded data. They are module-level (global)
# so every callback can read them without passing them as arguments.
# assetsClasses: list of distinct asset class strings, e.g. ['stocks','crypto']
# df:            the master metadata DataFrame – one row per asset, columns
#                include: asset_class, symbol, name, filename, exchange,
#                country, interval.
assetsClasses = []
df = None

if not base_url or base_url.strip() == "":
    # Log at CRITICAL because the app cannot function without a data source.
    log.critical("BASE_URL environment variable is not set.")
else:
    try:
        # master.parquet is the catalogue file: one row per available asset.
        # pd.read_parquet reads the binary Parquet format directly into a
        # DataFrame; it is much faster than CSV for large tables.
        df = pd.read_parquet(f"{base_url}/master.parquet")

        # Sort the catalogue so dropdowns are presented alphabetically by
        # asset class, then symbol, then exchange. inplace=True modifies the
        # DataFrame in place (no copy). ignore_index=True resets the row
        # numbers to 0, 1, 2, … after sorting.
        df.sort_values(['asset_class', 'symbol', 'exchange'], inplace=True, ignore_index=True)

        # .unique() returns an array of distinct values in 'asset_class'.
        # .tolist() converts it from a numpy array to a plain Python list,
        # which is what Dash RadioItems / Dropdown options expect.
        assetsClasses = df['asset_class'].unique().tolist()
        log.info(f"Data loaded. AssetClasses: {assetsClasses}")
    except Exception:
        # log.exception automatically includes the full traceback so we can
        # diagnose the problem without adding extra debug code.
        log.exception("Failed to load master.parquet from BASE_URL")
