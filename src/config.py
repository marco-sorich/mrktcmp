import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.DEBUG)

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(_log_level)
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(level=_log_level, handlers=[_stdout_handler, _stderr_handler])

log = logging.getLogger(__name__)

base_url = os.getenv("BASE_URL")
assetsClasses = []
df = None

if not base_url or base_url.strip() == "":
    log.critical("BASE_URL environment variable is not set.")
else:
    try:
        df = pd.read_parquet(f"{base_url}/master.parquet")
        df.sort_values(['asset_class', 'symbol', 'exchange'], inplace=True, ignore_index=True)
        assetsClasses = df['asset_class'].unique().tolist()
        log.info(f"Data loaded. AssetClasses: {assetsClasses}")
    except Exception:
        log.exception("Failed to load master.parquet from BASE_URL")
