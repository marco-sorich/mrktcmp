# ---------------------------------------------------------------------------
# strategies/base.py – plugin contract for backtesting strategies
#
# Every strategy plugin must:
#   1. Subclass BacktestStrategy.
#   2. Implement all abstract class- and instance-methods.
#   3. Decorate itself with @register (from registry.py) so it is discoverable.
# ---------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class ConfigParam:
    """Declares one user-configurable parameter for a strategy.

    The GUI uses these declarations to render the correct input widget and
    to pre-fill it with *default* so the user never has to enter a value
    manually (all parameters are always optional from the user's perspective).

    Fields
    ------
    key           – Python identifier used as the key in the params dict
                    passed to BacktestStrategy.run().
    label         – Human-readable label shown next to the input widget.
    type          – Determines the widget type:
                    'int'    → integer number input
                    'float'  → decimal number input
                    'select' → dropdown of predefined string choices
    default       – Value pre-filled in the GUI; always required so the
                    strategy can run without any user interaction.
    min_value     – Minimum allowed value (required for 'int' and 'float').
                    The GUI enforces this so plugins receive only valid values.
    max_value     – Maximum allowed value (required for 'int' and 'float').
                    The GUI enforces this so plugins receive only valid values.
    options       – Exhaustive list of allowed string values (required for
                    'select'). The GUI renders these as dropdown choices.
    """

    key: str
    label: str
    type: Literal['int', 'float', 'select']
    default: int | float | str  # always required; GUI pre-fills inputs with this value
    min_value: int | float | None = None  # required when type in ('int', 'float')
    max_value: int | float | None = None  # required when type in ('int', 'float')
    options: list[str] = field(default_factory=list)  # required when type == 'select'


class BacktestStrategy(ABC):
    """Abstract base class that every strategy plugin must implement.

    Subclasses are registered via the @register decorator from registry.py and
    discovered at startup when src.strategies is imported.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Short, unique identifier shown in the strategy selector (e.g. 'DCA')."""
        ...

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """One-sentence explanation of the strategy shown as a tooltip or subtitle."""
        ...

    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> list[ConfigParam]:
        """Declare the parameters the user can configure for this strategy.

        Every returned ConfigParam must have a valid *default* so the strategy
        can run without any user input.  The GUI enforces min/max and options
        constraints, so run() can trust that params values are always valid.
        """
        ...

    @abstractmethod
    def run(
        self,
        base_url: str,
        filenames: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        df_meta: pd.DataFrame,
        params: dict[str, int | float | str],
    ) -> tuple[pd.Series | None, dict[str, str] | None]:
        """Execute the backtest for one basket and return results.

        Parameters
        ----------
        base_url   – root URL/path where the per-asset parquet files are hosted.
        filenames  – parquet filenames for every asset in this basket.
        start_date – first month-end date to include (inclusive).
        end_date   – last month-end date to include (inclusive).
        df_meta    – master metadata table (symbol, name, filename, …).
        params     – user-supplied config values, keyed by ConfigParam.key.
                     May be empty; the plugin must fall back to each param's
                     *default* for any missing key.

        Returns
        -------
        (portfolio_series, metrics_dict) on success, or (None, None) on failure.
        portfolio_series – monthly portfolio value as a pandas Series.
        metrics_dict     – exactly the same 11 keys as compute_metrics() returns.
        """
        ...
