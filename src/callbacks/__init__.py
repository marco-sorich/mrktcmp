# ---------------------------------------------------------------------------
# callbacks/__init__.py – Callback registration package
#
# Dash callbacks are registered by the @callback decorator at *import time*:
# the moment Python executes the decorated function definition, Dash records
# the Input/Output/State wiring. That means a callback module must be
# imported before app.run() is called, even if no name from that module is
# explicitly used anywhere else.
#
# This __init__.py serves as the single import point: app.py does
#
#   import src.callbacks
#
# and this file imports each callback sub-module in turn, causing all
# @callback decorators to execute and register their callbacks with Dash.
# ---------------------------------------------------------------------------

# Importing backtesting triggers all @callback decorators defined in that
# module. The noqa comment suppresses the "imported but unused" linter
# warning – the side effect (callback registration) is the purpose.
from src.callbacks import backtesting  # noqa: F401
