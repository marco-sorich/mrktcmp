# Import each plugin module here to trigger its @register side effect.
# Add a new line for each additional strategy plugin.  The first import becomes
# the default GUI selection (the dropdown lists strategies in registration
# order), so Buy & Hold is imported first to be the default.
import src.strategies.lumpsum  # noqa: F401
import src.strategies.dca  # noqa: F401
import src.strategies.riskoff  # noqa: F401
import src.strategies.summergap  # noqa: F401
