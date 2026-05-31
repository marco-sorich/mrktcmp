# Import each plugin module here to trigger its @register side effect.
# Add a new line for each additional strategy plugin.
import src.strategies.dca  # noqa: F401
import src.strategies.riskoff  # noqa: F401
