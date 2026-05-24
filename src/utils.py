# ---------------------------------------------------------------------------
# utils.py – Shared utility functions used across the application
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

# functools.wraps: preserves the original function's name, docstring, and
# attributes when a decorator wraps it. Without this, the decorator would
# shadow the function's identity (e.g. func.__name__ would return 'wrapper').
import functools

# time: provides time.time() which returns the current Unix timestamp as a
# float (seconds since 1970-01-01 00:00:00 UTC). Used here to measure how
# long callbacks take to execute.
import time

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

# config is imported as a module object (not individual names) so that the
# logger reference is always resolved at call time. This means any changes
# made to config.log during testing are automatically visible here.
import src.config as _config


# ---------------------------------------------------------------------------
# Decorator: log the execution time of every callback
# ---------------------------------------------------------------------------

def log_time(func):
    """Decorator that logs how long a callback function takes to run.

    A decorator is a function that *wraps* another function to add behaviour
    before and/or after it. The @log_time syntax is shorthand for:
        func = log_time(func)

    Apply it directly above a callback function definition:

        @callback(...)
        @log_time
        def my_callback(...):
            ...

    Note: @log_time must be placed *below* @callback so that Dash registers
    the original function name, not the wrapper.

    Parameters
    ----------
    func : callable – the callback function to wrap.

    Returns
    -------
    callable – a wrapper that calls func, measures the elapsed wall-clock
    time, logs it at DEBUG level, and returns func's result unchanged.
    """
    # functools.wraps(func) copies the original function's __name__, __doc__,
    # etc. onto the wrapper so tools like debuggers and Dash still see the
    # original function's name.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record the wall-clock time just before calling the real function.
        t0 = time.time()
        # *args / **kwargs forwards all positional and keyword arguments to
        # the wrapped function unchanged, as if it were called directly.
        result = func(*args, **kwargs)
        # Log the elapsed time. func.__name__ gives the original callback's
        # name thanks to @functools.wraps.
        _config.log.debug(f'{func.__name__} callback time: {(time.time() - t0)*1000:,.2f}ms')
        return result
    return wrapper
