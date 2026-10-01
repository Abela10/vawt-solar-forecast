"""Optional numba JIT.

The 10 Hz physics kernel is a tight scalar loop. With numba a 30-day run takes seconds (also on a
Raspberry Pi 5, aarch64 wheels exist); without numba the same code runs as plain Python, slower but
identical in behaviour.
"""
from __future__ import annotations

try:  # pragma: no cover - import guard
    from numba import njit as _numba_njit

    HAVE_NUMBA = True

    def njit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", False)
        return _numba_njit(*args, **kwargs)

except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
