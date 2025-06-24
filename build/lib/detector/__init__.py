"""TokenSlasher detector package.

Core public API lives here so external users can::

    import tokenslasher as ts
    ts.SENTINEL
    ts.__version__
"""

from importlib.metadata import version as _pkg_version


# Semantic version of the installed package
try:
    __version__: str = _pkg_version("tokenslasher")
except Exception:  # pragma: no cover – local dev path
    __version__ = "1.0.0"


# Special token inserted between documents so shingles never span boundaries
SENTINEL: str = "<doc>"

__all__ = [
    "SENTINEL",
    "__version__",
] 