"""
This is the rlrom module.

It contains methods for robust online monitoring of reinforcement learning models.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("rlrom")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
