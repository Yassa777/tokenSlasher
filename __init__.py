"""TokenSlasher top-level package.

This *namespace* init also exposes ``tokenSlasher.detector`` under the simpler
alias ``detector`` so that unit tests can just ``import detector`` when the
project root isn't on ``PYTHONPATH``.
"""
from importlib import import_module
import sys

# Lazily create the alias only if it doesn't exist already.
if 'detector' not in sys.modules:
    sys.modules['detector'] = import_module(__name__ + '.detector')

__all__ = ['detector'] 