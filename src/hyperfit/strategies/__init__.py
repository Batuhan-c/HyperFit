"""
Strategy modules for the HyperFit library.

This package contains configurable strategies for different aspects of the
fitting process, including initial guess generation and objective functions.
"""

from . import initializers
from . import objectives

__all__ = ['initializers', 'objectives']
