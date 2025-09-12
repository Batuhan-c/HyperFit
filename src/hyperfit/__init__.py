"""
HyperFit: A comprehensive Python library for fitting hyperelastic and Mullins effect material models.

This library provides a configuration-driven approach to fitting various hyperelastic material models
including Ogden, Polynomial, and Reduced Polynomial models, with optional Mullins damage effects.

The library is designed for both direct Python usage and C++ integration via Pybind11.
"""

from .api import fit
from .exceptions import HyperFitError, ConfigurationError, DataError, FittingError

__version__ = "0.1.0"
__author__ = "HyperFit Contributors"

# Public API exports
__all__ = [
    "fit",
    "HyperFitError", 
    "ConfigurationError",
    "DataError", 
    "FittingError",
]
