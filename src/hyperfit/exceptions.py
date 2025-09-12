"""
Custom exceptions for the HyperFit library.

This module defines all custom exception types used throughout the library
to provide clear error reporting and handling.
"""


class HyperFitError(Exception):
    """Base exception class for all HyperFit-related errors."""
    pass


class ConfigurationError(HyperFitError):
    """Raised when there are issues with the fitting configuration."""
    pass


class DataError(HyperFitError):
    """Raised when there are issues with experimental data."""
    pass


class FittingError(HyperFitError):
    """Raised when the optimization process fails."""
    pass


class ModelError(HyperFitError):
    """Raised when there are issues with material model calculations."""
    pass


class ConvergenceError(FittingError):
    """Raised when the optimization fails to converge."""
    pass
