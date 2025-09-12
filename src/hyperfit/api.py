"""
Public API for the HyperFit library.

This module provides the main entry point for all users through the `fit` function.
The function accepts a configuration dictionary and returns fitting results.
"""

from typing import Dict, Any
import traceback

from .config import FittingConfiguration
from .fitter import Fitter
from .data import prepare_experimental_data
from .exceptions import HyperFitError


def fit(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs a hyperelastic material model fit based on the provided configuration.

    This function is the primary entry point for the HyperFit library. It provides
    a clean, stable interface that relies entirely on the configuration dictionary
    for specifying the fitting strategy, data, and model parameters.

    Args:
        config (dict): A dictionary specifying the entire fitting strategy.
                      Must contain at minimum:
                      - 'model': str, the material model type ('ogden', 'polynomial', 
                                'reduced_polynomial')
                      - 'experimental_data': dict, the experimental data
                      - 'fitting_strategy': dict, the fitting strategy parameters

    Returns:
        dict: A dictionary containing the fit results with the following structure:
              - 'success': bool, whether the fit was successful
              - 'parameters': dict, fitted model parameters (if successful)
              - 'diagnostics': dict, fitting diagnostics and quality metrics
              - 'message': str, status message
              - 'error': str, error message (if unsuccessful)

    Example:
        >>> import hyperfit
        >>> import numpy as np
        >>> 
        >>> config = {
        ...     "model": "reduced_polynomial",
        ...     "model_order": 3,
        ...     "experimental_data": {
        ...         "uniaxial": {
        ...             "strain": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        ...             "stress": np.array([100, 180, 240, 280, 300])
        ...         }
        ...     },
        ...     "fitting_strategy": {
        ...         "initial_guess": {"method": "lls"},
        ...         "optimizer": {"methods": ["L-BFGS-B"]},
        ...         "objective_function": {"type": "relative_error"}
        ...     }
        ... }
        >>> 
        >>> result = hyperfit.fit(config)
        >>> if result['success']:
        ...     print("Fitted parameters:", result['parameters'])
    """
    try:
        # 1. Validate and structure the configuration
        fit_config = FittingConfiguration(config)

        # 2. Load and preprocess data
        prepared_data = prepare_experimental_data(fit_config.data)

        # 3. Create and run the fitter
        fitter = Fitter(fit_config)
        result = fitter.execute(prepared_data)

        return result.to_dict()

    except HyperFitError as e:
        # Return a standardized error dictionary for known HyperFit errors
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "parameters": None,
            "diagnostics": None,
            "message": f"HyperFit error: {str(e)}"
        }
    
    except Exception as e:
        # Handle unexpected errors gracefully
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "parameters": None,
            "diagnostics": None,
            "message": f"Unexpected error during fitting: {str(e)}",
            "traceback": traceback.format_exc()
        }
