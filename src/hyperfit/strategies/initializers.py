"""
Initial guess strategies for parameter optimization.

This module provides different strategies for generating initial parameter
guesses, including Linear Least Squares (LLS) and heuristic methods.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from ..models.base import HyperelasticModel
from ..data import ExperimentalData
from ..exceptions import FittingError


def generate_initial_guess(strategy: str,
                          model: HyperelasticModel,
                          experimental_data: Dict[str, ExperimentalData],
                          config: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[Tuple[float, float]]]:
    """
    Generate initial parameter guess using specified strategy.
    
    Args:
        strategy: Initial guess strategy ('lls', 'heuristic')
        model: Material model instance
        experimental_data: Dictionary of experimental data
        config: Strategy configuration
        
    Returns:
        Tuple of (initial_parameters, parameter_names, parameter_bounds)
        
    Raises:
        FittingError: If initial guess generation fails
    """
    if strategy == 'lls':
        return generate_lls_guess(model, experimental_data, config)
    elif strategy == 'heuristic':
        return generate_heuristic_guess(model, experimental_data, config)
    else:
        raise FittingError(f"Unknown initial guess strategy: {strategy}")


def generate_lls_guess(model: HyperelasticModel,
                      experimental_data: Dict[str, ExperimentalData],
                      config: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[Tuple[float, float]]]:
    """
    Generate initial guess using Linear Least Squares method.
    
    This method fixes some parameters (like α in Ogden model) and solves
    for others using linear least squares. It's generally more stable
    than random initialization.
    """
    try:
        # Try model-specific LLS method first
        if hasattr(model, 'get_initial_guess_lls'):
            # Convert ExperimentalData objects to format expected by model
            model_data = {}
            for loading_type, exp_data in experimental_data.items():
                model_data[loading_type] = exp_data
            
            # Get LLS-specific configuration
            alpha_guesses = config.get('alpha_guesses', None)
            if hasattr(model, 'get_initial_guess_lls'):
                if alpha_guesses is not None:
                    params = model.get_initial_guess_lls(model_data, alpha_guesses)
                else:
                    params = model.get_initial_guess_lls(model_data)
            else:
                raise FittingError(f"Model {type(model).__name__} does not support LLS initialization")
        
        else:
            # Fall back to generic LLS approach
            params = _generic_lls_guess(model, experimental_data, config)
        
        param_names = model.get_parameter_names()
        bounds = model.get_parameter_bounds()
        
        return params, param_names, bounds
        
    except Exception as e:
        raise FittingError(f"LLS initial guess failed: {e}") from e


def generate_heuristic_guess(model: HyperelasticModel,
                           experimental_data: Dict[str, ExperimentalData],
                           config: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[Tuple[float, float]]]:
    """
    Generate initial guess using heuristic (random) method.
    
    This method generates random parameter values within reasonable bounds,
    with constraints to ensure physical stability.
    """
    try:
        # Try model-specific heuristic method first
        if hasattr(model, 'get_initial_guess_heuristic'):
            # Convert ExperimentalData objects to format expected by model
            model_data = {}
            for loading_type, exp_data in experimental_data.items():
                model_data[loading_type] = exp_data
            
            params = model.get_initial_guess_heuristic(model_data)
        else:
            # Fall back to generic heuristic approach
            params = _generic_heuristic_guess(model, experimental_data, config)
        
        param_names = model.get_parameter_names()
        bounds = model.get_parameter_bounds()
        
        return params, param_names, bounds
        
    except Exception as e:
        raise FittingError(f"Heuristic initial guess failed: {e}") from e


def _generic_lls_guess(model: HyperelasticModel,
                      experimental_data: Dict[str, ExperimentalData],
                      config: Dict[str, Any]) -> np.ndarray:
    """
    Generic Linear Least Squares initial guess.
    
    This is a simplified approach that works for models where some parameters
    can be determined by linear regression.
    """
    # For now, fall back to heuristic if no model-specific method
    return _generic_heuristic_guess(model, experimental_data, config)


def _generic_heuristic_guess(model: HyperelasticModel,
                           experimental_data: Dict[str, ExperimentalData],
                           config: Dict[str, Any]) -> np.ndarray:
    """
    Generic heuristic initial guess.
    
    Generates random parameters within bounds, with some physics-based scaling.
    """
    bounds = model.get_parameter_bounds()
    num_params = len(bounds)
    
    # Estimate material stiffness scale from data
    stiffness_scale = _estimate_stiffness_scale(experimental_data)
    
    # Generate random parameters within bounds
    np.random.seed(config.get('random_seed', 12345))
    
    params = np.zeros(num_params)
    for i, (lower, upper) in enumerate(bounds):
        if upper == np.inf:
            upper = stiffness_scale * 10
        if lower == -np.inf:
            lower = -stiffness_scale * 10
        
        # Scale bounds based on estimated stiffness
        if upper > 1e3:
            upper = min(upper, stiffness_scale * 10)
        if lower < -1e3:
            lower = max(lower, -stiffness_scale * 10)
        
        params[i] = np.random.uniform(lower, upper)
    
    return params


def _estimate_stiffness_scale(experimental_data: Dict[str, ExperimentalData]) -> float:
    """
    Estimate material stiffness scale from experimental data.
    
    This provides a rough estimate of the material's stiffness to help
    scale the initial parameter guesses appropriately.
    """
    total_slope = 0.0
    total_points = 0
    
    for loading_type, exp_data in experimental_data.items():
        if loading_type == 'volumetric':
            # For volumetric data, use pressure/volume_change
            volume_ratios = exp_data.x_data
            pressures = exp_data.y_data
            
            for j, p in zip(volume_ratios, pressures):
                volume_change = abs(j - 1.0)
                if volume_change > 1e-6:
                    total_slope += abs(p) / volume_change
                    total_points += 1
        else:
            # For mechanical data, use stress/strain
            strains = exp_data.x_data
            stresses = exp_data.y_data
            
            for eps, sigma in zip(strains, stresses):
                if abs(eps) > 1e-6:
                    total_slope += abs(sigma) / abs(eps)
                    total_points += 1
    
    if total_points > 0:
        avg_slope = total_slope / total_points
        return max(1.0, avg_slope)
    else:
        return 1e6  # Default stiffness scale


def validate_initial_guess(params: np.ndarray,
                         bounds: List[Tuple[float, float]],
                         model: HyperelasticModel) -> bool:
    """
    Validate that initial guess satisfies bounds and stability constraints.
    
    Args:
        params: Parameter vector
        bounds: Parameter bounds
        model: Material model
        
    Returns:
        True if parameters are valid
    """
    # Check bounds
    for i, (param, (lower, upper)) in enumerate(zip(params, bounds)):
        if param < lower or param > upper:
            return False
    
    # Check model-specific stability constraints
    try:
        # For models with stability constraints (e.g., G0 > 0 for Ogden)
        param_dict = model.format_parameters(params)
        
        # Try a simple stress calculation to check for numerical issues
        test_strain = 0.1
        test_stress = model.calculate_stress(test_strain, param_dict, 'uniaxial')
        
        if not np.isfinite(test_stress):
            return False
            
    except Exception:
        return False
    
    return True


def refine_initial_guess(params: np.ndarray,
                        bounds: List[Tuple[float, float]],
                        model: HyperelasticModel,
                        max_attempts: int = 10) -> np.ndarray:
    """
    Refine initial guess if it doesn't satisfy constraints.
    
    Args:
        params: Initial parameter vector
        bounds: Parameter bounds
        model: Material model
        max_attempts: Maximum refinement attempts
        
    Returns:
        Refined parameter vector
        
    Raises:
        FittingError: If refinement fails
    """
    for attempt in range(max_attempts):
        if validate_initial_guess(params, bounds, model):
            return params
        
        # Try to fix parameters
        for i, (param, (lower, upper)) in enumerate(zip(params, bounds)):
            if param < lower:
                params[i] = lower + 0.01 * (upper - lower)
            elif param > upper:
                params[i] = upper - 0.01 * (upper - lower)
        
        # Add small random perturbation
        perturbation = 0.01 * np.random.randn(len(params))
        params = params + perturbation
        
        # Clip to bounds again
        for i, (lower, upper) in enumerate(bounds):
            params[i] = np.clip(params[i], lower, upper)
    
    raise FittingError("Failed to generate valid initial guess after maximum attempts")
