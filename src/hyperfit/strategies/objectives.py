"""
Objective functions for parameter optimization.

This module provides different objective functions for fitting material
parameters, including absolute error, relative error, and specialized
objectives for Mullins effect fitting.
"""

from typing import Dict, Any, Callable, Tuple, Optional
import numpy as np

from ..models.base import HyperelasticModel
from ..data import ExperimentalData
from ..exceptions import FittingError


def create_objective_function(objective_type: str,
                             model: HyperelasticModel,
                             experimental_data: Dict[str, ExperimentalData],
                             config: Dict[str, Any]) -> Callable[[np.ndarray], float]:
    """
    Create objective function for optimization.
    
    Args:
        objective_type: Type of objective ('absolute_error', 'relative_error', 'stress', 'eta')
        model: Material model instance
        experimental_data: Dictionary of experimental data
        config: Objective function configuration
        
    Returns:
        Objective function that takes parameter vector and returns cost
        
    Raises:
        FittingError: If objective type is unsupported
    """
    if objective_type == 'absolute_error':
        return create_absolute_error_objective(model, experimental_data, config)
    elif objective_type == 'relative_error':
        return create_relative_error_objective(model, experimental_data, config)
    elif objective_type == 'stress':
        return create_stress_objective(model, experimental_data, config)
    elif objective_type == 'eta':
        return create_eta_objective(model, experimental_data, config)
    else:
        raise FittingError(f"Unknown objective function type: {objective_type}")


def create_absolute_error_objective(model: HyperelasticModel,
                                   experimental_data: Dict[str, ExperimentalData],
                                   config: Dict[str, Any]) -> Callable[[np.ndarray], float]:
    """
    Create absolute error objective function.
    
    Minimizes: Σ (σ_exp - σ_model)²
    """
    def objective(params: np.ndarray) -> float:
        try:
            param_dict = model.format_parameters(params)
            total_error = 0.0
            
            for loading_type, exp_data in experimental_data.items():
                if loading_type == 'volumetric':
                    # Handle volumetric data
                    if hasattr(model, 'calculate_pressure'):
                        predicted = model.calculate_pressure(exp_data.x_data, param_dict)
                        experimental = exp_data.y_data
                    else:
                        continue
                else:
                    # Handle mechanical data
                    predicted = model.calculate_stress(exp_data.x_data, param_dict, loading_type)
                    experimental = exp_data.y_data
                
                # Calculate squared errors
                residuals = experimental - predicted
                squared_errors = residuals ** 2
                
                # Apply weights if specified
                weights = config.get('weights', {}).get(loading_type, 1.0)
                if isinstance(weights, (list, np.ndarray)):
                    if len(weights) != len(squared_errors):
                        weights = np.ones_like(squared_errors)
                    total_error += np.sum(weights * squared_errors)
                else:
                    total_error += weights * np.sum(squared_errors)
            
            return float(total_error)
            
        except Exception as e:
            # Return large penalty for invalid parameters
            return 1e12
    
    return objective


def create_relative_error_objective(model: HyperelasticModel,
                                   experimental_data: Dict[str, ExperimentalData],
                                   config: Dict[str, Any]) -> Callable[[np.ndarray], float]:
    """
    Create relative error objective function.
    
    Minimizes: Σ ((σ_exp - σ_model) / (σ_exp + ε))²
    """
    epsilon = config.get('epsilon', 1e-6)
    
    def objective(params: np.ndarray) -> float:
        try:
            param_dict = model.format_parameters(params)
            total_error = 0.0
            
            for loading_type, exp_data in experimental_data.items():
                if loading_type == 'volumetric':
                    # Handle volumetric data
                    if hasattr(model, 'calculate_pressure'):
                        predicted = model.calculate_pressure(exp_data.x_data, param_dict)
                        experimental = exp_data.y_data
                    else:
                        continue
                else:
                    # Handle mechanical data
                    predicted = model.calculate_stress(exp_data.x_data, param_dict, loading_type)
                    experimental = exp_data.y_data
                
                # Calculate relative errors
                residuals = experimental - predicted
                denominators = np.abs(experimental) + epsilon
                relative_errors = (residuals / denominators) ** 2
                
                # Apply weights if specified
                weights = config.get('weights', {}).get(loading_type, 1.0)
                if isinstance(weights, (list, np.ndarray)):
                    if len(weights) != len(relative_errors):
                        weights = np.ones_like(relative_errors)
                    total_error += np.sum(weights * relative_errors)
                else:
                    total_error += weights * np.sum(relative_errors)
            
            return float(total_error)
            
        except Exception as e:
            # Return large penalty for invalid parameters
            return 1e12
    
    return objective


def create_stress_objective(model: HyperelasticModel,
                           experimental_data: Dict[str, ExperimentalData],
                           config: Dict[str, Any]) -> Callable[[np.ndarray], float]:
    """
    Create stress-based objective for Mullins effect fitting.
    
    This objective minimizes the difference between predicted and experimental
    damaged stresses, accounting for the Mullins effect.
    """
    mullins_params = config.get('mullins_parameters', {})
    
    def objective(params: np.ndarray) -> float:
        try:
            # Split parameters between hyperelastic and Mullins
            if 'parameter_split' in config:
                split_idx = config['parameter_split']
                hyperelastic_params = params[:split_idx]
                mullins_param_values = params[split_idx:]
            else:
                # Assume all parameters are hyperelastic
                hyperelastic_params = params
                mullins_param_values = [mullins_params.get(p, 1.0) 
                                      for p in ['r', 'm', 'beta']]
            
            # Format hyperelastic parameters
            hyperelastic_dict = model.format_parameters(hyperelastic_params)
            
            # Format Mullins parameters
            mullins_dict = dict(zip(['r', 'm', 'beta'], mullins_param_values))
            
            total_error = 0.0
            
            for loading_type, exp_data in experimental_data.items():
                if loading_type == 'volumetric':
                    continue  # Skip volumetric data for stress objective
                
                strains = exp_data.x_data
                damaged_stresses_exp = exp_data.y_data
                
                # Calculate undamaged stresses
                undamaged_stresses = model.calculate_stress(strains, hyperelastic_dict, loading_type)
                
                # Calculate strain energies and apply Mullins effect
                strain_energies = model.calculate_strain_energy(strains, hyperelastic_dict, loading_type)
                max_strain_energies = np.maximum.accumulate(strain_energies)
                
                # Apply Mullins damage
                from ..models.mullins import apply_mullins_damage
                damaged_stresses_pred = apply_mullins_damage(
                    undamaged_stresses, strain_energies, max_strain_energies, mullins_dict
                )
                
                # Calculate error
                residuals = damaged_stresses_exp - damaged_stresses_pred
                squared_errors = residuals ** 2
                
                weights = config.get('weights', {}).get(loading_type, 1.0)
                if isinstance(weights, (list, np.ndarray)):
                    total_error += np.sum(weights * squared_errors)
                else:
                    total_error += weights * np.sum(squared_errors)
            
            return float(total_error)
            
        except Exception as e:
            return 1e12
    
    return objective


def create_eta_objective(model: HyperelasticModel,
                        experimental_data: Dict[str, ExperimentalData],
                        config: Dict[str, Any]) -> Callable[[np.ndarray], float]:
    """
    Create eta (damage function) objective for Mullins effect fitting.
    
    This objective minimizes the difference between predicted and experimental
    damage function values η = σ_damaged / σ_undamaged.
    """
    def objective(params: np.ndarray) -> float:
        try:
            # Assume params are Mullins parameters [r, m, beta]
            mullins_dict = dict(zip(['r', 'm', 'beta'], params))
            
            # Get hyperelastic parameters from config
            hyperelastic_dict = config.get('hyperelastic_parameters', {})
            
            total_error = 0.0
            
            for loading_type, exp_data in experimental_data.items():
                if loading_type == 'volumetric':
                    continue
                
                strains = exp_data.x_data
                damaged_stresses_exp = exp_data.y_data
                
                # Calculate undamaged stresses
                undamaged_stresses = model.calculate_stress(strains, hyperelastic_dict, loading_type)
                
                # Calculate experimental eta values
                eta_exp = damaged_stresses_exp / undamaged_stresses
                eta_exp = np.clip(eta_exp, 1e-9, 1.0)  # Clamp to physical range
                
                # Calculate strain energies
                strain_energies = model.calculate_strain_energy(strains, hyperelastic_dict, loading_type)
                max_strain_energies = np.maximum.accumulate(strain_energies)
                
                # Calculate predicted eta values
                from ..models.mullins import MullinsEffectModel
                mullins_model = MullinsEffectModel()
                eta_pred = mullins_model.calculate_damage_function(
                    strain_energies, max_strain_energies, mullins_dict
                )
                
                # Calculate error in eta
                residuals = eta_exp - eta_pred
                squared_errors = residuals ** 2
                
                weights = config.get('weights', {}).get(loading_type, 1.0)
                if isinstance(weights, (list, np.ndarray)):
                    total_error += np.sum(weights * squared_errors)
                else:
                    total_error += weights * np.sum(squared_errors)
            
            return float(total_error)
            
        except Exception as e:
            return 1e12
    
    return objective


def create_jacobian_function(objective_type: str,
                           model: HyperelasticModel,
                           experimental_data: Dict[str, ExperimentalData],
                           config: Dict[str, Any]) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    Create analytical Jacobian function if available.
    
    Args:
        objective_type: Type of objective function
        model: Material model instance
        experimental_data: Dictionary of experimental data
        config: Configuration dictionary
        
    Returns:
        Jacobian function or None if not available
    """
    # For now, return None to use numerical differentiation
    # In the future, we can implement analytical Jacobians for specific models
    return None


def create_weighted_objective(base_objective: Callable[[np.ndarray], float],
                            weights: Dict[str, float]) -> Callable[[np.ndarray], float]:
    """
    Create weighted combination of multiple objectives.
    
    Args:
        base_objective: Base objective function
        weights: Weights for different components
        
    Returns:
        Weighted objective function
    """
    def weighted_objective(params: np.ndarray) -> float:
        return base_objective(params)
    
    return weighted_objective
