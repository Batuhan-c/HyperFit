"""
Mullins effect model implementation.

This module implements the Mullins damage model based on the algorithms
from the original fit_mullins_*.py scripts.
"""

from typing import Dict, Any, Union, List
import numpy as np
from scipy.special import erf

from .base import MullinsModel
from ..exceptions import ModelError


class MullinsEffectModel(MullinsModel):
    """
    Mullins effect (stress softening) model.
    
    The damage function is:
    η = 1 - erf((W_max - W) / (m + β * W_max)) / r
    
    Where:
    - η is the damage function (0 < η ≤ 1)
    - W is the current strain energy density
    - W_max is the maximum strain energy in loading history
    - r, m, β are the Mullins parameters
    """
    
    def __init__(self):
        """Initialize Mullins effect model."""
        pass
    
    def calculate_damage_function(self, 
                                strain_energy: Union[float, np.ndarray],
                                max_strain_energy: Union[float, np.ndarray],
                                parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Calculate damage function η for Mullins effect.
        
        Implementation based on fit_mullins_*.py algorithms.
        """
        strain_energy = np.asarray(strain_energy)
        max_strain_energy = np.asarray(max_strain_energy)
        
        r = parameters['r']
        m = parameters['m']
        beta = parameters['beta']
        
        # Validate parameters
        if r <= 0 or m <= 0 or beta <= 0:
            raise ModelError("Mullins parameters r, m, beta must be positive")
        
        # Calculate denominator with safety check
        denominator = m + beta * max_strain_energy
        safe_denominator = np.where(denominator == 0, 1e-9, denominator)
        
        # Calculate erf argument
        term_arg = (max_strain_energy - strain_energy) / safe_denominator
        
        # Calculate erf term
        term = erf(term_arg)
        
        # Calculate damage function
        eta = 1.0 - term / r
        
        # Clamp to physically meaningful range (0, 1]
        eta = np.clip(eta, 1e-9, 1.0)
        
        return eta.item() if eta.size == 1 else eta
    
    def calculate_damage_function_variant_erf_arg_div_r(self,
                                                       strain_energy: Union[float, np.ndarray],
                                                       max_strain_energy: Union[float, np.ndarray],
                                                       parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Variant: η = 1 - erf(((W_max - W)/(m + β*W_max)) / r)
        
        Divide the erf argument by r instead of dividing erf by r.
        """
        strain_energy = np.asarray(strain_energy)
        max_strain_energy = np.asarray(max_strain_energy)
        
        r = parameters['r']
        m = parameters['m']
        beta = parameters['beta']
        
        # Calculate denominator with safety check
        denominator = m + beta * max_strain_energy
        safe_denominator = np.where(denominator == 0, 1e-9, denominator)
        
        # Calculate erf argument divided by r
        term_arg = (max_strain_energy - strain_energy) / safe_denominator
        safe_r = r if r != 0 else 1e-9
        term = erf(term_arg / safe_r)
        
        # Calculate damage function
        eta = 1.0 - term
        
        # Clamp to physically meaningful range
        eta = np.clip(eta, 1e-9, 1.0)
        
        return eta.item() if eta.size == 1 else eta
    
    def calculate_damage_function_variant_denom_W(self,
                                                 strain_energy: Union[float, np.ndarray],
                                                 max_strain_energy: Union[float, np.ndarray],
                                                 parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Variant: Use denominator = m + β * W_history (instead of W_max_history).
        
        This tests sensitivity to whether current or max energy is used in denominator.
        """
        strain_energy = np.asarray(strain_energy)
        max_strain_energy = np.asarray(max_strain_energy)
        
        r = parameters['r']
        m = parameters['m']
        beta = parameters['beta']
        
        # Use current strain energy in denominator
        denominator = m + beta * strain_energy
        safe_denominator = np.where(denominator == 0, 1e-9, denominator)
        
        # Calculate erf argument
        term_arg = (max_strain_energy - strain_energy) / safe_denominator
        term = erf(term_arg)
        
        # Calculate damage function
        safe_r = r if r != 0 else 1e-9
        eta = 1.0 - term / safe_r
        
        # Clamp to physically meaningful range
        eta = np.clip(eta, 1e-9, 1.0)
        
        return eta.item() if eta.size == 1 else eta
    
    def get_mullins_parameter_names(self) -> List[str]:
        """Get list of Mullins parameter names."""
        return ['r', 'm', 'beta']
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for Mullins parameters."""
        return [
            (0.1, 10.0),   # r bounds
            (1.0, 100.0),  # m bounds  
            (0.01, 1.0)    # beta bounds
        ]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate Mullins parameters.
        
        Args:
            parameters: Mullins parameters dictionary
            
        Raises:
            ModelError: If parameters are invalid
        """
        required_params = self.get_mullins_parameter_names()
        
        for param_name in required_params:
            if param_name not in parameters:
                raise ModelError(f"Missing required Mullins parameter: {param_name}")
            
            value = parameters[param_name]
            if not isinstance(value, (int, float)) or value <= 0:
                raise ModelError(f"Mullins parameter '{param_name}' must be positive number")


def apply_mullins_damage(undamaged_stress: Union[float, np.ndarray],
                        strain_energy: Union[float, np.ndarray],
                        max_strain_energy: Union[float, np.ndarray],
                        mullins_parameters: Dict[str, Any],
                        variant: str = 'standard') -> Union[float, np.ndarray]:
    """
    Apply Mullins damage to undamaged stress.
    
    Args:
        undamaged_stress: Stress from hyperelastic model
        strain_energy: Current strain energy density
        max_strain_energy: Maximum strain energy in loading history
        mullins_parameters: Mullins model parameters
        variant: Mullins model variant ('standard', 'erf_arg_div_r', 'denom_W')
        
    Returns:
        Damaged stress = η * undamaged_stress
    """
    mullins_model = MullinsEffectModel()
    
    if variant == 'standard':
        eta = mullins_model.calculate_damage_function(
            strain_energy, max_strain_energy, mullins_parameters
        )
    elif variant == 'erf_arg_div_r':
        eta = mullins_model.calculate_damage_function_variant_erf_arg_div_r(
            strain_energy, max_strain_energy, mullins_parameters
        )
    elif variant == 'denom_W':
        eta = mullins_model.calculate_damage_function_variant_denom_W(
            strain_energy, max_strain_energy, mullins_parameters
        )
    else:
        raise ModelError(f"Unknown Mullins variant: {variant}")
    
    # Apply damage to stress
    damaged_stress = eta * undamaged_stress
    
    return damaged_stress
