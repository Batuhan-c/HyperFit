"""
Base classes for material models in the HyperFit library.

This module defines abstract base classes that ensure all material models
have a consistent interface for stress and strain energy calculations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
import numpy as np

from ..exceptions import ModelError


class HyperelasticModel(ABC):
    """
    Abstract base class for hyperelastic material models.
    
    This class defines the interface that all hyperelastic models must implement,
    ensuring consistency across different model types (Ogden, Polynomial, etc.).
    """
    
    def __init__(self, model_order: int):
        """
        Initialize the hyperelastic model.
        
        Args:
            model_order: Order or number of terms in the model
        """
        if not isinstance(model_order, int) or model_order < 1:
            raise ModelError("model_order must be a positive integer")
        
        self.model_order = model_order
        self.parameters = {}
    
    @abstractmethod
    def calculate_stress(self, 
                        strain: Union[float, np.ndarray], 
                        parameters: Dict[str, Any], 
                        loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate stress for given strain and loading type.
        
        Args:
            strain: Nominal strain value(s)
            parameters: Model parameters dictionary
            loading_type: Type of loading ('uniaxial', 'biaxial', 'planar')
            
        Returns:
            Calculated nominal stress value(s)
            
        Raises:
            ModelError: If calculation fails or parameters are invalid
        """
        pass
    
    @abstractmethod
    def calculate_strain_energy(self, 
                               strain: Union[float, np.ndarray], 
                               parameters: Dict[str, Any], 
                               loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate strain energy density for given strain and loading type.
        
        Args:
            strain: Nominal strain value(s)
            parameters: Model parameters dictionary
            loading_type: Type of loading ('uniaxial', 'biaxial', 'planar')
            
        Returns:
            Calculated strain energy density value(s)
            
        Raises:
            ModelError: If calculation fails or parameters are invalid
        """
        pass
    
    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        """
        Get list of parameter names for this model.
        
        Returns:
            List of parameter names in the order expected by optimization
        """
        pass
    
    @abstractmethod
    def get_parameter_bounds(self) -> list[Tuple[float, float]]:
        """
        Get parameter bounds for optimization.
        
        Returns:
            List of (lower_bound, upper_bound) tuples for each parameter
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate model parameters.
        
        Args:
            parameters: Model parameters dictionary
            
        Raises:
            ModelError: If parameters are invalid
        """
        required_params = self.get_parameter_names()
        
        for param_name in required_params:
            if param_name not in parameters:
                raise ModelError(f"Missing required parameter: {param_name}")
        
        # Check for extra parameters
        extra_params = set(parameters.keys()) - set(required_params)
        if extra_params:
            raise ModelError(f"Unexpected parameters: {extra_params}")
    
    def calculate_principal_stretches(self, 
                                    strain: Union[float, np.ndarray], 
                                    loading_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate principal stretches from nominal strain for different loading types.
        
        Args:
            strain: Nominal strain value(s)
            loading_type: Type of loading
            
        Returns:
            Tuple of (lambda1, lambda2, lambda3) principal stretch arrays
            
        Raises:
            ModelError: If loading type is unsupported
        """
        strain = np.asarray(strain)
        
        if loading_type == 'uniaxial':
            # Uniaxial tension: λ₁ = 1 + ε, λ₂ = λ₃ = 1/√λ₁ (incompressible)
            lambda1 = strain + 1.0
            lambda2 = lambda3 = 1.0 / np.sqrt(lambda1)
            
        elif loading_type == 'biaxial':
            # Equibiaxial tension: λ₁ = λ₂ = 1 + ε, λ₃ = 1/(λ₁λ₂) = 1/λ₁²
            lambda1 = lambda2 = strain + 1.0
            lambda3 = 1.0 / (lambda1 * lambda2)
            
        elif loading_type == 'planar':
            # Planar tension: λ₁ = 1 + ε, λ₂ = 1, λ₃ = 1/λ₁
            lambda1 = strain + 1.0
            lambda2 = np.ones_like(lambda1)
            lambda3 = 1.0 / lambda1
            
        else:
            raise ModelError(f"Unsupported loading type: {loading_type}")
        
        # Validate physical constraints
        if np.any(lambda1 <= 0) or np.any(lambda2 <= 0) or np.any(lambda3 <= 0):
            raise ModelError("Principal stretches must be positive")
        
        return lambda1, lambda2, lambda3
    
    def calculate_invariants(self, 
                           lambda1: np.ndarray, 
                           lambda2: np.ndarray, 
                           lambda3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate strain invariants from principal stretches.
        
        Args:
            lambda1, lambda2, lambda3: Principal stretches
            
        Returns:
            Tuple of (I1_bar, I2_bar) invariants for isochoric deformation
        """
        # For incompressible materials: J = λ₁λ₂λ₃ = 1
        # Isochoric invariants: Ī₁ = J^(-2/3) * I₁, Ī₂ = J^(-4/3) * I₂
        
        I1 = lambda1**2 + lambda2**2 + lambda3**2
        I2 = lambda1**2 * lambda2**2 + lambda2**2 * lambda3**2 + lambda3**2 * lambda1**2
        
        # For incompressible case, J = 1, so Ī₁ = I₁, Ī₂ = I₂
        I1_bar = I1
        I2_bar = I2
        
        return I1_bar, I2_bar
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(order={self.model_order})"


class MullinsModel(ABC):
    """
    Abstract base class for Mullins damage models.
    
    This class defines the interface for models that incorporate
    the Mullins effect (stress softening) in hyperelastic materials.
    """
    
    @abstractmethod
    def calculate_damage_function(self, 
                                strain_energy: Union[float, np.ndarray],
                                max_strain_energy: Union[float, np.ndarray],
                                parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Calculate damage function η for Mullins effect.
        
        Args:
            strain_energy: Current strain energy density
            max_strain_energy: Maximum strain energy in loading history
            parameters: Mullins model parameters
            
        Returns:
            Damage function values (0 < η ≤ 1)
        """
        pass
    
    @abstractmethod
    def get_mullins_parameter_names(self) -> list[str]:
        """
        Get list of Mullins parameter names.
        
        Returns:
            List of Mullins parameter names
        """
        pass


class CompressibleModel(ABC):
    """
    Abstract base class for compressible hyperelastic models.
    
    This extends the base hyperelastic model to handle volumetric deformation
    and pressure-volume relationships.
    """
    
    @abstractmethod
    def calculate_pressure(self, 
                          volume_ratio: Union[float, np.ndarray],
                          parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Calculate hydrostatic pressure from volume ratio.
        
        Args:
            volume_ratio: Volume ratio J = V/V₀
            parameters: Model parameters including bulk modulus terms
            
        Returns:
            Hydrostatic pressure
        """
        pass
    
    @abstractmethod
    def calculate_bulk_modulus(self, parameters: Dict[str, Any]) -> float:
        """
        Calculate initial bulk modulus from model parameters.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Initial bulk modulus K₀
        """
        pass
