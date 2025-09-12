"""
Polynomial hyperelastic model implementation.

This module implements the full Polynomial (Mooney-Rivlin) model based on
the algorithms from the original fit_polynomial.py script.
"""

from typing import Dict, Any, Union, Tuple, List
import numpy as np

from .base import HyperelasticModel, CompressibleModel
from ..exceptions import ModelError


class PolynomialModel(HyperelasticModel, CompressibleModel):
    """
    Polynomial (Mooney-Rivlin) hyperelastic model.
    
    The strain energy density function is:
    W = Σᵢⱼ C_ij * (I₁ - 3)^i * (I₂ - 3)^j + Σᵢ (1/D_i) * (J - 1)^(2i)
    
    Where:
    - C_ij are the deviatoric material parameters
    - D_i are the volumetric material parameters
    - I₁, I₂ are the strain invariants
    - J is the volume ratio
    """
    
    def __init__(self, model_order: int):
        """
        Initialize Polynomial model.
        
        Args:
            model_order: Order (N) of the polynomial
        """
        super().__init__(model_order)
        
        # Generate C_ij parameter pairs for given order
        self.c_ij_pairs = []
        for order in range(1, model_order + 1):
            for i in range(order, -1, -1):
                j = order - i
                self.c_ij_pairs.append((i, j))
        
        self.num_c_params = len(self.c_ij_pairs)
        
        # Parameter limits for stability
        self.max_order = 4
        if model_order > self.max_order:
            raise ModelError(f"Polynomial order {model_order} > {self.max_order} may be unstable")
    
    def calculate_stress(self, 
                        strain: Union[float, np.ndarray], 
                        parameters: Dict[str, Any], 
                        loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate nominal stress using Polynomial model.
        
        Implementation based on fit_polynomial.py algorithm.
        """
        self.validate_parameters(parameters)
        
        strain = np.asarray(strain)
        C_ij = parameters['C_ij']
        
        # Convert nominal strain to principal stretch
        lam = strain + 1.0
        
        # Check for invalid stretch values
        if np.any(lam <= 0):
            raise ModelError("Principal stretch must be positive (strain > -1)")
        
        # Calculate invariants based on loading type
        if loading_type == 'uniaxial':
            I1_bar = lam**2 + 2 / lam
            I2_bar = 2 * lam + 1 / lam**2
            factor = 2 * (1 - lam**(-3))  # Abaqus formula factor
            
        elif loading_type == 'biaxial':
            I1_bar = 2 * lam**2 + lam**(-4)
            I2_bar = lam**4 + 2 * lam**(-2)
            factor = 2 * (lam - lam**(-5))
            
        elif loading_type == 'planar':
            I1_bar = lam**2 + 1 + lam**(-2)
            I2_bar = I1_bar  # For planar deformation
            factor = 2 * (lam - lam**(-3))
            
        else:
            raise ModelError(f"Unsupported loading type: {loading_type}")
        
        # Calculate stress sum
        stress = np.zeros_like(lam, dtype=float)
        
        for idx, (i, j) in enumerate(self.c_ij_pairs):
            C = C_ij.get((i, j), 0) if isinstance(C_ij, dict) else C_ij[idx]
            
            # Calculate partial derivatives
            dU_dI1_term = i * C * np.power(I1_bar - 3, max(0, i - 1)) * np.power(I2_bar - 3, j) if i > 0 else 0
            dU_dI2_term = j * C * np.power(I1_bar - 3, i) * np.power(I2_bar - 3, max(0, j - 1)) if j > 0 else 0
            
            # Apply loading-specific formula
            if loading_type == 'uniaxial':
                # Abaqus formula: factor * [lam * dU/dI1 + dU/dI2]
                term = factor * (lam * dU_dI1_term + dU_dI2_term)
            elif loading_type == 'biaxial':
                # Abaqus formula: factor * [dU/dI1 + lam^2 * dU/dI2]
                term = factor * (dU_dI1_term + (lam**2) * dU_dI2_term)
            else:  # planar
                term = factor * (dU_dI1_term + dU_dI2_term)
            
            # Handle numerical issues
            term = np.where(np.isfinite(term), term, 0)
            stress += term
        
        return stress.item() if stress.size == 1 else stress
    
    def calculate_strain_energy(self, 
                               strain: Union[float, np.ndarray], 
                               parameters: Dict[str, Any], 
                               loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate strain energy density using Polynomial model.
        """
        self.validate_parameters(parameters)
        
        strain = np.asarray(strain)
        C_ij = parameters['C_ij']
        
        # Convert nominal strain to principal stretches
        lambda1, lambda2, lambda3 = self.calculate_principal_stretches(strain, loading_type)
        
        # Calculate strain invariants
        I1_bar, I2_bar = self.calculate_invariants(lambda1, lambda2, lambda3)
        
        # Calculate strain energy
        strain_energy = np.zeros_like(I1_bar, dtype=float)
        
        for idx, (i, j) in enumerate(self.c_ij_pairs):
            C = C_ij.get((i, j), 0) if isinstance(C_ij, dict) else C_ij[idx]
            
            # W_ij = C_ij * (I1 - 3)^i * (I2 - 3)^j
            term = C * np.power(I1_bar - 3, i) * np.power(I2_bar - 3, j)
            term = np.where(np.isfinite(term), term, 0)
            strain_energy += term
        
        return strain_energy.item() if strain_energy.size == 1 else strain_energy
    
    def calculate_pressure(self, 
                          volume_ratio: Union[float, np.ndarray],
                          parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Calculate hydrostatic pressure from volume ratio.
        """
        if 'D_i' not in parameters:
            raise ModelError("Volumetric parameters D_i required for pressure calculation")
        
        volume_ratio = np.asarray(volume_ratio)
        D_i = np.asarray(parameters['D_i'])
        
        if np.any(volume_ratio <= 0):
            raise ModelError("Volume ratio must be positive")
        
        pressure = np.zeros_like(volume_ratio, dtype=float)
        
        for i in range(1, min(self.model_order, len(D_i)) + 1):
            if D_i[i-1] != 0:
                # p_i = 2i * (1/D_i) * (J-1)^(2i-1)
                term_power = (volume_ratio - 1)**(2 * i - 1)
                term = 2 * i * (1.0 / D_i[i-1]) * term_power
                term = np.where(np.isfinite(term), term, 0)
                pressure += term
        
        return pressure.item() if pressure.size == 1 else pressure
    
    def calculate_bulk_modulus(self, parameters: Dict[str, Any]) -> float:
        """
        Calculate initial bulk modulus K₀ = 2/D₁.
        """
        if 'D_i' not in parameters:
            raise ModelError("Volumetric parameters D_i required for bulk modulus calculation")
        
        D_i = np.asarray(parameters['D_i'])
        
        if len(D_i) == 0 or D_i[0] == 0:
            return np.inf  # Incompressible material
        
        return 2.0 / D_i[0]
    
    def get_parameter_names(self) -> List[str]:
        """Get parameter names for Polynomial model."""
        # C_ij parameters
        param_names = [f'C_{i}{j}' for i, j in self.c_ij_pairs]
        
        # D_i parameters
        param_names.extend([f'D_{i}' for i in range(1, self.model_order + 1)])
        
        return param_names
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        
        # Bounds for C_ij parameters
        for i, j in self.c_ij_pairs:
            if i == 1 and j == 0:  # C_10 should be positive
                bounds.append((1e-6, 1e6))
            elif i == 0 and j == 1:  # C_01 should be positive
                bounds.append((1e-6, 1e6))
            else:  # Higher order terms can be negative
                bounds.append((-1e6, 1e6))
        
        # Bounds for D_i parameters
        for i in range(self.model_order):
            bounds.append((1e-12, 1e3))
        
        return bounds
    
    def format_parameters(self, param_vector: np.ndarray) -> Dict[str, Any]:
        """
        Format parameter vector into dictionary.
        
        Args:
            param_vector: Flat array of parameters [C_ij..., D_i...]
            
        Returns:
            Dictionary with 'C_ij' and 'D_i'
        """
        expected_length = self.num_c_params + self.model_order
        if len(param_vector) != expected_length:
            raise ModelError(
                f"Expected {expected_length} parameters, got {len(param_vector)}"
            )
        
        # Extract C_ij parameters
        C_ij_flat = param_vector[:self.num_c_params]
        C_ij = {}
        for k, (i, j) in enumerate(self.c_ij_pairs):
            C_ij[(i, j)] = C_ij_flat[k]
        
        # Extract D_i parameters
        D_i = param_vector[self.num_c_params:]
        
        return {
            'C_ij': C_ij,
            'D_i': D_i
        }
    
    def calculate_elastic_moduli(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate initial elastic moduli from Polynomial parameters.
        
        Returns:
            Dictionary with G0, K0, E0, nu0
        """
        C_ij = parameters['C_ij']
        
        # Initial shear modulus: G₀ = 2 * (C₁₀ + C₀₁)
        C10 = C_ij.get((1, 0), 0.0)
        C01 = C_ij.get((0, 1), 0.0)
        G0 = 2 * (C10 + C01)
        
        # Initial bulk modulus
        K0 = self.calculate_bulk_modulus(parameters)
        
        if np.isinf(K0):
            # Incompressible case
            E0 = 3 * G0
            nu0 = 0.5
        else:
            # Compressible case
            E0 = (9 * K0 * G0) / (3 * K0 + G0)
            nu0 = (3 * K0 - 2 * G0) / (2 * (3 * K0 + G0))
        
        return {
            'G0': float(G0),
            'K0': float(K0),
            'E0': float(E0),
            'nu0': float(nu0)
        }
