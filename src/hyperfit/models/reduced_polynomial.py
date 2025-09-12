"""
Reduced Polynomial hyperelastic model implementation.

This module implements the Reduced Polynomial (Mooney-Rivlin) model based on
the algorithms from the original fit_rpolynomial.py script.
"""

from typing import Dict, Any, Union, Tuple, List
import numpy as np

from .base import HyperelasticModel, CompressibleModel
from ..exceptions import ModelError


class ReducedPolynomialModel(HyperelasticModel, CompressibleModel):
    """
    Reduced Polynomial (Mooney-Rivlin) hyperelastic model.
    
    The strain energy density function is:
    W = Σᵢ C_i0 * (I₁ - 3)^i + Σᵢ (1/D_i) * (J - 1)^(2i)
    
    Where:
    - C_i0 are the deviatoric material parameters
    - D_i are the volumetric material parameters  
    - I₁ is the first strain invariant
    - J is the volume ratio
    """
    
    def __init__(self, model_order: int):
        """
        Initialize Reduced Polynomial model.
        
        Args:
            model_order: Number of terms (N) in the polynomial
        """
        super().__init__(model_order)
        
        # Parameter limits for physical stability
        self.max_order = 5
        if model_order > self.max_order:
            raise ModelError(f"Reduced Polynomial order {model_order} > {self.max_order} may be unstable")
    
    def calculate_stress(self, 
                        strain: Union[float, np.ndarray], 
                        parameters: Dict[str, Any], 
                        loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate nominal stress using Reduced Polynomial model.
        
        Implementation based on fit_rpolynomial.py algorithm.
        """
        self.validate_parameters(parameters)
        
        strain = np.asarray(strain)
        C_i0 = np.asarray(parameters['C_i0'])
        
        # Convert nominal strain to principal stretch
        lam = strain + 1.0
        
        # Check for invalid stretch values
        if np.any(lam <= 0):
            raise ModelError("Principal stretch must be positive (strain > -1)")
        
        # Calculate invariants and stress factors based on loading type
        if loading_type == 'uniaxial':
            # From original: T_U = 2 * (lam - lam**(-2)) * sum_{i=1 to N} [i * C_i0 * (I1_bar - 3)**(i-1)]
            I1_bar = lam**2 + 2 / lam
            factor = 2 * (lam - lam**(-2))
            
        elif loading_type == 'biaxial':
            # From original: T_B = 2 * (lam - lam**(-5)) * sum_{i=1 to N} [i * C_i0 * (I1_bar - 3)**(i-1)]
            I1_bar = 2 * lam**2 + lam**(-4)
            factor = 2 * (lam - lam**(-5))
            
        elif loading_type == 'planar':
            # From original: T_S = 2 * (lam - lam**(-3)) * sum_{i=1 to N} [i * C_i0 * (I1_bar - 3)**(i-1)]
            I1_bar = lam**2 + 1 + lam**(-2)
            factor = 2 * (lam - lam**(-3))
            
        else:
            raise ModelError(f"Unsupported loading type: {loading_type}")
        
        # Calculate deviatoric stress sum
        deviatoric_stress_sum = np.zeros_like(lam, dtype=float)
        
        # The term (I1_bar - 3) can be slightly negative due to floating point precision
        # when lam is very close to 1. Handle this to avoid complex numbers.
        base = I1_bar - 3
        base = np.where((base < 0) & (base > -1e-9), 0, base)  # Handle small negative values
        
        for i in range(1, self.model_order + 1):
            # The exponent is (i - 1)
            if i == 1:
                # For i=1, exponent is 0, so base^0 = 1
                term = C_i0[i-1] * np.ones_like(base)
            else:
                # For i>1, calculate base^(i-1)
                term = C_i0[i-1] * np.power(base, i-1)
            
            # Handle NaN/inf from large exponents
            term = np.where(np.isfinite(term), term, 0)
            deviatoric_stress_sum += i * term
        
        # Final stress is the product of the factor and the summation
        stress = factor * deviatoric_stress_sum
        
        return stress.item() if stress.size == 1 else stress
    
    def calculate_strain_energy(self, 
                               strain: Union[float, np.ndarray], 
                               parameters: Dict[str, Any], 
                               loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate strain energy density using Reduced Polynomial model.
        """
        self.validate_parameters(parameters)
        
        strain = np.asarray(strain)
        C_i0 = np.asarray(parameters['C_i0'])
        
        # Convert nominal strain to principal stretches
        lambda1, lambda2, lambda3 = self.calculate_principal_stretches(strain, loading_type)
        
        # Calculate first strain invariant
        I1_bar = lambda1**2 + lambda2**2 + lambda3**2
        
        # Calculate deviatoric strain energy
        base = I1_bar - 3
        base = np.where((base < 0) & (base > -1e-9), 0, base)  # Handle small negative values
        
        strain_energy = np.zeros_like(base, dtype=float)
        
        for i in range(1, self.model_order + 1):
            # For strain energy: W = sum(C_i0 * (I1-3)^i), note: exponent is i, not i-1
            term = C_i0[i-1] * np.power(base, i)
            term = np.where(np.isfinite(term), term, 0)
            strain_energy += term
        
        return strain_energy.item() if strain_energy.size == 1 else strain_energy
    
    def calculate_pressure(self, 
                          volume_ratio: Union[float, np.ndarray],
                          parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Calculate hydrostatic pressure from volume ratio.
        
        Based on volumetric part: p = sum_{i=1 to N} [2i * (1/D_i) * (J-1)^(2i-1)]
        """
        if 'D_i' not in parameters:
            raise ModelError("Volumetric parameters D_i required for pressure calculation")
        
        volume_ratio = np.asarray(volume_ratio)
        D_i = np.asarray(parameters['D_i'])
        
        if np.any(volume_ratio <= 0):
            raise ModelError("Volume ratio must be positive")
        
        pressure = np.zeros_like(volume_ratio, dtype=float)
        
        for i in range(1, self.model_order + 1):
            if i <= len(D_i) and D_i[i-1] != 0:
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
        """Get parameter names for Reduced Polynomial model."""
        # Deviatoric parameters
        param_names = [f'C_{i}0' for i in range(1, self.model_order + 1)]
        
        # Volumetric parameters
        param_names.extend([f'D_{i}' for i in range(1, self.model_order + 1)])
        
        return param_names
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        
        # Bounds for C_i0 parameters (can be negative for higher order terms)
        for i in range(self.model_order):
            if i == 0:  # C_10 should be positive for stability
                bounds.append((1e-6, 1e6))
            else:  # Higher order terms can be negative
                bounds.append((-1e6, 1e6))
        
        # Bounds for D_i parameters (must be positive, or very small for near-incompressible)
        for i in range(self.model_order):
            bounds.append((1e-12, 1e3))  # Allow very small values for incompressible behavior
        
        return bounds
    
    def get_initial_guess_lls(self, experimental_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate initial guess using Linear Least Squares method.
        
        This implements the weighted LLS approach from fit_rpolynomial.py.
        """
        # Build the linear system A*params = b for C_i0 and 1/D_i parameters
        A_rows = []
        b_elements = []
        weights_list = []
        
        for loading_type, exp_data in experimental_data.items():
            if loading_type == 'volumetric':
                # Handle volumetric data
                volume_ratio = exp_data.x_data
                pressure = exp_data.y_data
                
                for j_v, p_v in zip(volume_ratio, pressure):
                    if j_v <= 0 or p_v == 0:
                        continue
                    
                    row = np.zeros(2 * self.model_order)
                    # Only volumetric terms contribute
                    for i in range(1, self.model_order + 1):
                        row[self.model_order + i - 1] = -2 * i * (j_v - 1)**(2 * i - 1)
                    
                    A_rows.append(row)
                    b_elements.append(p_v)
                    weights_list.append(1.0 / (p_v**2))
            
            else:
                # Handle mechanical data
                strain = exp_data.x_data
                stress = exp_data.y_data
                
                for eps, sigma in zip(strain, stress):
                    lam = eps + 1.0
                    if lam <= 0 or sigma == 0:
                        continue
                    
                    # Calculate invariants and factors
                    if loading_type == 'uniaxial':
                        I1_bar = lam**2 + 2 / lam
                        factor = 2 * (lam - lam**(-2))
                    elif loading_type == 'biaxial':
                        I1_bar = 2 * lam**2 + lam**(-4)
                        factor = 2 * (lam - lam**(-5))
                    elif loading_type == 'planar':
                        I1_bar = lam**2 + 1 + lam**(-2)
                        factor = 2 * (lam - lam**(-3))
                    else:
                        continue
                    
                    row = np.zeros(2 * self.model_order)
                    # Only deviatoric terms contribute to mechanical response
                    for i in range(1, self.model_order + 1):
                        row[i - 1] = factor * i * (I1_bar - 3)**(i - 1)
                    
                    A_rows.append(row)
                    b_elements.append(sigma)
                    weights_list.append(1.0 / (sigma**2))
        
        if not A_rows:
            raise ModelError("No valid data points for LLS initial guess")
        
        # Solve weighted least squares system
        A = np.array(A_rows)
        b = np.array(b_elements)
        weights = np.array(weights_list)
        
        # Apply weights
        w_sqrt = np.sqrt(weights)
        A_weighted = A * w_sqrt[:, np.newaxis]
        b_weighted = b * w_sqrt
        
        try:
            params, residuals, rank, s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
            
            # Split parameters
            C_i0 = params[:self.model_order]
            inv_D_i = params[self.model_order:]
            
            # Convert 1/D_i back to D_i
            D_i = np.zeros_like(inv_D_i)
            for i, inv_d in enumerate(inv_D_i):
                if abs(inv_d) < 1e-20:
                    D_i[i] = np.inf  # Incompressible
                else:
                    D_i[i] = 1.0 / inv_d
            
            # Return combined parameter vector
            return np.concatenate([C_i0, D_i])
            
        except np.linalg.LinAlgError as e:
            raise ModelError(f"LLS initial guess failed: {e}")
    
    def format_parameters(self, param_vector: np.ndarray) -> Dict[str, Any]:
        """
        Format parameter vector into dictionary.
        
        Args:
            param_vector: Flat array of parameters [C_10, C_20, ..., D_1, D_2, ...]
            
        Returns:
            Dictionary with 'C_i0' and 'D_i' arrays
        """
        if len(param_vector) != 2 * self.model_order:
            raise ModelError(
                f"Expected {2 * self.model_order} parameters, got {len(param_vector)}"
            )
        
        C_i0 = param_vector[:self.model_order]
        D_i = param_vector[self.model_order:]
        
        return {
            'C_i0': C_i0,
            'D_i': D_i
        }
    
    def calculate_elastic_moduli(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate initial elastic moduli from model parameters.
        
        Returns:
            Dictionary with G0, K0, E0, nu0
        """
        C_i0 = np.asarray(parameters['C_i0'])
        
        # Initial shear modulus: G₀ = 2 * C₁₀
        G0 = 2 * C_i0[0]
        
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
