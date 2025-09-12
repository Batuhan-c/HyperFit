"""
Ogden hyperelastic model implementation.

This module implements the Ogden model based on the algorithms from 
the original ogden_fit_scipy.py script.
"""

from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
import math

from .base import HyperelasticModel, CompressibleModel
from ..exceptions import ModelError


class OgdenModel(HyperelasticModel, CompressibleModel):
    """
    Ogden hyperelastic model.
    
    The strain energy density function is:
    W = Σᵢ (2μᵢ/αᵢ²) * (λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3) + Σᵢ (1/D_i) * (J - 1)^(2i)
    
    Where:
    - μᵢ, αᵢ are the Ogden material parameters
    - D_i are the volumetric material parameters
    - λ₁, λ₂, λ₃ are the principal stretches
    - J is the volume ratio
    """
    
    def __init__(self, model_order: int):
        """
        Initialize Ogden model.
        
        Args:
            model_order: Number of Ogden pairs (N)
        """
        super().__init__(model_order)
        
        # Parameter limits for numerical stability
        self.max_order = 5
        if model_order > self.max_order:
            raise ModelError(f"Ogden order {model_order} > {self.max_order} may be unstable")
    
    def calculate_stress(self, 
                        strain: Union[float, np.ndarray], 
                        parameters: Dict[str, Any], 
                        loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate nominal stress using Ogden model.
        
        Implementation based on ogden_fit_scipy.py algorithm.
        """
        self.validate_parameters(parameters)
        
        strain = np.asarray(strain)
        mu_i = np.asarray(parameters['mu'])
        alpha_i = np.asarray(parameters['alpha'])
        
        # Convert nominal strain to principal stretch
        lam = strain + 1.0
        
        # Check for invalid stretch values
        if np.any(lam <= 0):
            raise ModelError("Principal stretch must be positive (strain > -1)")
        
        # Calculate stress based on loading type using formulas from original code
        stress = np.zeros_like(lam, dtype=float)
        
        for i in range(self.model_order):
            mu = mu_i[i]
            alpha = alpha_i[i]
            
            if abs(alpha) < 1e-12:
                continue  # Skip terms with very small alpha
            
            if loading_type == 'uniaxial':
                # From original: c = -0.5 for uniaxial
                c = -0.5
                term = (2.0 * mu / alpha) * (lam**(alpha - 1.0) - lam**(c * alpha - 1.0))
                
            elif loading_type == 'biaxial':
                # From original: c = -2.0 for biaxial  
                c = -2.0
                term = (2.0 * mu / alpha) * (lam**(alpha - 1.0) - lam**(c * alpha - 1.0))
                
            elif loading_type == 'planar':
                # From original: c = -1.0 for planar
                c = -1.0
                term = (2.0 * mu / alpha) * (lam**(alpha - 1.0) - lam**(c * alpha - 1.0))
                
            else:
                raise ModelError(f"Unsupported loading type: {loading_type}")
            
            # Handle potential numerical issues
            term = np.where(np.isfinite(term), term, 0)
            stress += term
        
        return stress.item() if stress.size == 1 else stress
    
    def calculate_stress_derivatives(self, 
                                   strain: Union[float, np.ndarray],
                                   parameters: Dict[str, Any],
                                   loading_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate derivatives of stress with respect to μ and α parameters.
        
        Used for Jacobian calculation in optimization.
        """
        strain = np.asarray(strain)
        mu_i = np.asarray(parameters['mu'])
        alpha_i = np.asarray(parameters['alpha'])
        
        lam = strain + 1.0
        
        # Determine loading factor
        if loading_type == 'uniaxial':
            c = -0.5
        elif loading_type == 'biaxial':
            c = -2.0
        elif loading_type == 'planar':
            c = -1.0
        else:
            raise ModelError(f"Unsupported loading type: {loading_type}")
        
        # Calculate derivatives
        dT_dmu = np.zeros((len(lam), self.model_order))
        dT_dalpha = np.zeros((len(lam), self.model_order))
        
        for i in range(self.model_order):
            alpha = alpha_i[i]
            mu = mu_i[i]
            
            if abs(alpha) < 1e-12:
                continue
            
            # Derivative w.r.t. μ (from original dT_dmu function)
            dT_dmu[:, i] = (2.0 / alpha) * (lam**(alpha - 1.0) - lam**(c * alpha - 1.0))
            
            # Derivative w.r.t. α (from original dT_dalpha function)
            log_lam = np.log(lam)
            term1 = -2.0 * mu / (alpha * alpha) * (lam**(alpha - 1.0) - lam**(c * alpha - 1.0))
            term2 = (2.0 * mu / alpha) * (lam**(alpha - 1.0) * log_lam - c * lam**(c * alpha - 1.0) * log_lam)
            dT_dalpha[:, i] = term1 + term2
        
        return dT_dmu, dT_dalpha
    
    def calculate_strain_energy(self, 
                               strain: Union[float, np.ndarray], 
                               parameters: Dict[str, Any], 
                               loading_type: str) -> Union[float, np.ndarray]:
        """
        Calculate strain energy density using Ogden model.
        """
        self.validate_parameters(parameters)
        
        strain = np.asarray(strain)
        mu_i = np.asarray(parameters['mu'])
        alpha_i = np.asarray(parameters['alpha'])
        
        # Convert nominal strain to principal stretches
        lambda1, lambda2, lambda3 = self.calculate_principal_stretches(strain, loading_type)
        
        # Calculate Ogden strain energy
        strain_energy = np.zeros_like(lambda1, dtype=float)
        
        for i in range(self.model_order):
            mu = mu_i[i]
            alpha = alpha_i[i]
            
            if abs(alpha) < 1e-12:
                continue
            
            # W_i = (2μᵢ/αᵢ²) * (λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3)
            term = (2.0 * mu / (alpha * alpha)) * (
                lambda1**alpha + lambda2**alpha + lambda3**alpha - 3.0
            )
            
            # Handle numerical issues
            term = np.where(np.isfinite(term), term, 0)
            strain_energy += term
        
        return strain_energy.item() if strain_energy.size == 1 else strain_energy
    
    def calculate_pressure(self, 
                          volume_ratio: Union[float, np.ndarray],
                          parameters: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Calculate hydrostatic pressure from volume ratio.
        
        For Ogden model: p = sum_{i=1 to N} [2i * (1/D_i) * (J-1)^(2i-1)]
        """
        if 'd' not in parameters:
            raise ModelError("Volumetric parameters 'd' required for pressure calculation")
        
        volume_ratio = np.asarray(volume_ratio)
        d_i = np.asarray(parameters['d'])
        
        if np.any(volume_ratio <= 0):
            raise ModelError("Volume ratio must be positive")
        
        pressure = np.zeros_like(volume_ratio, dtype=float)
        
        for i in range(1, min(self.model_order, len(d_i)) + 1):
            if d_i[i-1] != 0:
                # p_i = 2i * (1/D_i) * (J-1)^(2i-1)
                term_power = (volume_ratio - 1)**(2 * i - 1)
                term = 2 * i * (1.0 / d_i[i-1]) * term_power
                term = np.where(np.isfinite(term), term, 0)
                pressure += term
        
        return pressure.item() if pressure.size == 1 else pressure
    
    def calculate_bulk_modulus(self, parameters: Dict[str, Any]) -> float:
        """
        Calculate initial bulk modulus K₀ = 2/D₁.
        """
        if 'd' not in parameters:
            raise ModelError("Volumetric parameters 'd' required for bulk modulus calculation")
        
        d_i = np.asarray(parameters['d'])
        
        if len(d_i) == 0 or d_i[0] == 0:
            return np.inf  # Incompressible material
        
        return 2.0 / d_i[0]
    
    def get_parameter_names(self) -> List[str]:
        """Get parameter names for Ogden model."""
        # Interleaved μ and α parameters as in original code
        param_names = []
        for i in range(self.model_order):
            param_names.extend([f'mu_{i+1}', f'alpha_{i+1}'])
        
        # Volumetric parameters
        param_names.extend([f'd_{i+1}' for i in range(self.model_order)])
        
        return param_names
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        
        # Bounds for μ and α parameters
        for i in range(self.model_order):
            # μ bounds (can be negative)
            bounds.append((-1e5, 1e5))
            # α bounds (avoid zero and very large values)
            bounds.append((-10.0, 10.0))
        
        # Bounds for d parameters (must be positive)
        for i in range(self.model_order):
            bounds.append((1e-12, 1e3))
        
        return bounds
    
    def get_initial_guess_heuristic(self, experimental_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate heuristic initial guess for Ogden parameters.
        
        Based on the approach in ogden_fit_scipy.py.
        """
        # Estimate material stiffness from data
        mu_max = self._estimate_stiffness_scale(experimental_data)
        
        # Generate random parameters with stability constraint
        np.random.seed(12345)  # For reproducibility
        
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            # Generate random μ and α values
            mu_vals = np.random.uniform(-mu_max, mu_max, self.model_order)
            alpha_vals = np.random.uniform(-10.0, 10.0, self.model_order)
            
            # Check stability constraint: G₀ = Σ(μᵢ * αᵢ) > 0
            g0 = np.sum(mu_vals * alpha_vals)
            if g0 > 0:
                # Generate d parameters
                d_vals = np.random.uniform(1e-6, 1e-3, self.model_order)
                
                # Interleave μ and α, then append d
                params = np.zeros(3 * self.model_order)
                for i in range(self.model_order):
                    params[2*i] = mu_vals[i]
                    params[2*i+1] = alpha_vals[i]
                params[2*self.model_order:] = d_vals
                
                return params
            
            attempts += 1
        
        raise ModelError("Failed to generate stable initial guess after maximum attempts")
    
    def get_initial_guess_lls(self, experimental_data: Dict[str, Any], 
                             alpha_guesses: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate initial guess using Linear Least Squares for μ with fixed α.
        
        Based on the LLS approach in ogden_fit_scipy.py.
        """
        # Set α values
        if alpha_guesses is not None and len(alpha_guesses) == self.model_order:
            alphas = np.array(alpha_guesses, dtype=float)
        else:
            # Default α values [1, 2, 3, ...]
            alphas = np.array([float(i+1) for i in range(self.model_order)], dtype=float)
        
        # Build linear system A * μ = b
        A_rows = []
        b_values = []
        
        for loading_type, exp_data in experimental_data.items():
            if loading_type == 'volumetric':
                continue  # Skip volumetric data for μ fitting
            
            strain = exp_data.x_data
            stress = exp_data.y_data
            
            # Determine loading factor
            if loading_type == 'uniaxial':
                c = -0.5
            elif loading_type == 'biaxial':
                c = -2.0
            elif loading_type == 'planar':
                c = -1.0
            else:
                continue
            
            for eps, sigma in zip(strain, stress):
                lam = eps + 1.0
                if lam <= 0:
                    continue
                
                b_values.append(sigma)
                row = np.zeros(self.model_order)
                
                for j in range(self.model_order):
                    alpha = alphas[j]
                    if abs(alpha) > 1e-12:
                        row[j] = (2.0 / alpha) * (lam**(alpha - 1.0) - lam**(c * alpha - 1.0))
                
                A_rows.append(row)
        
        if not A_rows:
            raise ModelError("No valid data for LLS initial guess")
        
        # Solve least squares system
        A = np.array(A_rows)
        b = np.array(b_values)
        
        try:
            mu_vals, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Generate d parameters
            d_vals = np.full(self.model_order, 1e-6)
            
            # Format parameters: interleaved μ and α, then d
            params = np.zeros(3 * self.model_order)
            for i in range(self.model_order):
                params[2*i] = mu_vals[i]
                params[2*i+1] = alphas[i]
            params[2*self.model_order:] = d_vals
            
            return params
            
        except np.linalg.LinAlgError as e:
            raise ModelError(f"LLS initial guess failed: {e}")
    
    def _estimate_stiffness_scale(self, experimental_data: Dict[str, Any]) -> float:
        """Estimate material stiffness scale from experimental data."""
        total_slope = 0.0
        total_points = 0
        
        for loading_type, exp_data in experimental_data.items():
            if loading_type == 'volumetric':
                continue
            
            strain = exp_data.x_data
            stress = exp_data.y_data
            
            for eps, sigma in zip(strain, stress):
                if abs(eps) > 1e-6:
                    total_slope += sigma / eps
                    total_points += 1
        
        avg_slope = (total_slope / total_points) if total_points > 0 else 20.0
        return max(20.0, abs(avg_slope))
    
    def format_parameters(self, param_vector: np.ndarray) -> Dict[str, Any]:
        """
        Format parameter vector into dictionary.
        
        Args:
            param_vector: Flat array [μ₁, α₁, μ₂, α₂, ..., d₁, d₂, ...]
            
        Returns:
            Dictionary with 'mu', 'alpha', and 'd' arrays
        """
        if len(param_vector) != 3 * self.model_order:
            raise ModelError(
                f"Expected {3 * self.model_order} parameters, got {len(param_vector)}"
            )
        
        # Extract interleaved μ and α
        mu = np.zeros(self.model_order)
        alpha = np.zeros(self.model_order)
        
        for i in range(self.model_order):
            mu[i] = param_vector[2*i]
            alpha[i] = param_vector[2*i+1]
        
        # Extract d parameters
        d = param_vector[2*self.model_order:]
        
        return {
            'mu': mu,
            'alpha': alpha,
            'd': d
        }
    
    def calculate_elastic_moduli(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate initial elastic moduli from Ogden parameters.
        
        Returns:
            Dictionary with G0, K0, E0, nu0
        """
        mu_i = np.asarray(parameters['mu'])
        alpha_i = np.asarray(parameters['alpha'])
        
        # Initial shear modulus: G₀ = Σ(μᵢ * αᵢ)
        G0 = np.sum(mu_i * alpha_i)
        
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
