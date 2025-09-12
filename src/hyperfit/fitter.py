"""
Core fitting engine for the HyperFit library.

This module contains the main Fitter class that orchestrates the entire
fitting process based on the validated configuration.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
import time

from .config import FittingConfiguration
from .data import ExperimentalData
from .results import FitResult, ResultsProcessor, create_failed_result
from .models import create_model
from .strategies import initializers, objectives
from .exceptions import FittingError, ModelError


class Fitter:
    """
    Core fitting engine that orchestrates the hyperelastic model fitting process.
    
    This class coordinates all aspects of the fitting process including:
    - Model creation and configuration
    - Initial parameter guess generation
    - Objective function creation
    - Optimization execution
    - Results processing and validation
    """
    
    def __init__(self, config: FittingConfiguration):
        """
        Initialize the fitter with validated configuration.
        
        Args:
            config: Validated fitting configuration
        """
        self.config = config
        self.model = None
        self.results_processor = None
        
        # Create material model
        self._create_model()
        
        # Initialize results processor
        self._initialize_results_processor()
    
    def _create_model(self) -> None:
        """Create and configure the material model."""
        try:
            self.model = create_model(
                self.config.model_name,
                self.config.model_order,
                **self.config.get_model_config()
            )
        except Exception as e:
            raise FittingError(f"Failed to create model: {e}") from e
    
    def _initialize_results_processor(self) -> None:
        """Initialize the results processor."""
        self.results_processor = ResultsProcessor(
            self.config.get_model_config(),
            self.config.get_data_config()
        )
    
    def execute(self, experimental_data: Dict[str, ExperimentalData]) -> FitResult:
        """
        Execute the complete fitting process.
        
        Args:
            experimental_data: Dictionary of prepared experimental data
            
        Returns:
            FitResult object containing fitting results and diagnostics
        """
        try:
            # Generate initial parameter guess
            initial_params, param_names, bounds = self._generate_initial_guess(experimental_data)
            
            # Create objective function
            objective_func = self._create_objective_function(experimental_data)
            
            # Create Jacobian function if available
            jacobian_func = self._create_jacobian_function(experimental_data)
            
            # Execute optimization
            optimization_result = self._run_optimization(
                objective_func, initial_params, bounds, jacobian_func
            )
            
            # Process and validate results
            result = self._process_results(
                optimization_result, param_names, experimental_data
            )
            
            return result
            
        except Exception as e:
            return create_failed_result(
                f"Fitting failed: {str(e)}", 
                self.config.get_model_config()
            )
    
    def _generate_initial_guess(self, experimental_data: Dict[str, ExperimentalData]) -> Tuple[np.ndarray, List[str], List[Tuple[float, float]]]:
        """
        Generate initial parameter guess using configured strategy.
        
        Returns:
            Tuple of (initial_parameters, parameter_names, parameter_bounds)
        """
        strategy_config = self.config.strategy['initial_guess']
        strategy_method = strategy_config['method']
        
        try:
            return initializers.generate_initial_guess(
                strategy_method, self.model, experimental_data, strategy_config
            )
        except Exception as e:
            raise FittingError(f"Initial guess generation failed: {e}") from e
    
    def _create_objective_function(self, experimental_data: Dict[str, ExperimentalData]) -> callable:
        """
        Create objective function using configured strategy.
        
        Returns:
            Objective function for optimization
        """
        objective_config = self.config.strategy['objective_function']
        objective_type = objective_config['type']
        
        try:
            return objectives.create_objective_function(
                objective_type, self.model, experimental_data, objective_config
            )
        except Exception as e:
            raise FittingError(f"Objective function creation failed: {e}") from e
    
    def _create_jacobian_function(self, experimental_data: Dict[str, ExperimentalData]) -> Optional[callable]:
        """
        Create analytical Jacobian function if available.
        
        Returns:
            Jacobian function or None
        """
        objective_config = self.config.strategy['objective_function']
        objective_type = objective_config['type']
        
        try:
            return objectives.create_jacobian_function(
                objective_type, self.model, experimental_data, objective_config
            )
        except Exception:
            # Fall back to numerical differentiation
            return None
    
    def _run_optimization(self, 
                         objective_func: callable,
                         initial_params: np.ndarray,
                         bounds: List[Tuple[float, float]],
                         jacobian_func: Optional[callable] = None) -> Any:
        """
        Execute the optimization process.
        
        Args:
            objective_func: Objective function to minimize
            initial_params: Initial parameter guess
            bounds: Parameter bounds
            jacobian_func: Optional analytical Jacobian
            
        Returns:
            Optimization result from scipy.optimize
        """
        optimizer_config = self.config.strategy['optimizer']
        methods = optimizer_config['methods']
        
        best_result = None
        best_cost = float('inf')
        
        # Try multiple optimization methods for robustness
        for method in methods:
            try:
                result = self._run_single_optimization(
                    objective_func, initial_params, bounds, method, jacobian_func
                )
                
                # Check if this result is better
                if result.success and result.fun < best_cost:
                    best_result = result
                    best_cost = result.fun
                    
            except Exception as e:
                # Log the error but continue with other methods
                continue
        
        if best_result is None:
            raise FittingError("All optimization methods failed")
        
        # Apply post-processing stability control if configured
        if self.config.strategy.get('stability_control') == 'post':
            best_result = self._apply_stability_post_processing(best_result)
        
        return best_result
    
    def _run_single_optimization(self,
                               objective_func: callable,
                               initial_params: np.ndarray,
                               bounds: List[Tuple[float, float]],
                               method: str,
                               jacobian_func: Optional[callable] = None) -> Any:
        """
        Run optimization with a single method.
        
        Args:
            objective_func: Objective function
            initial_params: Initial parameters
            bounds: Parameter bounds
            method: Optimization method
            jacobian_func: Optional Jacobian function
            
        Returns:
            Optimization result
        """
        # Prepare optimization options
        options = {
            'maxiter': self.config.convergence.get('max_iterations', 1000),
            'ftol': self.config.convergence.get('tolerance', 1e-8),
            'gtol': self.config.convergence.get('relative_tolerance', 1e-6)
        }
        
        # Handle method-specific options
        if method == 'L-BFGS-B':
            options['maxfun'] = options.get('maxiter', 1000) * 10
        elif method == 'trust-constr':
            options = {
                'maxiter': options['maxiter'],
                'gtol': options['gtol'],
                'xtol': options['ftol']
            }
        
        # Create bounds in scipy format
        scipy_bounds = bounds if bounds else None
        
        # Handle penalty-based stability control
        final_objective = objective_func
        if self.config.strategy.get('stability_control') == 'penalty':
            penalty_factor = self.config.strategy.get('penalty_factor', 1e6)
            final_objective = self._create_penalty_objective(objective_func, penalty_factor)
        
        # Run optimization
        result = minimize(
            fun=final_objective,
            x0=initial_params,
            method=method,
            bounds=scipy_bounds,
            jac=jacobian_func,
            options=options
        )
        
        return result
    
    def _create_penalty_objective(self, base_objective: callable, penalty_factor: float) -> callable:
        """
        Create penalized objective function for stability control.
        
        Args:
            base_objective: Base objective function
            penalty_factor: Penalty factor for constraint violations
            
        Returns:
            Penalized objective function
        """
        def penalized_objective(params: np.ndarray) -> float:
            base_cost = base_objective(params)
            
            # Add penalty for stability violations
            penalty = 0.0
            
            try:
                param_dict = self.model.format_parameters(params)
                
                # Model-specific stability checks
                if hasattr(self.model, 'calculate_elastic_moduli'):
                    moduli = self.model.calculate_elastic_moduli(param_dict)
                    G0 = moduli.get('G0', 0)
                    
                    # Penalize negative shear modulus
                    if G0 <= 0:
                        penalty += penalty_factor * abs(G0)
                
            except Exception:
                penalty += penalty_factor
            
            return base_cost + penalty
        
        return penalized_objective
    
    def _apply_stability_post_processing(self, opt_result: Any) -> Any:
        """
        Apply post-processing stability control.
        
        Args:
            opt_result: Optimization result
            
        Returns:
            Modified optimization result
        """
        try:
            params = opt_result.x.copy()
            param_dict = self.model.format_parameters(params)
            
            # Model-specific post-processing
            if hasattr(self.model, 'calculate_elastic_moduli'):
                moduli = self.model.calculate_elastic_moduli(param_dict)
                G0 = moduli.get('G0', 0)
                
                # For Ogden model: flip μ signs if G0 < 0
                if G0 <= 0 and 'mu' in param_dict:
                    mu_values = param_dict['mu']
                    # Flip signs of all μ parameters
                    for i in range(len(mu_values)):
                        if 'mu' in self.model.get_parameter_names():
                            # Find μ parameter indices and flip them
                            param_names = self.model.get_parameter_names()
                            for j, name in enumerate(param_names):
                                if name.startswith('mu_'):
                                    params[j] = -params[j]
            
            # Update the result
            opt_result.x = params
            
        except Exception:
            # If post-processing fails, return original result
            pass
        
        return opt_result
    
    def _process_results(self,
                        opt_result: Any,
                        param_names: List[str],
                        experimental_data: Dict[str, ExperimentalData]) -> FitResult:
        """
        Process optimization results into standardized format.
        
        Args:
            opt_result: Optimization result from scipy
            param_names: Parameter names
            experimental_data: Experimental data
            
        Returns:
            Processed FitResult object
        """
        try:
            # Calculate model predictions for quality metrics
            predicted_data = self._calculate_model_predictions(
                opt_result.x, experimental_data
            )
            
            # Process results using ResultsProcessor
            result = self.results_processor.process_optimization_result(
                opt_result, param_names, experimental_data, predicted_data
            )
            
            return result
            
        except Exception as e:
            return create_failed_result(
                f"Results processing failed: {str(e)}",
                self.config.get_model_config()
            )
    
    def _calculate_model_predictions(self,
                                   params: np.ndarray,
                                   experimental_data: Dict[str, ExperimentalData]) -> Dict[str, np.ndarray]:
        """
        Calculate model predictions for given parameters.
        
        Args:
            params: Parameter vector
            experimental_data: Experimental data
            
        Returns:
            Dictionary of model predictions
        """
        try:
            param_dict = self.model.format_parameters(params)
            predictions = {}
            
            for loading_type, exp_data in experimental_data.items():
                if loading_type == 'volumetric':
                    if hasattr(self.model, 'calculate_pressure'):
                        predictions[loading_type] = self.model.calculate_pressure(
                            exp_data.x_data, param_dict
                        )
                else:
                    predictions[loading_type] = self.model.calculate_stress(
                        exp_data.x_data, param_dict, loading_type
                    )
            
            return predictions
            
        except Exception:
            return {}
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the fitting configuration for potential issues.
        
        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        
        # Check data compatibility with model
        data_types = set(self.config.data.keys())
        
        if self.config.model_name == 'reduced_polynomial':
            if 'volumetric' not in data_types:
                warnings.append(
                    "Reduced Polynomial model benefits from volumetric data "
                    "for better parameter identification"
                )
        
        # Check parameter count vs data points
        total_data_points = sum(len(exp_data) for exp_data in self.config.data.values())
        expected_params = len(self.model.get_parameter_names())
        
        if total_data_points < expected_params * 3:
            warnings.append(
                f"Limited data ({total_data_points} points) for {expected_params} parameters. "
                "Consider reducing model order or adding more data."
            )
        
        # Check optimization method compatibility
        methods = self.config.strategy['optimizer']['methods']
        if 'Nelder-Mead' in methods and len(self.model.get_parameter_bounds()) > 0:
            warnings.append(
                "Nelder-Mead method does not support bounds. "
                "Consider using L-BFGS-B or TNC instead."
            )
        
        return warnings
    
    def get_fitting_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the fitting configuration.
        
        Returns:
            Dictionary with fitting configuration summary
        """
        return {
            'model': {
                'type': self.config.model_name,
                'order': self.config.model_order,
                'parameters': len(self.model.get_parameter_names())
            },
            'data': {
                'types': list(self.config.data.keys()),
                'total_points': sum(len(exp_data) for exp_data in self.config.data.values())
            },
            'strategy': {
                'initial_guess': self.config.strategy['initial_guess']['method'],
                'optimizer': self.config.strategy['optimizer']['methods'],
                'objective': self.config.strategy['objective_function']['type'],
                'stability_control': self.config.strategy.get('stability_control', 'post')
            },
            'mullins_effect': self.config.mullins_effect
        }
