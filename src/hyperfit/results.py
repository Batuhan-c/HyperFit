"""
Results handling and formatting for the HyperFit library.

This module provides standardized result containers and formatting
for fitting results, diagnostics, and quality metrics.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
from dataclasses import dataclass, asdict
import time

from .exceptions import HyperFitError


@dataclass
class FitResult:
    """
    Standardized container for fitting results.
    
    This class provides a consistent interface for storing and accessing
    fitting results, including parameters, diagnostics, and quality metrics.
    """
    
    # Core results
    success: bool
    message: str
    parameters: Optional[Dict[str, Union[float, np.ndarray]]] = None
    
    # Optimization details
    cost: Optional[float] = None
    iterations: Optional[int] = None
    method_used: Optional[str] = None
    convergence_reason: Optional[str] = None
    
    # Quality metrics
    rms_error: Optional[float] = None
    max_error: Optional[float] = None
    r_squared: Optional[float] = None
    
    # Diagnostics
    jacobian_condition: Optional[float] = None
    parameter_uncertainties: Optional[Dict[str, float]] = None
    residuals: Optional[np.ndarray] = None
    
    # Metadata
    model_type: Optional[str] = None
    model_order: Optional[int] = None
    data_types: Optional[List[str]] = None
    fitting_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.
        
        Returns:
            Dictionary representation of the results
        """
        result_dict = asdict(self)
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
            elif isinstance(value, dict):
                # Handle nested dictionaries that might contain numpy arrays
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        value[sub_key] = sub_value.tolist()
        
        return result_dict
    
    def get_parameter_summary(self) -> str:
        """
        Get a formatted summary of fitted parameters.
        
        Returns:
            Human-readable parameter summary
        """
        if not self.parameters:
            return "No parameters available"
        
        lines = [f"Fitted Parameters ({self.model_type}, N={self.model_order}):"]
        lines.append("-" * 50)
        
        for param_name, value in self.parameters.items():
            if isinstance(value, np.ndarray):
                if len(value) <= 5:
                    # Show all values for small arrays
                    value_str = ", ".join(f"{v:.6e}" for v in value)
                    lines.append(f"  {param_name}: [{value_str}]")
                else:
                    # Show first few and last few for large arrays
                    first_vals = ", ".join(f"{v:.6e}" for v in value[:2])
                    last_vals = ", ".join(f"{v:.6e}" for v in value[-2:])
                    lines.append(f"  {param_name}: [{first_vals}, ..., {last_vals}] ({len(value)} values)")
            else:
                lines.append(f"  {param_name}: {value:.6e}")
        
        return "\n".join(lines)
    
    def get_quality_summary(self) -> str:
        """
        Get a formatted summary of fitting quality metrics.
        
        Returns:
            Human-readable quality summary
        """
        lines = ["Fitting Quality Metrics:"]
        lines.append("-" * 30)
        
        if self.rms_error is not None:
            lines.append(f"  RMS Error: {self.rms_error:.6e}")
        
        if self.max_error is not None:
            lines.append(f"  Max Error: {self.max_error:.6e}")
        
        if self.r_squared is not None:
            lines.append(f"  R-squared: {self.r_squared:.6f}")
        
        if self.cost is not None:
            lines.append(f"  Final Cost: {self.cost:.6e}")
        
        if self.jacobian_condition is not None:
            lines.append(f"  Jacobian Condition: {self.jacobian_condition:.2e}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation of fit result."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"FitResult({status}: {self.message})"


class ResultsProcessor:
    """
    Processes raw optimization results into standardized FitResult objects.
    """
    
    def __init__(self, model_config: Dict[str, Any], data_config: Dict[str, Any]):
        """
        Initialize results processor.
        
        Args:
            model_config: Model configuration dictionary
            data_config: Data configuration dictionary
        """
        self.model_config = model_config
        self.data_config = data_config
        self.start_time = time.time()
    
    def process_optimization_result(self, 
                                  opt_result: Any, 
                                  parameter_names: List[str],
                                  experimental_data: Dict[str, Any],
                                  predicted_data: Optional[Dict[str, Any]] = None) -> FitResult:
        """
        Process scipy optimization result into standardized format.
        
        Args:
            opt_result: Result from scipy.optimize
            parameter_names: Names of fitted parameters
            experimental_data: Original experimental data
            predicted_data: Model predictions (optional)
            
        Returns:
            Standardized FitResult object
        """
        fitting_time = time.time() - self.start_time
        
        # Extract basic optimization info
        success = getattr(opt_result, 'success', False)
        message = getattr(opt_result, 'message', 'No message available')
        cost = getattr(opt_result, 'fun', None)
        iterations = getattr(opt_result, 'nit', None)
        
        # Process parameters
        parameters = None
        if hasattr(opt_result, 'x') and opt_result.x is not None:
            parameters = self._format_parameters(opt_result.x, parameter_names)
        
        # Calculate quality metrics
        quality_metrics = {}
        if predicted_data and experimental_data:
            quality_metrics = self._calculate_quality_metrics(
                experimental_data, predicted_data
            )
        
        # Extract diagnostics
        jacobian_condition = self._calculate_jacobian_condition(opt_result)
        parameter_uncertainties = self._estimate_parameter_uncertainties(opt_result)
        
        return FitResult(
            success=success,
            message=message,
            parameters=parameters,
            cost=cost,
            iterations=iterations,
            method_used=getattr(opt_result, 'method', None),
            convergence_reason=message if not success else 'Converged',
            rms_error=quality_metrics.get('rms_error'),
            max_error=quality_metrics.get('max_error'),
            r_squared=quality_metrics.get('r_squared'),
            jacobian_condition=jacobian_condition,
            parameter_uncertainties=parameter_uncertainties,
            residuals=getattr(opt_result, 'residuals', None),
            model_type=self.model_config.get('name'),
            model_order=self.model_config.get('order'),
            data_types=list(self.data_config.keys()),
            fitting_time=fitting_time
        )
    
    def _format_parameters(self, param_values: np.ndarray, param_names: List[str]) -> Dict[str, Any]:
        """Format parameter values with appropriate names."""
        if len(param_values) != len(param_names):
            raise HyperFitError(
                f"Parameter count mismatch: {len(param_values)} values, "
                f"{len(param_names)} names"
            )
        
        parameters = {}
        
        # Group parameters by type (e.g., C_i0, mu_i, alpha_i, etc.)
        param_groups = {}
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            # Extract parameter type (everything before the index)
            if '_' in name:
                param_type = '_'.join(name.split('_')[:-1])
                if param_type not in param_groups:
                    param_groups[param_type] = []
                param_groups[param_type].append(value)
            else:
                # Single parameter
                parameters[name] = float(value)
        
        # Convert parameter groups to arrays
        for param_type, values in param_groups.items():
            if len(values) == 1:
                parameters[param_type] = float(values[0])
            else:
                parameters[param_type] = np.array(values)
        
        return parameters
    
    def _calculate_quality_metrics(self, 
                                 experimental_data: Dict[str, Any], 
                                 predicted_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fitting quality metrics."""
        all_residuals = []
        all_experimental = []
        all_predicted = []
        
        # Collect all residuals across loading types
        for loading_type in experimental_data.keys():
            if loading_type in predicted_data:
                exp_values = experimental_data[loading_type].y_data
                pred_values = predicted_data[loading_type]
                
                residuals = exp_values - pred_values
                all_residuals.extend(residuals)
                all_experimental.extend(exp_values)
                all_predicted.extend(pred_values)
        
        if not all_residuals:
            return {}
        
        all_residuals = np.array(all_residuals)
        all_experimental = np.array(all_experimental)
        all_predicted = np.array(all_predicted)
        
        # Calculate metrics
        rms_error = np.sqrt(np.mean(all_residuals**2))
        max_error = np.max(np.abs(all_residuals))
        
        # R-squared calculation
        ss_res = np.sum(all_residuals**2)
        ss_tot = np.sum((all_experimental - np.mean(all_experimental))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'rms_error': float(rms_error),
            'max_error': float(max_error),
            'r_squared': float(r_squared)
        }
    
    def _calculate_jacobian_condition(self, opt_result: Any) -> Optional[float]:
        """Calculate condition number of Jacobian matrix."""
        if hasattr(opt_result, 'jac') and opt_result.jac is not None:
            try:
                # Calculate condition number
                jac = opt_result.jac
                if jac.ndim == 2:
                    cond = np.linalg.cond(jac)
                    return float(cond) if np.isfinite(cond) else None
            except Exception:
                pass
        return None
    
    def _estimate_parameter_uncertainties(self, opt_result: Any) -> Optional[Dict[str, float]]:
        """Estimate parameter uncertainties from Hessian or covariance."""
        # This is a simplified implementation
        # In practice, you might want to calculate proper confidence intervals
        if hasattr(opt_result, 'hess_inv') and opt_result.hess_inv is not None:
            try:
                # Extract diagonal elements (variances)
                if hasattr(opt_result.hess_inv, 'diagonal'):
                    variances = opt_result.hess_inv.diagonal()
                elif isinstance(opt_result.hess_inv, np.ndarray):
                    variances = np.diag(opt_result.hess_inv)
                else:
                    return None
                
                # Convert to standard deviations
                std_errors = np.sqrt(np.abs(variances))
                return {f'param_{i}': float(err) for i, err in enumerate(std_errors)}
            except Exception:
                pass
        return None


def create_failed_result(error_message: str, 
                        model_config: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Create a FitResult object for failed fitting attempts.
    
    Args:
        error_message: Description of the failure
        model_config: Optional model configuration
        
    Returns:
        FitResult object indicating failure
    """
    return FitResult(
        success=False,
        message=error_message,
        model_type=model_config.get('name') if model_config else None,
        model_order=model_config.get('order') if model_config else None
    )
