"""
Configuration validation and management for the HyperFit library.

This module handles validation of user configuration dictionaries and provides
structured access to configuration parameters with appropriate defaults.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np

from .exceptions import ConfigurationError


class FittingConfiguration:
    """
    Validates and manages fitting configuration parameters.
    
    This class takes a raw configuration dictionary, validates all parameters,
    applies defaults where appropriate, and provides structured access to
    configuration settings.
    """
    
    # Supported models and their required parameters
    SUPPORTED_MODELS = {
        'ogden': ['model_order'],
        'polynomial': ['model_order'], 
        'reduced_polynomial': ['model_order']
    }
    
    # Supported loading types
    SUPPORTED_LOADING_TYPES = ['uniaxial', 'biaxial', 'planar', 'volumetric']
    
    # Supported optimizer methods
    SUPPORTED_OPTIMIZERS = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'Nelder-Mead']
    
    # Supported initial guess methods
    SUPPORTED_INITIAL_GUESS = ['lls', 'heuristic']
    
    # Supported objective function types
    SUPPORTED_OBJECTIVES = ['absolute_error', 'relative_error', 'stress', 'eta']

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize and validate the configuration.
        
        Args:
            config_dict: Raw configuration dictionary from user
            
        Raises:
            ConfigurationError: If configuration is invalid or incomplete
        """
        self.raw_config = config_dict.copy()
        self._validate_and_parse()

    def _validate_and_parse(self) -> None:
        """Validate the configuration and extract parameters."""
        
        # Validate required top-level keys
        required_keys = ['model', 'experimental_data', 'fitting_strategy']
        for key in required_keys:
            if key not in self.raw_config:
                raise ConfigurationError(f"Configuration must contain '{key}' key")

        # Validate and parse model configuration
        self._validate_model()
        
        # Validate and parse experimental data configuration
        self._validate_experimental_data()
        
        # Validate and parse fitting strategy
        self._validate_fitting_strategy()
        
        # Parse optional parameters
        self._parse_optional_parameters()

    def _validate_model(self) -> None:
        """Validate model configuration."""
        model_name = self.raw_config['model'].lower()
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ConfigurationError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_name = model_name
        
        # Check for required model parameters
        required_params = self.SUPPORTED_MODELS[model_name]
        for param in required_params:
            if param not in self.raw_config:
                raise ConfigurationError(
                    f"Model '{model_name}' requires parameter '{param}'"
                )
        
        # Validate model order
        self.model_order = self.raw_config.get('model_order', 1)
        if not isinstance(self.model_order, int) or self.model_order < 1:
            raise ConfigurationError(
                "model_order must be a positive integer"
            )
        
        # Model order limits for stability
        max_orders = {'ogden': 5, 'polynomial': 4, 'reduced_polynomial': 5}
        if self.model_order > max_orders.get(model_name, 5):
            raise ConfigurationError(
                f"model_order {self.model_order} too high for {model_name} model. "
                f"Maximum recommended: {max_orders.get(model_name, 5)}"
            )

    def _validate_experimental_data(self) -> None:
        """Validate experimental data configuration."""
        data_config = self.raw_config['experimental_data']
        
        if not isinstance(data_config, dict):
            raise ConfigurationError("experimental_data must be a dictionary")
        
        if not data_config:
            raise ConfigurationError("experimental_data cannot be empty")
        
        self.data = {}
        
        for loading_type, data in data_config.items():
            if loading_type not in self.SUPPORTED_LOADING_TYPES:
                raise ConfigurationError(
                    f"Unsupported loading type '{loading_type}'. "
                    f"Supported types: {self.SUPPORTED_LOADING_TYPES}"
                )
            
            # Validate data format
            if not isinstance(data, dict):
                raise ConfigurationError(
                    f"Data for '{loading_type}' must be a dictionary"
                )
            
            # Check for required fields (strain/stress or pressure/j for volumetric)
            if loading_type == 'volumetric':
                required_fields = ['pressure', 'j']
            else:
                required_fields = ['strain', 'stress']
            
            for field in required_fields:
                if field not in data:
                    raise ConfigurationError(
                        f"'{loading_type}' data missing required field '{field}'"
                    )
                
                # Validate data is array-like
                try:
                    arr = np.asarray(data[field])
                    if arr.size == 0:
                        raise ConfigurationError(
                            f"'{loading_type}' {field} data cannot be empty"
                        )
                    data[field] = arr
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(
                        f"'{loading_type}' {field} data must be array-like: {e}"
                    )
            
            # Validate data lengths match
            field_lengths = [len(data[field]) for field in required_fields]
            if len(set(field_lengths)) > 1:
                raise ConfigurationError(
                    f"'{loading_type}' data arrays must have same length. "
                    f"Got lengths: {dict(zip(required_fields, field_lengths))}"
                )
            
            self.data[loading_type] = data

    def _validate_fitting_strategy(self) -> None:
        """Validate fitting strategy configuration."""
        strategy_config = self.raw_config['fitting_strategy']
        
        if not isinstance(strategy_config, dict):
            raise ConfigurationError("fitting_strategy must be a dictionary")
        
        self.strategy = {}
        
        # Validate initial guess strategy
        initial_guess = strategy_config.get('initial_guess', {'method': 'lls'})
        if isinstance(initial_guess, str):
            initial_guess = {'method': initial_guess}
        elif not isinstance(initial_guess, dict):
            raise ConfigurationError("initial_guess must be string or dictionary")
        
        method = initial_guess.get('method', 'lls').lower()
        if method not in self.SUPPORTED_INITIAL_GUESS:
            raise ConfigurationError(
                f"Unsupported initial guess method '{method}'. "
                f"Supported methods: {self.SUPPORTED_INITIAL_GUESS}"
            )
        
        initial_guess['method'] = method
        self.strategy['initial_guess'] = initial_guess
        
        # Validate optimizer configuration
        optimizer = strategy_config.get('optimizer', {'methods': ['L-BFGS-B']})
        if isinstance(optimizer, str):
            optimizer = {'methods': [optimizer]}
        elif isinstance(optimizer, list):
            optimizer = {'methods': optimizer}
        elif not isinstance(optimizer, dict):
            raise ConfigurationError("optimizer must be string, list, or dictionary")
        
        methods = optimizer.get('methods', ['L-BFGS-B'])
        if isinstance(methods, str):
            methods = [methods]
        
        for method in methods:
            if method not in self.SUPPORTED_OPTIMIZERS:
                raise ConfigurationError(
                    f"Unsupported optimizer method '{method}'. "
                    f"Supported methods: {self.SUPPORTED_OPTIMIZERS}"
                )
        
        optimizer['methods'] = methods
        self.strategy['optimizer'] = optimizer
        
        # Validate objective function
        objective = strategy_config.get('objective_function', {'type': 'relative_error'})
        if isinstance(objective, str):
            objective = {'type': objective}
        elif not isinstance(objective, dict):
            raise ConfigurationError("objective_function must be string or dictionary")
        
        obj_type = objective.get('type', 'relative_error').lower()
        if obj_type not in self.SUPPORTED_OBJECTIVES:
            raise ConfigurationError(
                f"Unsupported objective function '{obj_type}'. "
                f"Supported types: {self.SUPPORTED_OBJECTIVES}"
            )
        
        objective['type'] = obj_type
        self.strategy['objective_function'] = objective
        
        # Validate stability control
        stability = strategy_config.get('stability_control', 'post')
        if stability not in ['post', 'penalty', 'ignore']:
            raise ConfigurationError(
                f"Unsupported stability_control '{stability}'. "
                "Supported: 'post', 'penalty', 'ignore'"
            )
        self.strategy['stability_control'] = stability

    def _parse_optional_parameters(self) -> None:
        """Parse optional configuration parameters."""
        
        # Mullins effect parameters
        self.mullins_effect = self.raw_config.get('mullins_effect', False)
        if self.mullins_effect:
            if not isinstance(self.mullins_effect, (bool, dict)):
                raise ConfigurationError(
                    "mullins_effect must be boolean or dictionary"
                )
            
            if isinstance(self.mullins_effect, dict):
                # Validate Mullins parameters if provided
                mullins_params = ['r', 'm', 'beta']
                for param in mullins_params:
                    if param in self.mullins_effect:
                        value = self.mullins_effect[param]
                        if not isinstance(value, (int, float)) or value <= 0:
                            raise ConfigurationError(
                                f"Mullins parameter '{param}' must be positive number"
                            )
        
        # Bounds and constraints
        self.parameter_bounds = self.raw_config.get('parameter_bounds', {})
        if not isinstance(self.parameter_bounds, dict):
            raise ConfigurationError("parameter_bounds must be dictionary")
        
        # Convergence criteria
        self.convergence = self.raw_config.get('convergence', {})
        if not isinstance(self.convergence, dict):
            raise ConfigurationError("convergence must be dictionary")
        
        # Default convergence parameters
        default_convergence = {
            'max_iterations': 1000,
            'tolerance': 1e-8,
            'relative_tolerance': 1e-6
        }
        
        for key, default_value in default_convergence.items():
            if key not in self.convergence:
                self.convergence[key] = default_value

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            'name': self.model_name,
            'order': self.model_order,
            'mullins_effect': self.mullins_effect
        }

    def get_data_config(self) -> Dict[str, Any]:
        """Get experimental data configuration."""
        return self.data.copy()

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get fitting strategy configuration."""
        return self.strategy.copy()

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"FittingConfiguration("
            f"model='{self.model_name}', "
            f"order={self.model_order}, "
            f"data_types={list(self.data.keys())}, "
            f"mullins={self.mullins_effect})"
        )
