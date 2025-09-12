"""
Material models for the HyperFit library.

This package contains implementations of various hyperelastic material models
including Ogden, Polynomial, and Reduced Polynomial models with optional
Mullins damage effects.
"""

from .base import HyperelasticModel, MullinsModel
from .ogden import OgdenModel
from .polynomial import PolynomialModel
from .reduced_polynomial import ReducedPolynomialModel
from .mullins import MullinsEffectModel

# Model registry for factory pattern
MODEL_REGISTRY = {
    'ogden': OgdenModel,
    'polynomial': PolynomialModel,
    'reduced_polynomial': ReducedPolynomialModel,
}

def create_model(model_name: str, model_order: int, **kwargs) -> HyperelasticModel:
    """
    Factory function to create material model instances.
    
    Args:
        model_name: Name of the model ('ogden', 'polynomial', 'reduced_polynomial')
        model_order: Order/number of terms in the model
        **kwargs: Additional model-specific parameters
        
    Returns:
        Initialized material model instance
        
    Raises:
        ValueError: If model name is not supported
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(model_order, **kwargs)

__all__ = [
    'HyperelasticModel',
    'MullinsModel', 
    'OgdenModel',
    'PolynomialModel',
    'ReducedPolynomialModel',
    'MullinsEffectModel',
    'create_model',
    'MODEL_REGISTRY'
]
