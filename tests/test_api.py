"""
Tests for the HyperFit API module.
"""

import pytest
import numpy as np
from hyperfit import fit
from hyperfit.exceptions import ConfigurationError, DataError


class TestAPIBasic:
    """Basic API functionality tests."""
    
    def test_simple_reduced_polynomial_fit(self):
        """Test basic Reduced Polynomial fitting."""
        # Simple synthetic data
        strain = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        stress = np.array([100e3, 180e3, 240e3, 280e3, 300e3])
        
        config = {
            "model": "reduced_polynomial",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": strain,
                    "stress": stress,
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        # Check that fitting completed
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'parameters' in result or 'error' in result
        
        # If successful, check parameter structure
        if result['success']:
            assert 'parameters' in result
            assert 'diagnostics' in result
            params = result['parameters']
            assert isinstance(params, dict)
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model name."""
        config = {
            "model": "invalid_model",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": np.array([0.1, 0.2]),
                    "stress": np.array([100e3, 180e3]),
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_missing_required_config(self):
        """Test error handling for missing required configuration."""
        config = {
            "model": "reduced_polynomial",
            # Missing model_order, experimental_data, fitting_strategy
        }
        
        result = fit(config)
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_empty_experimental_data(self):
        """Test error handling for empty experimental data."""
        config = {
            "model": "reduced_polynomial",
            "model_order": 2,
            "experimental_data": {},  # Empty data
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        assert result['success'] is False
        assert 'error' in result


class TestAPIConfiguration:
    """Test configuration validation and handling."""
    
    def test_default_values(self):
        """Test that default values are applied correctly."""
        config = {
            "model": "reduced_polynomial",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": np.array([0.1, 0.2, 0.3]),
                    "stress": np.array([100e3, 180e3, 240e3]),
                }
            },
            # Minimal fitting strategy - should get defaults
            "fitting_strategy": {}
        }
        
        result = fit(config)
        
        # Should not fail due to missing strategy details
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_multiple_data_types(self):
        """Test fitting with multiple experimental data types."""
        strain = np.array([0.1, 0.2, 0.3])
        stress_u = np.array([100e3, 180e3, 240e3])
        stress_b = np.array([120e3, 200e3, 260e3])
        
        config = {
            "model": "reduced_polynomial",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": strain,
                    "stress": stress_u,
                },
                "biaxial": {
                    "strain": strain,
                    "stress": stress_b,
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        assert isinstance(result, dict)
        assert 'success' in result


class TestAPIModels:
    """Test different material models through the API."""
    
    def test_ogden_model(self):
        """Test Ogden model fitting."""
        strain = np.array([0.1, 0.2, 0.3, 0.4])
        stress = np.array([100e3, 180e3, 240e3, 280e3])
        
        config = {
            "model": "ogden",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": strain,
                    "stress": stress,
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "heuristic"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "absolute_error"}
            }
        }
        
        result = fit(config)
        
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_polynomial_model(self):
        """Test Polynomial model fitting."""
        strain = np.array([0.1, 0.2, 0.3, 0.4])
        stress = np.array([100e3, 180e3, 240e3, 280e3])
        
        config = {
            "model": "polynomial",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": strain,
                    "stress": stress,
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        assert isinstance(result, dict)
        assert 'success' in result


class TestAPIErrorHandling:
    """Test comprehensive error handling."""
    
    def test_nan_in_data(self):
        """Test handling of NaN values in experimental data."""
        strain = np.array([0.1, 0.2, np.nan, 0.4])
        stress = np.array([100e3, 180e3, 240e3, 280e3])
        
        config = {
            "model": "reduced_polynomial",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": strain,
                    "stress": stress,
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_mismatched_data_lengths(self):
        """Test handling of mismatched data array lengths."""
        strain = np.array([0.1, 0.2, 0.3])
        stress = np.array([100e3, 180e3])  # Different length
        
        config = {
            "model": "reduced_polynomial",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": strain,
                    "stress": stress,
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_negative_stress(self):
        """Test handling of negative stress values."""
        strain = np.array([0.1, 0.2, 0.3])
        stress = np.array([100e3, -180e3, 240e3])  # Negative stress
        
        config = {
            "model": "reduced_polynomial",
            "model_order": 2,
            "experimental_data": {
                "uniaxial": {
                    "strain": strain,
                    "stress": stress,
                }
            },
            "fitting_strategy": {
                "initial_guess": {"method": "lls"},
                "optimizer": {"methods": ["L-BFGS-B"]},
                "objective_function": {"type": "relative_error"}
            }
        }
        
        result = fit(config)
        
        assert result['success'] is False
        assert 'error' in result


if __name__ == "__main__":
    pytest.main([__file__])
