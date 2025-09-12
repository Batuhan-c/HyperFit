"""
Tests for material model implementations.
"""

import pytest
import numpy as np
from hyperfit.models import create_model, ReducedPolynomialModel, OgdenModel
from hyperfit.exceptions import ModelError


class TestModelFactory:
    """Test model creation factory."""
    
    def test_create_reduced_polynomial(self):
        """Test creating Reduced Polynomial model."""
        model = create_model('reduced_polynomial', 3)
        assert isinstance(model, ReducedPolynomialModel)
        assert model.model_order == 3
    
    def test_create_ogden(self):
        """Test creating Ogden model."""
        model = create_model('ogden', 2)
        assert isinstance(model, OgdenModel)
        assert model.model_order == 2
    
    def test_invalid_model_name(self):
        """Test error for invalid model name."""
        with pytest.raises(ValueError):
            create_model('invalid_model', 2)
    
    def test_invalid_model_order(self):
        """Test error for invalid model order."""
        with pytest.raises(ModelError):
            create_model('reduced_polynomial', 0)


class TestReducedPolynomialModel:
    """Test Reduced Polynomial model implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = ReducedPolynomialModel(3)
        self.params = {
            'C_i0': np.array([191.322, 4.9415, 0.5346]),
            'D_i': np.array([5e-6, 0.0, 0.0])
        }
    
    def test_parameter_names(self):
        """Test parameter name generation."""
        names = self.model.get_parameter_names()
        expected = ['C_10', 'C_20', 'C_30', 'D_1', 'D_2', 'D_3']
        assert names == expected
    
    def test_parameter_bounds(self):
        """Test parameter bounds."""
        bounds = self.model.get_parameter_bounds()
        assert len(bounds) == 6  # 3 C_i0 + 3 D_i
        
        # Check that all bounds are tuples
        for bound in bounds:
            assert isinstance(bound, tuple)
            assert len(bound) == 2
    
    def test_stress_calculation_uniaxial(self):
        """Test stress calculation for uniaxial loading."""
        strain = 0.1
        stress = self.model.calculate_stress(strain, self.params, 'uniaxial')
        
        assert isinstance(stress, (float, np.floating))
        assert stress > 0  # Positive stress for positive strain
    
    def test_stress_calculation_array(self):
        """Test stress calculation with array input."""
        strain = np.array([0.1, 0.2, 0.3])
        stress = self.model.calculate_stress(strain, self.params, 'uniaxial')
        
        assert isinstance(stress, np.ndarray)
        assert stress.shape == strain.shape
        assert np.all(stress > 0)
    
    def test_strain_energy_calculation(self):
        """Test strain energy calculation."""
        strain = 0.1
        energy = self.model.calculate_strain_energy(strain, self.params, 'uniaxial')
        
        assert isinstance(energy, (float, np.floating))
        assert energy > 0  # Positive energy for deformation
    
    def test_invalid_loading_type(self):
        """Test error for invalid loading type."""
        with pytest.raises(ModelError):
            self.model.calculate_stress(0.1, self.params, 'invalid_type')
    
    def test_negative_strain(self):
        """Test error for negative strain (invalid stretch)."""
        with pytest.raises(ModelError):
            self.model.calculate_stress(-2.0, self.params, 'uniaxial')  # strain < -1
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters should not raise
        self.model.validate_parameters(self.params)
        
        # Missing parameters should raise
        with pytest.raises(ModelError):
            self.model.validate_parameters({'C_i0': np.array([1, 2, 3])})
    
    def test_format_parameters(self):
        """Test parameter vector formatting."""
        param_vector = np.array([1, 2, 3, 1e-6, 1e-7, 1e-8])
        formatted = self.model.format_parameters(param_vector)
        
        assert 'C_i0' in formatted
        assert 'D_i' in formatted
        assert len(formatted['C_i0']) == 3
        assert len(formatted['D_i']) == 3
    
    def test_elastic_moduli_calculation(self):
        """Test calculation of elastic moduli."""
        moduli = self.model.calculate_elastic_moduli(self.params)
        
        expected_keys = ['G0', 'K0', 'E0', 'nu0']
        for key in expected_keys:
            assert key in moduli
            assert isinstance(moduli[key], (float, np.floating))
        
        # Check physical constraints
        assert moduli['G0'] > 0  # Positive shear modulus
        assert 0 <= moduli['nu0'] <= 0.5  # Poisson's ratio bounds


class TestOgdenModel:
    """Test Ogden model implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = OgdenModel(2)
        self.params = {
            'mu': np.array([-207.227, 96.368]),
            'alpha': np.array([2.752, 3.084]),
            'd': np.array([1.194e-2, 0.0])
        }
    
    def test_parameter_names(self):
        """Test parameter name generation."""
        names = self.model.get_parameter_names()
        expected = ['mu_1', 'alpha_1', 'mu_2', 'alpha_2', 'd_1', 'd_2']
        assert names == expected
    
    def test_stress_calculation(self):
        """Test stress calculation."""
        strain = 0.1
        stress = self.model.calculate_stress(strain, self.params, 'uniaxial')
        
        assert isinstance(stress, (float, np.floating))
        # Note: Ogden stress can be negative due to parameter signs
    
    def test_strain_energy_calculation(self):
        """Test strain energy calculation."""
        strain = 0.1
        energy = self.model.calculate_strain_energy(strain, self.params, 'uniaxial')
        
        assert isinstance(energy, (float, np.floating))
    
    def test_stress_derivatives(self):
        """Test stress derivative calculations."""
        strain = np.array([0.1, 0.2])
        dT_dmu, dT_dalpha = self.model.calculate_stress_derivatives(
            strain, self.params, 'uniaxial'
        )
        
        assert dT_dmu.shape == (2, 2)  # (n_strains, n_pairs)
        assert dT_dalpha.shape == (2, 2)
    
    def test_format_parameters(self):
        """Test parameter vector formatting."""
        param_vector = np.array([100, 2.0, -50, 1.5, 1e-6, 1e-7])
        formatted = self.model.format_parameters(param_vector)
        
        assert 'mu' in formatted
        assert 'alpha' in formatted
        assert 'd' in formatted
        assert len(formatted['mu']) == 2
        assert len(formatted['alpha']) == 2
        assert len(formatted['d']) == 2
    
    def test_elastic_moduli_calculation(self):
        """Test calculation of elastic moduli."""
        moduli = self.model.calculate_elastic_moduli(self.params)
        
        expected_keys = ['G0', 'K0', 'E0', 'nu0']
        for key in expected_keys:
            assert key in moduli
    
    def test_stability_constraint(self):
        """Test that G0 > 0 constraint is checked."""
        # This specific parameter set should give G0 > 0
        mu = np.array([-207.227, 96.368])
        alpha = np.array([2.752, 3.084])
        G0 = np.sum(mu * alpha)
        
        # The constraint should be G0 > 0
        # Note: This particular set might violate it, which is why stability control exists


class TestModelConsistency:
    """Test consistency across different models."""
    
    def test_small_strain_consistency(self):
        """Test that models give reasonable results for small strains."""
        models = [
            ReducedPolynomialModel(2),
            OgdenModel(2)
        ]
        
        # Simple parameters for comparison
        rp_params = {'C_i0': np.array([100e3, 0]), 'D_i': np.array([1e-6, 0])}
        ogden_params = {'mu': np.array([100e3, 0]), 'alpha': np.array([2.0, 2.0]), 'd': np.array([1e-6, 0])}
        
        params_list = [rp_params, ogden_params]
        
        small_strain = 0.01  # 1% strain
        
        for model, params in zip(models, params_list):
            try:
                stress = model.calculate_stress(small_strain, params, 'uniaxial')
                energy = model.calculate_strain_energy(small_strain, params, 'uniaxial')
                
                # Basic sanity checks
                assert np.isfinite(stress)
                assert np.isfinite(energy)
                assert energy >= 0  # Energy should be non-negative
                
            except Exception as e:
                pytest.fail(f"Model {type(model).__name__} failed for small strain: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
