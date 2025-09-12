"""
Data loading and preprocessing for the HyperFit library.

This module handles the preparation of experimental data for fitting,
including validation, preprocessing, and conversion to internal formats.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .exceptions import DataError


class ExperimentalData:
    """
    Container for experimental data with preprocessing capabilities.
    
    This class stores and preprocesses experimental data for different loading types,
    ensuring data quality and consistency for the fitting process.
    """
    
    def __init__(self, loading_type: str, raw_data: Dict[str, Any]):
        """
        Initialize experimental data for a specific loading type.
        
        Args:
            loading_type: Type of mechanical loading ('uniaxial', 'biaxial', 'planar', 'volumetric')
            raw_data: Dictionary containing strain/stress or pressure/j data
            
        Raises:
            DataError: If data is invalid or inconsistent
        """
        self.loading_type = loading_type
        self.raw_data = raw_data.copy()
        
        # Process and validate data
        self._process_data()
        self._validate_data_quality()

    def _process_data(self) -> None:
        """Process and standardize the experimental data."""
        
        if self.loading_type == 'volumetric':
            # Volumetric data: pressure vs volume ratio
            self.pressure = np.asarray(self.raw_data['pressure'], dtype=float)
            self.volume_ratio = np.asarray(self.raw_data['j'], dtype=float)
            
            # Validate physical constraints
            if np.any(self.volume_ratio <= 0):
                raise DataError("Volume ratio (J) must be positive")
            
            if np.any(self.pressure < 0):
                raise DataError("Pressure values cannot be negative")
                
            self.x_data = self.volume_ratio
            self.y_data = self.pressure
            
        else:
            # Mechanical testing data: strain vs stress
            self.strain = np.asarray(self.raw_data['strain'], dtype=float)
            self.stress = np.asarray(self.raw_data['stress'], dtype=float)
            
            # Convert to principal stretch if needed
            self.stretch = self.strain + 1.0
            
            # Validate physical constraints
            if np.any(self.stretch <= 0):
                raise DataError(f"Principal stretch must be positive (strain > -1)")
            
            if np.any(self.stress < 0):
                raise DataError("Stress values cannot be negative for hyperelastic materials")
                
            self.x_data = self.strain  # Keep strain as primary x-axis
            self.y_data = self.stress

    def _validate_data_quality(self) -> None:
        """Validate data quality and detect potential issues."""
        
        # Check for NaN or infinite values
        if np.any(np.isnan(self.x_data)) or np.any(np.isnan(self.y_data)):
            raise DataError(f"NaN values detected in {self.loading_type} data")
        
        if np.any(np.isinf(self.x_data)) or np.any(np.isinf(self.y_data)):
            raise DataError(f"Infinite values detected in {self.loading_type} data")
        
        # Check data length
        if len(self.x_data) < 3:
            raise DataError(f"Insufficient data points for {self.loading_type} (minimum 3 required)")
        
        # Check for monotonicity in loading (should generally increase)
        if self.loading_type != 'volumetric':
            # For mechanical tests, strain should generally increase
            if not np.all(np.diff(self.x_data) >= 0):
                # Allow some tolerance for experimental noise
                decreasing_points = np.sum(np.diff(self.x_data) < 0)
                if decreasing_points > len(self.x_data) * 0.1:  # More than 10% decreasing
                    raise DataError(
                        f"Non-monotonic strain data in {self.loading_type} test. "
                        "Consider sorting or cleaning the data."
                    )
        
        # Check for duplicate data points
        if self.loading_type != 'volumetric':
            unique_strains = np.unique(self.x_data)
            if len(unique_strains) < len(self.x_data) * 0.95:  # Less than 95% unique
                raise DataError(
                    f"Too many duplicate strain values in {self.loading_type} data. "
                    "Consider removing duplicates."
                )
        
        # Validate stress-strain relationship makes physical sense
        if self.loading_type != 'volumetric':
            self._validate_stress_strain_relationship()

    def _validate_stress_strain_relationship(self) -> None:
        """Validate that stress-strain relationship is physically reasonable."""
        
        # Check that stress generally increases with strain (for hyperelastic materials)
        if len(self.y_data) > 1:
            stress_increases = np.sum(np.diff(self.y_data) > 0)
            total_intervals = len(self.y_data) - 1
            
            if stress_increases < total_intervals * 0.7:  # Less than 70% increasing
                # This might be acceptable for some loading/unloading cycles
                # Just issue a warning rather than error
                pass
        
        # Check for reasonable stress magnitudes
        max_stress = np.max(self.y_data)
        if max_stress > 1e9:  # > 1 GPa, might be unit issue
            raise DataError(
                f"Very high stress values detected ({max_stress:.2e}). "
                "Please check units (should be in Pa or similar base units)."
            )

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the experimental data."""
        if self.loading_type == 'volumetric':
            return {
                'loading_type': self.loading_type,
                'num_points': len(self.x_data),
                'volume_ratio_range': (float(np.min(self.x_data)), float(np.max(self.x_data))),
                'pressure_range': (float(np.min(self.y_data)), float(np.max(self.y_data))),
            }
        else:
            return {
                'loading_type': self.loading_type,
                'num_points': len(self.x_data),
                'strain_range': (float(np.min(self.x_data)), float(np.max(self.x_data))),
                'stress_range': (float(np.min(self.y_data)), float(np.max(self.y_data))),
                'stretch_range': (float(np.min(self.stretch)), float(np.max(self.stretch))),
            }

    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.x_data)

    def __repr__(self) -> str:
        """String representation of experimental data."""
        return f"ExperimentalData({self.loading_type}, {len(self)} points)"


def prepare_experimental_data(data_config: Dict[str, Any]) -> Dict[str, ExperimentalData]:
    """
    Prepare experimental data from configuration.
    
    Args:
        data_config: Dictionary containing experimental data for different loading types
        
    Returns:
        Dictionary mapping loading types to ExperimentalData objects
        
    Raises:
        DataError: If data preparation fails
    """
    prepared_data = {}
    
    for loading_type, raw_data in data_config.items():
        try:
            exp_data = ExperimentalData(loading_type, raw_data)
            prepared_data[loading_type] = exp_data
        except Exception as e:
            raise DataError(
                f"Failed to prepare {loading_type} data: {str(e)}"
            ) from e
    
    # Validate that we have compatible data types
    _validate_data_compatibility(prepared_data)
    
    return prepared_data


def _validate_data_compatibility(data: Dict[str, ExperimentalData]) -> None:
    """
    Validate that the combination of data types is suitable for fitting.
    
    Args:
        data: Dictionary of prepared experimental data
        
    Raises:
        DataError: If data combination is incompatible
    """
    loading_types = set(data.keys())
    
    # Check for minimum data requirements
    mechanical_types = {'uniaxial', 'biaxial', 'planar'}
    has_mechanical = bool(loading_types & mechanical_types)
    has_volumetric = 'volumetric' in loading_types
    
    if not has_mechanical:
        raise DataError(
            "At least one mechanical test type (uniaxial, biaxial, or planar) is required"
        )
    
    # Warn about recommended combinations
    if has_mechanical and not has_volumetric:
        # This is OK, but volumetric data helps with compressibility
        pass
    
    # Check data quantity
    total_points = sum(len(exp_data) for exp_data in data.values())
    if total_points < 10:
        raise DataError(
            f"Insufficient total data points ({total_points}). "
            "Recommend at least 10 points across all tests for stable fitting."
        )
    
    # Validate strain/stress ranges are reasonable for hyperelastic fitting
    mechanical_data = [data[lt] for lt in loading_types & mechanical_types]
    if mechanical_data:
        max_strain = max(np.max(exp_data.strain) for exp_data in mechanical_data)
        if max_strain < 0.1:
            raise DataError(
                f"Maximum strain ({max_strain:.3f}) is very small. "
                "Hyperelastic models typically require larger deformations (>10% strain)."
            )


def create_weight_vector(data: Dict[str, ExperimentalData], 
                        weighting_scheme: str = 'relative') -> np.ndarray:
    """
    Create weight vector for weighted least squares fitting.
    
    Args:
        data: Dictionary of experimental data
        weighting_scheme: Weighting scheme ('uniform', 'relative', 'custom')
        
    Returns:
        Weight vector for all data points
    """
    weights = []
    
    for loading_type, exp_data in data.items():
        if weighting_scheme == 'uniform':
            # Equal weights for all points
            point_weights = np.ones(len(exp_data))
            
        elif weighting_scheme == 'relative':
            # Weight by inverse square of response (stress/pressure)
            # This approximates minimizing relative error
            y_values = exp_data.y_data
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6 * (np.abs(y_values) + 1e-9)
            point_weights = 1.0 / (y_values + epsilon)**2
            
        else:
            raise DataError(f"Unsupported weighting scheme: {weighting_scheme}")
        
        weights.extend(point_weights)
    
    return np.array(weights)
