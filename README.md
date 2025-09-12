# HyperFit

A comprehensive Python library for fitting hyperelastic and Mullins effect material models to experimental data.

## Overview

HyperFit is a configuration-driven library designed for both direct Python usage and C++ integration via Pybind11. It provides robust, extensible fitting capabilities for hyperelastic material models commonly used in computational mechanics.

### Key Features

- **Configuration-Driven**: Complete fitting process controlled by a single configuration dictionary
- **Modular Architecture**: Easy to extend with new material models and optimization strategies  
- **Dual-Use API**: Clean Python API with C++ bindings for integration
- **Multiple Models**: Support for Ogden, Polynomial, and Reduced Polynomial models
- **Mullins Effect**: Optional stress softening for filled elastomers
- **Robust Optimization**: Multiple optimization algorithms with stability controls
- **Quality Metrics**: Comprehensive fitting diagnostics and quality assessment

### Supported Models

- **Ogden Model**: W = Σᵢ (2μᵢ/αᵢ²) * (λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3)
- **Polynomial Model**: W = Σᵢⱼ C_ij * (I₁ - 3)^i * (I₂ - 3)^j  
- **Reduced Polynomial Model**: W = Σᵢ C_i0 * (I₁ - 3)^i
- **Mullins Effect**: η = 1 - erf((W_max - W) / (m + β * W_max)) / r

## Installation

### Python Installation

```bash
# Install from source
git clone https://github.com/hyperfit/hyperfit.git
cd hyperfit
pip install -e .

# Or install from PyPI (when available)
pip install hyperfit
```

### C++ Bindings

```bash
# Build C++ bindings
cd cpp_bindings
python setup.py build_ext --inplace

# Or use CMake integration (see documentation)
```

## Quick Start

### Python Usage

```python
import hyperfit
import numpy as np

# Define experimental data
uniaxial_strain = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
uniaxial_stress = np.array([100e3, 180e3, 240e3, 280e3, 300e3])

# Configuration dictionary
config = {
    "model": "reduced_polynomial",
    "model_order": 3,
    "experimental_data": {
        "uniaxial": {
            "strain": uniaxial_strain,
            "stress": uniaxial_stress,
        }
    },
    "fitting_strategy": {
        "initial_guess": {"method": "lls"},
        "optimizer": {"methods": ["L-BFGS-B"]},
        "objective_function": {"type": "relative_error"}
    }
}

# Perform fitting
result = hyperfit.fit(config)

if result['success']:
    print("Fitted parameters:", result['parameters'])
    print("RMS error:", result['diagnostics']['rms_error'])
else:
    print("Fitting failed:", result['error'])
```

### C++ Usage

```cpp
#include "hyperfit_cpp.hpp"
#include <iostream>
#include <vector>

int main() {
    // Experimental data
    std::vector<double> strain = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> stress = {100e3, 180e3, 240e3, 280e3, 300e3};
    
    // Fit model
    auto result = fit_material_with_arrays(
        "reduced_polynomial", 3,    // model and order
        strain, stress,             // uniaxial data
        {}, {},                     // no biaxial data
        {}, {},                     // no planar data
        {}, {},                     // no volumetric data
        "lls",                      // initial guess method
        {"L-BFGS-B"},              // optimizer
        "relative_error"            // objective
    );
    
    if (is_fit_successful(result)) {
        auto params = extract_parameters(result);
        std::cout << "Fitting successful!" << std::endl;
        // Use fitted parameters...
    } else {
        std::cout << "Error: " << get_error_message(result) << std::endl;
    }
    
    return 0;
}
```

## Configuration Reference

### Model Configuration

```python
config = {
    "model": "reduced_polynomial",  # "ogden", "polynomial", "reduced_polynomial"
    "model_order": 3,               # Number of terms/pairs
    "mullins_effect": False,        # Enable Mullins damage (optional)
}
```

### Experimental Data

```python
"experimental_data": {
    "uniaxial": {                   # Required: at least one mechanical test
        "strain": strain_array,     # Engineering strain
        "stress": stress_array      # Nominal stress (Pa)
    },
    "biaxial": {                    # Optional
        "strain": strain_array,
        "stress": stress_array
    },
    "planar": {                     # Optional
        "strain": strain_array,
        "stress": stress_array
    },
    "volumetric": {                 # Optional: for compressible materials
        "j": volume_ratio_array,    # Volume ratio J = V/V₀
        "pressure": pressure_array  # Hydrostatic pressure (Pa)
    }
}
```

### Fitting Strategy

```python
"fitting_strategy": {
    "initial_guess": {
        "method": "lls",            # "lls" or "heuristic"
        "alpha_guesses": [1, 2, 3]  # For Ogden LLS (optional)
    },
    "optimizer": {
        "methods": ["L-BFGS-B", "TNC"]  # Try multiple methods
    },
    "objective_function": {
        "type": "relative_error",   # "absolute_error", "relative_error", "stress", "eta"
        "weights": {...}            # Optional data weighting
    },
    "stability_control": "post"     # "post", "penalty", "ignore"
}
```

## Advanced Features

### Mullins Effect Fitting

```python
config = {
    "model": "reduced_polynomial",
    "model_order": 3,
    "mullins_effect": {
        "r": 2.0,      # Initial guess (optional)
        "m": 25.0,     # Initial guess (optional) 
        "beta": 0.1    # Initial guess (optional)
    },
    "experimental_data": {
        # Include loading/unloading cycle data
        "uniaxial": {"strain": [...], "stress": [...]}
    },
    "fitting_strategy": {
        "objective_function": {"type": "stress"}  # Use stress objective for Mullins
    }
}
```

### Parameter Bounds and Constraints

```python
config = {
    # ... other config ...
    "parameter_bounds": {
        "C_10": (1e-6, 1e6),       # Bounds for specific parameters
        "mu_1": (-1e5, 1e5)
    },
    "convergence": {
        "max_iterations": 1000,
        "tolerance": 1e-8,
        "relative_tolerance": 1e-6
    }
}
```

## Model Details

### Reduced Polynomial Model

The Reduced Polynomial (Mooney-Rivlin) model is ideal for moderate deformations:

- **Strain Energy**: W = Σᵢ C_i0 * (I₁ - 3)^i + Σᵢ (1/D_i) * (J - 1)^(2i)
- **Parameters**: C_i0 (deviatoric), D_i (volumetric)
- **Recommended Order**: N = 2-3 for most materials
- **Use Cases**: General hyperelastic materials, moderate strains

### Ogden Model

The Ogden model provides excellent flexibility for large deformations:

- **Strain Energy**: W = Σᵢ (2μᵢ/αᵢ²) * (λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3)
- **Parameters**: μᵢ, αᵢ (material constants)
- **Recommended Order**: N = 2-3 pairs
- **Use Cases**: Large deformations, biological tissues, rubber

### Polynomial Model

The full Polynomial model includes both I₁ and I₂ dependence:

- **Strain Energy**: W = Σᵢⱼ C_ij * (I₁ - 3)^i * (I₂ - 3)^j
- **Parameters**: C_ij (material constants)
- **Recommended Order**: N = 2 for stability
- **Use Cases**: When I₂ dependence is significant

## Best Practices

### Data Requirements

- **Minimum Data**: At least one mechanical test (uniaxial, biaxial, or planar)
- **Recommended**: Multiple test types for better parameter identification
- **Data Quality**: Ensure monotonic loading, remove noise, check units
- **Strain Range**: Include sufficient deformation (>10% strain recommended)

### Model Selection

- **Start Simple**: Try Reduced Polynomial N=2 first
- **Add Complexity**: Increase order or try Ogden if needed
- **Validate**: Check physical reasonableness of parameters
- **Cross-Validate**: Test predictions on independent data

### Optimization Strategy

- **Initial Guess**: LLS generally more robust than heuristic
- **Multiple Methods**: Try L-BFGS-B, TNC, and trust-constr
- **Stability Control**: Use "post" processing for Ogden model
- **Convergence**: Monitor fitting diagnostics and quality metrics

## API Reference

### Main Functions

- `hyperfit.fit(config)`: Main fitting function
- `hyperfit.HyperFitError`: Base exception class
- `hyperfit.ConfigurationError`: Configuration validation errors

### C++ Bindings

- `fit_material(config)`: Main C++ fitting function
- `fit_material_with_arrays(...)`: Convenience function with arrays
- `extract_parameters(result)`: Extract fitted parameters
- `is_fit_successful(result)`: Check fitting success
- `get_error_message(result)`: Get error description

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/hyperfit/hyperfit.git
cd hyperfit
pip install -e ".[dev]"
pytest tests/
```

### Adding New Models

1. Inherit from `HyperelasticModel` base class
2. Implement required abstract methods
3. Add to model registry
4. Include tests and documentation

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use HyperFit in your research, please cite:

```bibtex
@software{hyperfit,
  title = {HyperFit: A Python Library for Hyperelastic Material Model Fitting},
  author = {Xiaotong Wang},
  year = {2024},
  url = {https://github.com/hyperfit/hyperfit}
}
```

## Support

- **Documentation**: [https://hyperfit.readthedocs.io](https://hyperfit.readthedocs.io)
- **Issues**: [https://github.com/hyperfit/hyperfit/issues](https://github.com/hyperfit/hyperfit/issues)
- **Discussions**: [https://github.com/hyperfit/hyperfit/discussions](https://github.com/hyperfit/hyperfit/discussions)

## Acknowledgments

This library builds upon established hyperelastic theory and incorporates algorithms validated against commercial finite element software. Special thanks to the computational mechanics community for their foundational work in this field.
