#!/usr/bin/env python3
"""
Basic HyperFit usage example.

This script demonstrates how to use the HyperFit library to fit
hyperelastic material models to experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
import hyperfit

def main():
    print("HyperFit Basic Example")
    print("=====================")
    
    # Generate synthetic experimental data (uniaxial tension)
    print("Generating synthetic experimental data...")
    
    # True parameters for synthetic data generation
    true_C10 = 200e3  # Pa
    true_C20 = 5e3    # Pa
    
    strains = np.linspace(0.05, 0.5, 15)
    
    # Generate synthetic stress using simple Mooney-Rivlin model
    stresses = []
    for strain in strains:
        lam = 1 + strain
        I1_bar = lam**2 + 2/lam
        stress = 2 * (lam - 1/lam**2) * true_C10 + 2 * (lam - 1/lam**2) * 2 * true_C20 * (I1_bar - 3)
        # Add some noise
        stress += np.random.normal(0, 0.05 * stress)
        stresses.append(stress)
    
    stresses = np.array(stresses)
    
    print(f"Generated {len(strains)} data points")
    print(f"Strain range: {strains[0]:.3f} to {strains[-1]:.3f}")
    print(f"Stress range: {stresses[0]/1e3:.1f} to {stresses[-1]/1e3:.1f} kPa")
    
    # Configuration for Reduced Polynomial model (N=2)
    print("\nFitting Reduced Polynomial model (N=2)...")
    
    config_rp = {
        "model": "reduced_polynomial",
        "model_order": 2,
        "experimental_data": {
            "uniaxial": {
                "strain": strains,
                "stress": stresses,
            }
        },
        "fitting_strategy": {
            "initial_guess": {"method": "lls"},
            "optimizer": {"methods": ["L-BFGS-B", "TNC"]},
            "objective_function": {"type": "relative_error"}
        }
    }
    
    # Perform fitting
    result_rp = hyperfit.fit(config_rp)
    
    # Display results
    print("\nReduced Polynomial Results:")
    print("-" * 30)
    
    if result_rp['success']:
        print("✓ Fitting successful!")
        
        params = result_rp['parameters']
        print(f"  C_i0: {params['C_i0']}")
        print(f"  D_i:  {params['D_i']}")
        
        if 'diagnostics' in result_rp:
            diag = result_rp['diagnostics']
            if 'rms_error' in diag:
                print(f"  RMS Error: {diag['rms_error']:.2e}")
            if 'r_squared' in diag:
                print(f"  R-squared: {diag['r_squared']:.6f}")
        
        # Compare with true values
        fitted_C10 = params['C_i0'][0]
        fitted_C20 = params['C_i0'][1]
        print(f"  True C10: {true_C10:.0f} Pa, Fitted: {fitted_C10:.0f} Pa (Error: {abs(fitted_C10-true_C10)/true_C10*100:.1f}%)")
        print(f"  True C20: {true_C20:.0f} Pa, Fitted: {fitted_C20:.0f} Pa (Error: {abs(fitted_C20-true_C20)/true_C20*100:.1f}%)")
        
    else:
        print("✗ Fitting failed!")
        print(f"  Error: {result_rp['error']}")
    
    # Try Ogden model for comparison
    print("\nFitting Ogden model (N=2)...")
    
    config_ogden = {
        "model": "ogden",
        "model_order": 2,
        "experimental_data": {
            "uniaxial": {
                "strain": strains,
                "stress": stresses,
            }
        },
        "fitting_strategy": {
            "initial_guess": {"method": "heuristic"},
            "optimizer": {"methods": ["L-BFGS-B"]},
            "objective_function": {"type": "absolute_error"}
        }
    }
    
    result_ogden = hyperfit.fit(config_ogden)
    
    print("\nOgden Results:")
    print("-" * 15)
    
    if result_ogden['success']:
        print("✓ Fitting successful!")
        
        params = result_ogden['parameters']
        print(f"  μ:  {params['mu']}")
        print(f"  α:  {params['alpha']}")
        print(f"  d:  {params['d']}")
        
        if 'diagnostics' in result_ogden:
            diag = result_ogden['diagnostics']
            if 'rms_error' in diag:
                print(f"  RMS Error: {diag['rms_error']:.2e}")
            if 'r_squared' in diag:
                print(f"  R-squared: {diag['r_squared']:.6f}")
        
    else:
        print("✗ Fitting failed!")
        print(f"  Error: {result_ogden['error']}")
    
    # Create plots if matplotlib is available
    try:
        print("\nCreating comparison plot...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot experimental data
        plt.scatter(strains, stresses/1e3, color='black', s=50, label='Experimental Data', zorder=3)
        
        # Plot fitted curves
        strain_fine = np.linspace(0, max(strains), 100)
        
        if result_rp['success']:
            # Calculate RP model prediction
            rp_params = result_rp['parameters']
            stress_rp = []
            for strain in strain_fine:
                lam = 1 + strain
                I1_bar = lam**2 + 2/lam
                C10, C20 = rp_params['C_i0'][0], rp_params['C_i0'][1]
                stress = 2 * (lam - 1/lam**2) * (C10 + 2*C20*(I1_bar-3))
                stress_rp.append(stress)
            
            plt.plot(strain_fine, np.array(stress_rp)/1e3, 'b-', linewidth=2, 
                    label=f'Reduced Polynomial (N=2)', alpha=0.8)
        
        if result_ogden['success']:
            # Calculate Ogden model prediction (simplified for plotting)
            plt.plot([], [], 'r--', linewidth=2, label='Ogden (N=2)', alpha=0.8)
        
        plt.xlabel('Engineering Strain')
        plt.ylabel('Nominal Stress (kPa)')
        plt.title('Hyperelastic Model Fitting Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('hyperfit_example.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'hyperfit_example.png'")
        
        # Show plot if in interactive mode
        try:
            plt.show()
        except:
            pass
            
    except ImportError:
        print("Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"Plotting error: {e}")
    
    print("\nExample completed!")
    print("\nNext steps:")
    print("- Try different model orders")
    print("- Add biaxial or planar test data")
    print("- Experiment with different optimization strategies")
    print("- Check parameter physical reasonableness")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()
