/**
 * Example C++ application using HyperFit library.
 * 
 * This demonstrates how to use the HyperFit C++ bindings in a real
 * C++ application for fitting hyperelastic material models.
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

// Forward declarations of HyperFit functions
// (These would normally be in a header file)
std::map<std::string, py::object> fit_material_with_arrays(
    const std::string& model_name,
    int model_order,
    const std::vector<double>& uniaxial_strain,
    const std::vector<double>& uniaxial_stress,
    const std::vector<double>& biaxial_strain,
    const std::vector<double>& biaxial_stress,
    const std::vector<double>& planar_strain,
    const std::vector<double>& planar_stress,
    const std::vector<double>& volumetric_j,
    const std::vector<double>& volumetric_pressure,
    const std::string& initial_guess_method,
    const std::vector<std::string>& optimizer_methods,
    const std::string& objective_type
);

std::map<std::string, std::vector<double>> extract_parameters(
    const std::map<std::string, py::object>& result
);

bool is_fit_successful(const std::map<std::string, py::object>& result);
std::string get_error_message(const std::map<std::string, py::object>& result);

int main() {
    // Initialize Python interpreter
    py::scoped_interpreter guard{};
    
    try {
        std::cout << "HyperFit C++ Example Application" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // Example experimental data (uniaxial tension test)
        std::vector<double> uniaxial_strain = {
            0.1338, 0.2675, 0.3567, 0.6242, 0.8917, 1.1592, 1.4268, 2.0510,
            2.5860, 3.0318, 3.7898, 4.3694, 4.8153, 5.1720, 5.4395, 5.7070
        };
        
        std::vector<double> uniaxial_stress = {
            1.5506E5, 2.4367E5, 3.1013E5, 4.2089E5, 5.3165E5, 5.9810E5, 6.8671E5,
            8.8608E5, 1.06329E6, 1.24051E6, 1.61709E6, 1.99367E6, 2.34810E6,
            2.74684E6, 3.10127E6, 3.45570E6
        };
        
        std::cout << "Loaded experimental data:" << std::endl;
        std::cout << "  Uniaxial points: " << uniaxial_strain.size() << std::endl;
        
        // Fit Reduced Polynomial model (N=3)
        std::cout << "\nFitting Reduced Polynomial model (N=3)..." << std::endl;
        
        auto result = fit_material_with_arrays(
            "reduced_polynomial",    // model name
            3,                      // model order
            uniaxial_strain,        // uniaxial strain data
            uniaxial_stress,        // uniaxial stress data
            {},                     // no biaxial data
            {},
            {},                     // no planar data  
            {},
            {},                     // no volumetric data
            {},
            "lls",                  // linear least squares initial guess
            {"L-BFGS-B", "TNC"},   // optimization methods
            "relative_error"        // objective function
        );
        
        // Check results
        if (is_fit_successful(result)) {
            std::cout << "Fitting successful!" << std::endl;
            
            // Extract fitted parameters
            auto parameters = extract_parameters(result);
            
            std::cout << "\nFitted parameters:" << std::endl;
            for (const auto& param : parameters) {
                std::cout << "  " << param.first << ": ";
                if (param.second.size() == 1) {
                    std::cout << param.second[0] << std::endl;
                } else {
                    std::cout << "[";
                    for (size_t i = 0; i < param.second.size(); ++i) {
                        std::cout << param.second[i];
                        if (i < param.second.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
            
            // Extract quality metrics
            try {
                if (result.find("diagnostics") != result.end()) {
                    py::dict diagnostics = result.at("diagnostics").cast<py::dict>();
                    
                    std::cout << "\nFitting quality:" << std::endl;
                    
                    if (diagnostics.contains("rms_error")) {
                        double rms = diagnostics["rms_error"].cast<double>();
                        std::cout << "  RMS Error: " << rms << std::endl;
                    }
                    
                    if (diagnostics.contains("r_squared")) {
                        double r2 = diagnostics["r_squared"].cast<double>();
                        std::cout << "  R-squared: " << r2 << std::endl;
                    }
                }
                
                if (result.find("fitting_time") != result.end()) {
                    double time = result.at("fitting_time").cast<double>();
                    std::cout << "  Fitting time: " << time << " seconds" << std::endl;
                }
                
            } catch (...) {
                std::cout << "  (Could not extract quality metrics)" << std::endl;
            }
            
        } else {
            std::cout << "Fitting failed!" << std::endl;
            std::cout << "Error: " << get_error_message(result) << std::endl;
        }
        
        // Example of fitting Ogden model
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "Fitting Ogden model (N=2)..." << std::endl;
        
        auto ogden_result = fit_material_with_arrays(
            "ogden",                // model name
            2,                      // model order (2 pairs)
            uniaxial_strain,        // uniaxial data
            uniaxial_stress,
            {},                     // no other data
            {},
            {},
            {},
            {},
            {},
            "heuristic",            // heuristic initial guess
            {"L-BFGS-B"},          // optimization method
            "absolute_error"        // objective function
        );
        
        if (is_fit_successful(ogden_result)) {
            std::cout << "Ogden fitting successful!" << std::endl;
            
            auto ogden_params = extract_parameters(ogden_result);
            std::cout << "\nOgden parameters:" << std::endl;
            for (const auto& param : ogden_params) {
                std::cout << "  " << param.first << ": ";
                if (param.second.size() == 1) {
                    std::cout << param.second[0] << std::endl;
                } else {
                    std::cout << "[";
                    for (size_t i = 0; i < param.second.size(); ++i) {
                        std::cout << param.second[i];
                        if (i < param.second.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
        } else {
            std::cout << "Ogden fitting failed: " << get_error_message(ogden_result) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nExample completed successfully!" << std::endl;
    return 0;
}
