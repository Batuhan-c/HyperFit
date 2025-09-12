/**
 * Pybind11 C++ bindings for the HyperFit library.
 * 
 * This file provides a clean C++ interface to the HyperFit Python library,
 * allowing seamless integration with C++ applications while leveraging
 * the full power of the Python implementation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

/**
 * C++ wrapper function for hyperelastic material fitting.
 * 
 * This function provides a clean C++ interface that can be easily called
 * from your main C++ application. It handles the conversion between C++
 * data types and Python objects.
 * 
 * @param config Configuration dictionary as std::map
 * @return Results dictionary as std::map
 */
std::map<std::string, py::object> fit_material(const std::map<std::string, py::object>& config) {
    try {
        // Import the HyperFit Python API
        py::module_ hyperfit_api = py::module_::import("hyperfit.api");
        
        // Convert C++ map to Python dict
        py::dict py_config;
        for (const auto& pair : config) {
            py_config[pair.first] = pair.second;
        }
        
        // Call the Python fit function
        py::object result = hyperfit_api.attr("fit")(py_config);
        
        // Convert result back to C++ map
        py::dict result_dict = result.cast<py::dict>();
        std::map<std::string, py::object> cpp_result;
        
        for (auto item : result_dict) {
            std::string key = item.first.cast<std::string>();
            cpp_result[key] = item.second;
        }
        
        return cpp_result;
        
    } catch (const std::exception& e) {
        // Return error result
        std::map<std::string, py::object> error_result;
        error_result["success"] = py::bool_(false);
        error_result["error"] = py::str(e.what());
        error_result["error_type"] = py::str("C++Exception");
        return error_result;
    }
}

/**
 * Convenience function for fitting with raw data arrays.
 * 
 * This function allows passing experimental data as raw C++ arrays,
 * which is often more convenient for C++ applications.
 */
std::map<std::string, py::object> fit_material_with_arrays(
    const std::string& model_name,
    int model_order,
    const std::vector<double>& uniaxial_strain,
    const std::vector<double>& uniaxial_stress,
    const std::vector<double>& biaxial_strain = {},
    const std::vector<double>& biaxial_stress = {},
    const std::vector<double>& planar_strain = {},
    const std::vector<double>& planar_stress = {},
    const std::vector<double>& volumetric_j = {},
    const std::vector<double>& volumetric_pressure = {},
    const std::string& initial_guess_method = "lls",
    const std::vector<std::string>& optimizer_methods = {"L-BFGS-B"},
    const std::string& objective_type = "relative_error"
) {
    try {
        // Build configuration dictionary
        std::map<std::string, py::object> config;
        
        // Model configuration
        config["model"] = py::str(model_name);
        config["model_order"] = py::int_(model_order);
        
        // Experimental data
        py::dict experimental_data;
        
        // Add uniaxial data (required)
        if (!uniaxial_strain.empty() && !uniaxial_stress.empty()) {
            py::dict uniaxial_data;
            uniaxial_data["strain"] = py::array_t<double>(
                uniaxial_strain.size(), uniaxial_strain.data()
            );
            uniaxial_data["stress"] = py::array_t<double>(
                uniaxial_stress.size(), uniaxial_stress.data()
            );
            experimental_data["uniaxial"] = uniaxial_data;
        }
        
        // Add biaxial data (optional)
        if (!biaxial_strain.empty() && !biaxial_stress.empty()) {
            py::dict biaxial_data;
            biaxial_data["strain"] = py::array_t<double>(
                biaxial_strain.size(), biaxial_strain.data()
            );
            biaxial_data["stress"] = py::array_t<double>(
                biaxial_stress.size(), biaxial_stress.data()
            );
            experimental_data["biaxial"] = biaxial_data;
        }
        
        // Add planar data (optional)
        if (!planar_strain.empty() && !planar_stress.empty()) {
            py::dict planar_data;
            planar_data["strain"] = py::array_t<double>(
                planar_strain.size(), planar_strain.data()
            );
            planar_data["stress"] = py::array_t<double>(
                planar_stress.size(), planar_stress.data()
            );
            experimental_data["planar"] = planar_data;
        }
        
        // Add volumetric data (optional)
        if (!volumetric_j.empty() && !volumetric_pressure.empty()) {
            py::dict volumetric_data;
            volumetric_data["j"] = py::array_t<double>(
                volumetric_j.size(), volumetric_j.data()
            );
            volumetric_data["pressure"] = py::array_t<double>(
                volumetric_pressure.size(), volumetric_pressure.data()
            );
            experimental_data["volumetric"] = volumetric_data;
        }
        
        config["experimental_data"] = experimental_data;
        
        // Fitting strategy
        py::dict fitting_strategy;
        
        py::dict initial_guess;
        initial_guess["method"] = py::str(initial_guess_method);
        fitting_strategy["initial_guess"] = initial_guess;
        
        py::dict optimizer;
        py::list methods_list;
        for (const auto& method : optimizer_methods) {
            methods_list.append(py::str(method));
        }
        optimizer["methods"] = methods_list;
        fitting_strategy["optimizer"] = optimizer;
        
        py::dict objective_function;
        objective_function["type"] = py::str(objective_type);
        fitting_strategy["objective_function"] = objective_function;
        
        config["fitting_strategy"] = fitting_strategy;
        
        // Call the main fitting function
        return fit_material(config);
        
    } catch (const std::exception& e) {
        std::map<std::string, py::object> error_result;
        error_result["success"] = py::bool_(false);
        error_result["error"] = py::str(e.what());
        return error_result;
    }
}

/**
 * Helper function to extract fitted parameters from result.
 * 
 * This provides a convenient way to access the fitted parameters
 * as C++ vectors for further use in C++ applications.
 */
std::map<std::string, std::vector<double>> extract_parameters(
    const std::map<std::string, py::object>& result
) {
    std::map<std::string, std::vector<double>> parameters;
    
    try {
        if (result.find("success") != result.end() && 
            result.at("success").cast<bool>() &&
            result.find("parameters") != result.end()) {
            
            py::dict param_dict = result.at("parameters").cast<py::dict>();
            
            for (auto item : param_dict) {
                std::string param_name = item.first.cast<std::string>();
                
                // Handle both scalar and array parameters
                try {
                    // Try as array first
                    py::array_t<double> param_array = item.second.cast<py::array_t<double>>();
                    std::vector<double> param_values;
                    
                    auto buf = param_array.request();
                    double* ptr = static_cast<double*>(buf.ptr);
                    
                    for (py::ssize_t i = 0; i < buf.size; i++) {
                        param_values.push_back(ptr[i]);
                    }
                    
                    parameters[param_name] = param_values;
                    
                } catch (...) {
                    // Try as scalar
                    try {
                        double param_value = item.second.cast<double>();
                        parameters[param_name] = {param_value};
                    } catch (...) {
                        // Skip parameters that can't be converted
                    }
                }
            }
        }
    } catch (...) {
        // Return empty map if extraction fails
    }
    
    return parameters;
}

/**
 * Helper function to check if fitting was successful.
 */
bool is_fit_successful(const std::map<std::string, py::object>& result) {
    try {
        if (result.find("success") != result.end()) {
            return result.at("success").cast<bool>();
        }
    } catch (...) {
        // Return false if check fails
    }
    return false;
}

/**
 * Helper function to get error message from result.
 */
std::string get_error_message(const std::map<std::string, py::object>& result) {
    try {
        if (result.find("error") != result.end()) {
            return result.at("error").cast<std::string>();
        } else if (result.find("message") != result.end()) {
            return result.at("message").cast<std::string>();
        }
    } catch (...) {
        // Return generic message if extraction fails
    }
    return "Unknown error occurred during fitting";
}

// Pybind11 module definition
PYBIND11_MODULE(hyperfit_cpp, m) {
    m.doc() = "HyperFit C++ Binding - Hyperelastic Material Model Fitting";
    
    // Main fitting functions
    m.def("fit", &fit_material, 
          "Fit hyperelastic material model using configuration dictionary",
          py::arg("config"));
    
    m.def("fit_with_arrays", &fit_material_with_arrays,
          "Fit hyperelastic material model using raw data arrays",
          py::arg("model_name"),
          py::arg("model_order"), 
          py::arg("uniaxial_strain"),
          py::arg("uniaxial_stress"),
          py::arg("biaxial_strain") = std::vector<double>(),
          py::arg("biaxial_stress") = std::vector<double>(),
          py::arg("planar_strain") = std::vector<double>(),
          py::arg("planar_stress") = std::vector<double>(),
          py::arg("volumetric_j") = std::vector<double>(),
          py::arg("volumetric_pressure") = std::vector<double>(),
          py::arg("initial_guess_method") = "lls",
          py::arg("optimizer_methods") = std::vector<std::string>{"L-BFGS-B"},
          py::arg("objective_type") = "relative_error");
    
    // Helper functions
    m.def("extract_parameters", &extract_parameters,
          "Extract fitted parameters from result as C++ vectors",
          py::arg("result"));
    
    m.def("is_successful", &is_fit_successful,
          "Check if fitting was successful",
          py::arg("result"));
    
    m.def("get_error", &get_error_message,
          "Get error message from fitting result",
          py::arg("result"));
    
    // Version information
    m.attr("__version__") = "0.1.0";
}
