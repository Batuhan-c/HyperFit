"""
Setup script for building HyperFit C++ bindings.

This script builds the Pybind11 C++ extension module for HyperFit,
allowing C++ applications to use the library seamlessly.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "hyperfit_cpp",
        [
            "bindings.cpp",
        ],
        # Include paths
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_cmake_dir() + "/../../../include",
        ],
        # Language standard
        cxx_std=14,
        # Compiler flags
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="hyperfit_cpp",
    version="0.1.0",
    author="HyperFit Contributors",
    description="C++ bindings for HyperFit hyperelastic material fitting library",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.20.0",
    ],
)
