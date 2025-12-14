# SFGN
Numerical experiments of Sensitivity-Free Gauss-Newton methods for large-scale PDE-constrained optimization.

## Installation Instructions
To install the required packages with anaconda, run

conda env create -f environment.yml
conda activate sfgn
pip install -e .

## Repository Structure
- `gn_s/objectives/`   - Objective functions for testing optimization methods, which are torch modules which count the number of forward and backward passes through the PDE simulation
- `gn_s/optimization/` - Optimization routines and algorithms  
- `tests/`             - Test suites and validation scripts
- `figures/`           - Generated output figures from tests
- `outputs/`           - Generated outputs from optimizations
