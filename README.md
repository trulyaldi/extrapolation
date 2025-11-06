# Extrapolation

A comprehensive data analysis and experimentation repository focused on exploring and implementing extrapolation techniques for **exponential decay convergence** datasets across various scales and scientific domains.

## 📋 Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Applications](#applications)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebooks](#running-the-notebooks)
- [Dataset Guide](#dataset-guide)
- [Extrapolation Methods](#extrapolation-methods)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains a collection of Jupyter Notebooks and Python utilities for exploring and implementing **large basis function extrapolation** techniques. It specializes in working with datasets that follow **exponential decay patterns converging to a limit at infinity**, a fundamental challenge in computational physics and engineering.

The project is designed to be both educational and practical, allowing researchers and data scientists to:
- Understand exponential decay extrapolation fundamentals
- Compare different extrapolation approaches for convergent sequences
- Test basis function algorithms on multiple dataset scales
- Visualize and interpret asymptotic convergence behavior
- Build reproducible analysis pipelines for scientific computing
- Apply techniques across diverse scientific domains

## Scientific Background

### Exponential Decay Convergence

This repository focuses on extrapolating data points that exhibit the following characteristics:

- **Pattern**: Values approach a limiting value (asymptote) as the variable approaches infinity
- **Mathematical Form**: $f(x) = L + A \cdot e^{-\lambda x}$, where:
  - $L$ = asymptotic limit (target value at infinity)
  - $A$ = amplitude
  - $\lambda$ = decay rate

- **Challenge**: Accurately estimating the limiting value from finite data points

### Large Basis Function Extrapolation

This project implements techniques for basis function extrapolation, which:
- Represents sequences using expanded basis sets
- Extracts limiting behavior from partial sequence information
- Improves convergence rate estimation
- Enables accurate prediction of asymptotic values

## Applications

This extrapolation methodology is widely used in:

### **Finite Element Methods (FEM)**
- Mesh convergence studies
- Richardson extrapolation for solution refinement
- Basis function expansion in Galerkin methods

### **Quantum Mechanics**
- Basis set convergence in quantum chemistry calculations
- Hartree-Fock and DFT convergence studies
- Electronic correlation energy extrapolation

### **Chemistry & Molecular Dynamics**
- Computational chemistry convergence analysis
- Potential energy surface extrapolation
- Vibrational frequency convergence

### **Biology & Biophysics**
- Molecular dynamics trajectory analysis
- Protein fold prediction convergence
- Ensemble averaging for stability estimation

### **General Scientific Computing**
- Numerical sequence acceleration
- Series convergence analysis
- Asymptotic behavior prediction



## Features

- **Specialized for Exponential Decay**: Optimized algorithms for convergent sequences approaching asymptotic limits
- **Multi-scale Dataset Analysis**: Work with small synthetic data, large realistic datasets, and experimental new data
- **Jupyter Notebook-based**: Interactive, well-documented analyses (98.5% of codebase)
- **Reusable Python Utilities**: Basis function implementations and extrapolation algorithms in `src/`
- **Multiple Basis Function Approaches**: Support for various basis function sets (polynomial, exponential, rational, etc.)
- **Convergence Analysis Tools**: Detect and quantify convergence rates
- **Cross-Domain Application**: Examples from FEM, QM, chemistry, and biology
- **Reproducible Workflows**: Clear, step-by-step analysis pipelines
- **Visualization Support**: Built-in convergence plots and result interpretation
- **Modular Design**: Easy to extend and adapt for new analyses

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.7 or higher**
- **pip** (Python package manager)
- **git**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/trulyaldi/extrapolation.git
   cd extrapolation
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, install the required packages:
   ```bash
   pip install jupyter numpy pandas matplotlib seaborn scikit-learn scipy sympy
   ```

### Running the Notebooks

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   
   Or use JupyterLab for an enhanced interface:
   ```bash
   pip install jupyterlab
   jupyter lab
   ```

2. **Navigate and Open Notebooks**
   - Your browser will open to `http://localhost:8888`
   - Browse to the notebook files in the repository
   - Click on any `.ipynb` file to open it

3. **Run Cells**
   - Click on a cell and press `Shift + Enter` to run it
   - Or use the "Run All" button to execute the entire notebook

## Dataset Guide

### Small Dataset
- **Purpose**: Quick iterations with synthetic exponential decay sequences
- **Location**: `small-dataset/`
- **Contents**:
  - Toy examples with known asymptotic limits
  - Synthetic sequences: $f(n) = L + A \cdot e^{-\lambda n}$
  - Simple basis function convergence tests
- **Use Case**: Algorithm validation, educational exploration, quick feedback loops
- **Advantages**: Fast execution, transparent data generation, easy to inspect and understand

### Large Dataset
- **Purpose**: Realistic data from scientific applications with complex convergence behavior
- **Location**: `large-dataset/`
- **Contents**:
  - Finite Element Method mesh convergence studies
  - Quantum chemistry basis set extrapolation data
  - Scientific computing benchmark datasets
  - Real-world convergence sequences
- **Use Case**: Production-level analysis, performance testing, domain-specific applications
- **Advantages**: Realistic patterns, robust results, practical validation

### New Dataset
- **Purpose**: Experimental datasets for novel convergence patterns and comparison studies
- **Location**: `new-dataset/`
- **Contents**:
  - Custom exponential decay sequences
  - Benchmark datasets from literature
  - Exploratory data for hypothesis testing
- **Use Case**: Testing new hypotheses, algorithm benchmarking, comparative studies
- **Advantages**: Flexibility, innovation support, comparative performance analysis

## Extrapolation Methods

This repository implements and compares several extrapolation techniques:

### 1. **Polynomial Basis Extrapolation**
- Fit polynomial basis functions to sequence data
- Estimate asymptotic limit from polynomial expansion
- Best for: Smooth convergence behavior

### 2. **Exponential Basis Extrapolation**
- Use exponential basis: $\{1, e^{-\lambda_1 x}, e^{-\lambda_2 x}, ...\}$
- Directly model decay rates
- Best for: Exponential decay patterns

### 3. **Rational Function Extrapolation**
- Approximate sequences with rational functions
- Capture multiple timescales
- Best for: Complex multi-rate convergence

### 4. **Richardson Extrapolation**
- Classic method for finite difference convergence
- Successive refinement approach
- Best for: Mesh and grid refinement studies

### 5. **Sequence Transformation Methods**
- Shanks transformation
- Epsilon algorithm
- Levin transformation
- Best for: General sequence acceleration

## Project Structure Details

### `src/` Directory

Contains reusable Python modules implementing extrapolation techniques:

```python
# basis_functions.py - Basis function definitions
from src.basis_functions import ExponentialBasis, PolynomialBasis

# extrapolation.py - Extrapolation algorithms
from src.extrapolation import ExtrapolationFitter, estimate_asymptotic_limit

# convergence.py - Analysis tools
from src.convergence import analyze_convergence_rate, detect_exponential_decay

# utils.py - Helper functions
from src.utils import generate_synthetic_data, plot_convergence
```

### Notebook Organization

Each notebook follows a standard structure:

1. **Introduction** - Problem statement and objectives
2. **Data Loading** - Import and inspect convergent sequences
3. **Exploratory Analysis** - Visualize exponential decay patterns
4. **Preprocessing** - Normalize and prepare data
5. **Basis Function Selection** - Choose appropriate basis functions
6. **Modeling & Extrapolation** - Fit and extrapolate to asymptotic limit
7. **Convergence Analysis** - Quantify convergence rates
8. **Evaluation & Visualization** - Compare results and visualize convergence
9. **Conclusions** - Summary and insights

## Usage Examples

### Basic Extrapolation Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from src.extrapolation import ExtrapolationFitter
from src.basis_functions import ExponentialBasis

# Generate or load exponential decay data
x = np.linspace(0, 10, 50)
asymptotic_limit = 100
data = asymptotic_limit + 50 * np.exp(-0.5 * x)

# Fit basis functions
basis = ExponentialBasis(num_terms=3)
fitter = ExtrapolationFitter(basis)
fitter.fit(x, data)

# Estimate asymptotic limit
estimated_limit = fitter.extrapolate()
print(f"True limit: {asymptotic_limit}")
print(f"Estimated limit: {estimated_limit}")

# Visualize
x_extended = np.linspace(0, 20, 200)
y_fitted = fitter.predict(x_extended)
plt.plot(x, data, 'o', label='Data')
plt.plot(x_extended, y_fitted, '-', label='Fitted')
plt.axhline(estimated_limit, '--', label=f'Extrapolated limit: {estimated_limit:.2f}')
plt.legend()
plt.show()
```

### Convergence Rate Analysis

```python
from src.convergence import analyze_convergence_rate

# Analyze convergence behavior
convergence_rate, decay_constant = analyze_convergence_rate(x, data)
print(f"Exponential decay rate: {decay_constant:.4f}")
print(f"Convergence rate: {convergence_rate}")
```

### Running Specific Analyses

1. **Start with small-dataset** notebooks for quick understanding of exponential decay patterns
2. **Move to large-dataset** notebooks for domain-specific applications (FEM, QM, Chemistry, Biology)
3. **Experiment with new-dataset** notebooks for custom explorations and novel problems

### Guidelines

- Write clear, descriptive commit messages
- Add detailed comments and documentation to code
- Include docstrings for all functions
- Test new algorithms on multiple datasets
- Add usage examples for new methods
- Update this README if adding major features

## License

This project is open source. Please check the LICENSE file in the repository for specific licensing information.

## Contact

**Author**: [trulyaldi](https://github.com/trulyaldi)

For questions, suggestions, or issues:
- Open an [Issue](https://github.com/trulyaldi/extrapolation/issues) on GitHub
- Contact the author directly through GitHub


## Typical Workflow by Domain

### Finite Element Analysis
```
Load mesh convergence data → Identify exponential convergence
→ Apply Richardson extrapolation → Estimate exact solution
```

### Quantum Chemistry
```
Generate basis set data → Fit exponential basis functions
→ Extrapolate to infinite basis limit → Obtain CBS (Complete Basis Set) result
```

### Molecular Dynamics
```
Analyze trajectory convergence → Detect decay rates
→ Accelerate sequence convergence → Compute bulk properties
```
