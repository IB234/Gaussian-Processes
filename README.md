# 📈 Gaussian Processes & Kernel Methods

This repository contains a full implementation of Gaussian Processes (GPs) from scratch, developed as part of the Bayesian Machine Learning course. The project explores non-parametric Bayesian regression, the expressive power of various kernel functions, and hyperparameter tuning using marginal log-evidence.

## 🧠 Overview

Gaussian Processes offer a powerful, flexible, and probabilistic approach to regression. Instead of calculating a distribution over weights (like in Bayesian Linear Regression), a GP computes a distribution directly over functions. 

This project implements a numerical-stable `GaussianProcess` class using Cholesky decomposition and explores how different covariance functions (kernels) encode our prior beliefs about the target function's shape, smoothness, and periodicity.

### ✨ Features
* **Custom GP Implementation:** A complete `GaussianProcess` class supporting fitting, prediction, standard deviation estimation, and function sampling.
* **Marginal Log-Evidence:** Built-in calculation of the log-evidence for principled hyperparameter selection.
* **Diverse Kernel Functions:** Implementation of Laplacian, RBF, Spectral, and Neural Network kernels.
* **Bayesian Linear Regression Utility:** Includes a classic parametric Bayesian Linear Regression module with polynomial basis functions for comparison.

## 🔬 Implemented Kernels

The behavior of a Gaussian Process is entirely defined by its mean function (assumed to be 0) and its covariance function (kernel). This repository implements four distinct kernels:

1. **RBF (Squared Exponential) Kernel:** 
   Yields infinitely differentiable, smooth functions.
   $$k(x, x') = \alpha \exp\left(-\beta \|x - x'\|^2\right)$$

2. **Laplacian Kernel:** 
   Yields continuous but non-differentiable (rougher) functions, using the $L_1$ distance.
   $$k(x, x') = \alpha \exp\left(-\beta \|x - x'\|_1\right)$$

3. **Spectral (Gibbs) Kernel:** 
   Encodes periodicity and varying length scales, allowing for highly expressive oscillating functions.
   $$k(x, x') = \alpha \exp\left(-\beta \|x - x'\|^2\right) \cos\left(\pi \frac{x - x'}{\gamma}\right)$$

4. **Neural Network Kernel:** 
   Simulates the behavior of a fully connected neural network with a single infinitely wide hidden layer and an arcsin (error-function) activation.
   $$k(x, x') = \alpha \frac{2}{\pi} \arcsin\left( \frac{2\beta(1 + x^\top x')}{\sqrt{(1 + 2\beta(1 + x^\top x))(1 + 2\beta(1 + x'^\top x'))}} \right)$$

## 📊 Experiments & Visualizations

Running the main script generates several visualizations that provide intuition into how GPs behave:

### 1. Prior vs. Posterior Distributions
For each of the four kernels, the script tests three different parameter settings. It plots:
* **The Prior:** The mean ($0$) and $95\%$ confidence intervals ($2$ standard deviations), along with 5 functions sampled randomly from the prior distribution.
* **The Posterior:** After observing 5 data points, the GP updates its beliefs. The plots show the posterior mean fitting the data, the collapsing uncertainty around the observed points, and 5 functions sampled from the updated posterior.

### 2. Hyperparameter Tuning via Log-Evidence
To find the optimal bandwidth ($\beta$) for the RBF kernel, we cannot simply minimize the training error (which would lead to overfitting). Instead, we maximize the **Marginal Log-Evidence**:
$$ \log p(y | X, \theta) = -\frac{1}{2} y^\top K_y^{-1} y - \frac{1}{2} \log |K_y| - \frac{N}{2} \log(2\pi) $$
The script sweeps through $\beta \in [0.1, 7]$, plots the evidence curve, and visualizes the posterior predictions for the minimum, median, and maximum evidence values to demonstrate underfitting, optimal fitting, and overfitting.

## 🚀 Getting Started

### Dependencies
The code relies on standard data science libraries:
* `numpy`
* `matplotlib`

You can install them via pip:
```bash
pip install numpy matplotlib
```

### File Structure
* `ex4.py`: The main entry point. Contains the kernel definitions, the `GaussianProcess` class, and the script to generate all experimental plots.
* `ex4_utils.py`: Contains a utility `BayesianLinearRegression` class and polynomial basis function generators.

### Usage
To run the Gaussian Process experiments and generate the plots, simply execute:
```bash
python ex4.py
```
To run the Bayesian Linear Regression example:
```bash
python ex4_utils.py
```

## 📝 Implementation Details
* **Numerical Stability:** The covariance matrix inversion uses Cholesky decomposition ($K_y = L L^\top$) and a small jitter ($10^{-6} I$) is added to the diagonal to prevent non-positive-definite matrix errors during fitting and sampling.
* **Vectorization:** Distance matrices and kernel calculations are heavily vectorized using NumPy broadcasting for optimal performance.
