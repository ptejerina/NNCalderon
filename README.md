# Recovering Sharp Conductivity Features in the Finite-Data Calderón Problem with Physics-Informed Neural Networks

This repository contains the implementation accompanying the paper

> **Recovering Sharp Conductivity Features in the Finite-Data Calderón Problem with Physics-Informed Neural Networks**  
> https://arxiv.org/abs/2606.28158

## Overview

This repository implements a Physics-Informed Neural Network (PINN) framework for solving the finite-data Calderón inverse problem (Electrical Impedance Tomography) from boundary measurements.

The method represents the unknown conductivity field and the corresponding electric potentials using separate neural networks trained simultaneously by enforcing

- the governing elliptic PDE,
- Dirichlet boundary conditions,
- Neumann (Dirichlet-to-Neumann) measurements.

The framework supports both raw-coordinate neural networks and Fourier Feature Encoding (FFE), allowing the reconstruction of both smooth and sharp conductivity distributions from a finite number of boundary measurements.

The paper additionally investigates the influence of multiscale randomized wavelet boundary excitations and Fourier feature embeddings on reconstruction quality.

## Paper Abstract

> Physics-informed neural networks (PINNs) have recently emerged as a promising framework for addressing the Calderón inverse problem from limited boundary data. In this work, we revisit neural Calderón inversion by introducing multiscale boundary excitations based on randomized wavelet functions and investigating the role of Fourier-feature encoding (FFE) for representing sharp conductivity variations. We propose a physics-informed reconstruction framework that represents the unknown conductivity and the associated family of electric potentials with separate neural networks conditioned on the applied boundary excitations. The governing elliptic PDE is enforced through physics-informed residuals, while finite Dirichlet-to-Neumann (DtN) data are incorporated through boundary losses. Using synthetic data from a finite-difference forward solver, we evaluate the method on conductivity fields with inclusions, sharp interfaces, smooth profiles, and heterogeneous media. Results show that the framework recovers dominant conductivity structures from finite boundary measurements with relative errors between approximately 3–12%. We show that FFE improves the reconstruction of localized sharp features, particularly for inclusions and interfaces, but is not universally optimal, with raw-coordinate networks performing competitively for smoother fields. These results highlight coordinate representations and boundary excitation design as key factors in neural Calderón inversion.

---

## Repository Structure

```
master/
│
├── Data_generation.ipynb
│   Notebook for generating synthetic Dirichlet-to-Neumann (DtN) datasets.
│
├── fdm_forward_solver.py
│   Finite-difference forward solver used to simulate boundary measurements
│   and generate synthetic training data.
│
├── NNCalderon_rutine.py
│   Core implementation of the PINN framework, including
│   - neural network architectures,
│   - Fourier Feature Encoding,
│   - loss functions,
│   - dataset utilities,
│   - training class,
│   - visualization utilities.
│
└── NNCalderon_TRAIN.py
    Example training script demonstrating how to configure and train the
    PINN for a selected conductivity reconstruction problem.
```

---

## Workflow

The typical workflow is

1. Generate synthetic DtN data using the finite-difference solver (`Data_generation.ipynb` or `fdm_forward_solver.py`).
2. Configure the training parameters in `NNCalderon_TRAIN.py`.
3. Train the PINN to reconstruct the conductivity field.
4. Evaluate the reconstructed conductivity and visualize the results using the built-in plotting utilities.

---

## Main Features

- Physics-Informed Neural Networks for Calderón inversion
- Finite-difference forward solver for synthetic data generation
- Fourier Feature Encoding (optional)
- Raw-coordinate neural network baseline
- Multiple conductivity test cases
- Configurable PDE and boundary sampling strategies
- Automatic visualization of
  - conductivity reconstructions
  - PDE residuals
  - induced potentials
  - boundary data
  - training losses

---

## Requirements

The implementation relies primarily on

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- tqdm

Install the required packages with

```bash
pip install torch numpy scipy matplotlib tqdm
```

---


## License

This repository is released for academic and research purposes. Please cite the accompanying paper if you use this code in your work.
