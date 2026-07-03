# NNCalderon

[![arXiv](https://img.shields.io/badge/arXiv-2606.28158-b31b1b.svg)](https://arxiv.org/abs/2606.28158) This repository contains the implementation accompanying the paper:

> **Recovering Sharp Conductivity Features in the Finite-Data Calderón Problem with Physics-Informed Neural Networks**  

## Overview

This repository implements a Physics-Informed Neural Network (PINN) framework for tackling the Calderón inverse problem from finite-data given by boundary measurements in a 2D square domain (also commonly referred to as "Electrical Impedance Tomography").

The method represents the unknown conductivity field $\gamma(x)$ and the internal electric potential $u(x)$ using separate neural networks (NNs) trained simultaneously by enforcing:

- the governing elliptic PDE: 
        $$\nabla\cdot\left(\gamma(x)\,\nabla u_k(x)\right) = 0 $$
- A finite set of Dirichlet boundary conditions: 
        $$u_k(x)|_{\partial\Omega} = f_k \ \ \ \text{with} \ \ \  k=1,\dots, K \, .$$
- The corresponding set of Neumann (Dirichlet-to-Neumann) measurements: 
        $$\Lambda_\gamma(f_k) = \left[ \gamma(x)\frac{\partial u_k}{\partial \hat{n}} \right]_{\partial\Omega}\equiv J_k$$.

The framework supports both raw-coordinate inputs to the NNs and Fourier Feature Encoding (FFE), a transformation of the input space to a higher-dimensional space populated with frequencies sampled from a random Gaussian distribution. The latter shows better reconstruction of conductivity profiles that present sharp features and high-frequency Fourier modes. The paper investigates the influence of multiscale randomized wavelet boundary excitations and FFE on reconstruction quality. For details see [this paper](https://arxiv.org/abs/2606.28158).


---

## Repository Structure

master/
│
├── `Data_generation.ipynb`
│   Notebook for generating synthetic Dirichlet-to-Neumann (DtN) datasets for a given ground truth conductivity (to be chosen - multiple options used in the paper are present). 
│   Uses the `fdm_forward_solver.py` file: Finite-difference forward solver used to simulate boundary measurements and generate synthetic training data.
│   

├── `NNCalderon_rutine.py`
│   Core implementation of the PINN framework, including
│   - neural network architectures,
│   - Fourier Feature Encoding,
│   - loss functions,
│   - dataset utilities,
│   - training class,
│   - visualization utilities.
│
│
└── `NNCalderon_TRAIN.py`
    Example training script demonstrating how to configure and train the
    PINN for a selected conductivity reconstruction problem.

data/
│
├── `dtn_data_single_inclusion.npz` (e.g. for the single inclusion case wit radius 0.2 of the box-length).
│    File with the DtN data-pairs used in training for the conductivity reconstruction, and ground truth conductivity for evaluation of the reconstruction quality.
│
└── `NN_epochs_xxx`
    NN parameters (weights and biases), optimizer state, and training history for the trained models in [our work](https://arxiv.org/abs/2606.28158) at the indicated epoch.

---

## Workflow

The typical workflow is

1. Generate synthetic DtN data using the finite-difference solver, executed in `Data_generation.ipynb`.
2. Train the PINN to reconstruct the conductivity field from the generated DtN data-pairs by running `NNCalderon_TRAIN.py` (training parameters are specified here).
3. Evaluate the reconstructed conductivity and visualize the results using the built-in plotting utilities.

---

## Main Features

- PINNs for Calderón inversion
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
pip install -r requirements.txt
```

---


## License

This repository is released for academic and research purposes. Please cite the accompanying paper if you use this code in your work.
