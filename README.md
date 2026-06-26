# PINNs — My Implementations

My own step-by-step **Physics-Informed Neural Network** implementations, written while working through Dr. Mohammad Samara's courses. The notebooks are numbered as a learning progression — from a first 1D heat solver up to 2D Navier–Stokes — and mix **from-scratch PyTorch** with **[DeepXDE](https://deepxde.readthedocs.io/)**, plus a finite-difference (FDM) reference for comparison.

---

## Projects (in learning order)

| # | Notebook | Problem | Approach |
|---|---|---|---|
| 1 | [`Erfan_heat_1.ipynb`](Erfan_heat_1.ipynb) | 1D heat (diffusion) equation | PyTorch (from scratch) |
| 2 | [`Burgers_2D_FDM_Erfan_2.ipynb`](Burgers_2D_FDM_Erfan_2.ipynb) | 2D Burgers' equation | Finite-difference reference |
| 3 | [`PINNS_1D_Burgers_Erfan_3.ipynb`](PINNS_1D_Burgers_Erfan_3.ipynb) | 1D Burgers' equation | PyTorch (from scratch) |
| 4 | [`PINNS_2D_Heat_Erfan_4.ipynb`](PINNS_2D_Heat_Erfan_4.ipynb) | 2D heat equation | PyTorch (from scratch) |
| 5 | [`DeepXDE_1D_Heat_Erfan_5.ipynb`](DeepXDE_1D_Heat_Erfan_5.ipynb) | 1D heat equation | DeepXDE |
| 6 | [`DeepXDE_2D_NS_Erfan_6.ipynb`](DeepXDE_2D_NS_Erfan_6.ipynb) | 2D Navier–Stokes flow | DeepXDE |

> `*.dat` files are training/evaluation logs produced by the DeepXDE notebooks.

## Tech stack

- Python 3.9+
- [PyTorch](https://pytorch.org/), [DeepXDE](https://deepxde.readthedocs.io/)
- NumPy, Matplotlib

## Getting started

```bash
git clone https://github.com/erfant00001/pinns-my-implementations.git
cd pinns-my-implementations
pip install torch deepxde numpy matplotlib jupyter
jupyter notebook
```

Run the notebooks in numerical order to follow the progression.

## Acknowledgements

These projects were created while learning Physics-Informed Neural Networks and Scientific Machine Learning from the Udemy courses of **Dr. Mohammad Samara** (Data Science / Machine Learning expert; PhD, University of Tokyo).

Instructor profile: **https://www.udemy.com/user/mohammad-samara-18/**

The problem setups and course material are credited to the instructor. This repository contains my own implementations and notes produced while following the courses, shared for learning and reference.
