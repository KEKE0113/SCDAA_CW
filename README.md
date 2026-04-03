# SCDAA Coursework Code

This repository contains the code for the 2025/26 SCDAA coursework.

## Group Information

- Keke Zhou, s2785390, contribution: 1/3
- Jiaxin Yang, s2804687, contribution: 1/3
- Shengzhen Xu, s2890679, contribution: 1/3

## Environment Setup

This project uses Python 3 together with the following libraries:

- numpy
- scipy
- matplotlib
- torch

Install them with:

```bash
pip install numpy scipy matplotlib torch
```

## Exercise-to-Code Mapping

- **Exercise 1.1**: `LQR.py`
  - defines the `LQR` class;
  - solves the Riccati ODE;
  - evaluates the benchmark value function and optimal Markov control.

- **Exercise 1.2**: `LQR.py` and `main.py`
  - `LQR.py` contains the Monte Carlo simulation and error analysis;
  - `main.py` runs the convergence checks with the coursework-scale settings:
    - **Exercise 1.2.2** uses `10^5` Monte Carlo samples when varying the number of time steps;
    - **Exercise 1.2.3** uses `5000` time steps when varying the number of Monte Carlo samples.

- **Exercise 2.1**: `DGM.py`
  - contains the shared DGM value-network architecture;
  - trains the value-function network using supervised learning with labels from Exercise 1.1.

- **Exercise 2.2**: `FNN.py`
  - contains the shared feed-forward control-network architecture;
  - trains the control network using supervised learning with labels from Exercise 1.1.

- **Exercise 3.1**: `PDE_Solve_dgm.py`
  - implements the Deep Galerkin Method for the linear PDE under fixed control `alpha = (1,1)^T`;
  - reuses the Monte Carlo simulation framework from Exercise 1.2, adapted from optimal control to constant control.

- **Exercise 4.1**: `Policy_Iteration.py`
  - implements policy iteration;
  - reuses the DGM value-network and FFN control-network architectures from Exercises 2.1 and 2.2;
  - starts from the supervised networks trained in Exercises 2.1 and 2.2 instead of reinitialising from scratch;
  - reuses the PDE-residual / terminal-loss structure from Exercise 3.1 in the value-update step.

- **Entry point**: `main.ipynb`
  - runs Exercises 1.1, 1.2, 2.1, 2.2, 3.1, and 4.1 in sequence;
  - passes the trained networks from Exercises 2.1 and 2.2 into Exercise 4.1 as warm starts.

## How the Tasks Connect

The coursework asks that code from one task should be usable in the next one. The code is organised so that this happens explicitly:

1. The exact benchmark from **Exercise 1.1** provides the value-function and control labels used in **Exercises 1.2, 2.1, and 2.2**.
2. The Monte Carlo code from **Exercise 1.2** is reused in **Exercise 3.1** after replacing the optimal control with the fixed control `alpha = (1,1)^T`.
3. The network architectures introduced in **Exercises 2.1 and 2.2** are reused in **Exercise 4.1**.
4. The trained networks from **Exercises 2.1 and 2.2** are passed directly into **Exercise 4.1** as the starting point for policy iteration.
5. The value-update step in **Exercise 4.1** follows the same DGM PDE-loss structure used in **Exercise 3.1**.

## How to Run

After installing the required libraries, run:

```bash
python main.py
```

This script will:

1. solve the Riccati ODE benchmark from Exercise 1.1;
2. run the Monte Carlo convergence checks from Exercise 1.2;
3. train the value-function network for Exercise 2.1;
4. train the control network for Exercise 2.2;
5. train the DGM solver for Exercise 3.1;
6. run policy iteration for Exercise 4.1 using the trained Exercise 2 networks as initial guesses.

## Notes

- Random seeds are fixed in `main.ipynb` for reproducibility.
- Plots are displayed directly using `matplotlib`.
- Exercise 1.2 is configured with the coursework-scale settings, so that part of `main.ipynb` can take substantially longer to run than the other sections.
- The written explanation and discussion are left for the report stage rather than the code repository.
