# SCDAA Coursework Code

Code for the 2025/26 SCDAA coursework.

## Group
- Keke Zhou, s2785390, contribution 1/3
- Jiaxin Yang, s2804687, contribution 1/3
- Shengzhen Xu, s2890679, contribution 1/3

## Requirements
Use Python 3 with:
- numpy
- scipy
- matplotlib
- torch

Install with:
```bash
pip install numpy scipy matplotlib torch
```

## Main files
- `LQR.py` — Exercise 1.1 and the Monte Carlo routines used later
- `DGM.py` — Exercise 2.1 value network
- `FNN.py` — Exercise 2.2 control network
- `PDE_Solve_dgm.py` — Exercise 3.1
- `Policy_Iteration.py` — Exercise 4.1
- `Test_Extra.py` — Extra tests for validation
- `Main.ipynb` — runs the coursework in order

## How to run
Open `Main.ipynb` and run the cells from top to bottom:
```bash
jupyter notebook Main.ipynb
```
or
```bash
jupyter lab Main.ipynb
```

Please run the notebook in order. Later sections use objects created earlier.

## What each part does
- **1.1** Solve the Riccati ODE and evaluate the benchmark value function and optimal control.
- **1.2** Use Monte Carlo simulation to check convergence to the benchmark from 1.1.
- **2.1** Train a DGM-style network on benchmark value labels from 1.1.
- **2.2** Train a feed-forward network on benchmark control labels from 1.1.
- **3.1** Solve the linear PDE with constant control `alpha=(1,1)^T` and compare with a Monte Carlo benchmark adapted from 1.2.
- **4.1** Run policy iteration and compare with the benchmark from 1.1.

## Output
Running the notebook reproduces the plots used in the report.

Neural-network training is random, so the exact loss curves may vary slightly from run to run, but the overall behaviour should be similar.
