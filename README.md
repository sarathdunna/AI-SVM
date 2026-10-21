# SVM from Scratch: Pulsar Star Classification

A linear **Support Vector Machine implemented from first principles**. The SVM dual problem is solved directly as a quadratic program with [CVXOPT](https://cvxopt.org/), without using scikit-learn's SVM. It is applied to the HTRU2 pulsar star dataset.

## How it works

1. Builds the kernel (Gram) matrix `K = X·Xᵀ`.
2. Solves the soft-margin dual QP for the Lagrange multipliers α, subject to `0 ≤ αᵢ ≤ C` and `Σ αᵢ yᵢ = 0`.
3. Treats samples with `α > 1e-5` as support vectors and computes the weight vector `w = Σ αᵢ yᵢ xᵢ`.
4. Computes the bias `b` from margin support vectors (`0 < α < C`).
5. Predicts with `sign(w·x + b)`.

Features are standardized using training-set statistics only, and labels are mapped to `{-1, +1}`.

## Results

Trained on 800 samples and tested on 200 held-out samples (80/20 split, `random_state=42`):

| C | Test accuracy |
|---|---|
| 0.1 | 95.5% |
| 1 | 97.5% |
| 10 | 98.0% |
| 100 | 98.0% |
| 1000 | 98.0% |

## Run it

```bash
pip install numpy pandas scikit-learn cvxopt
python SVM.py
```

## Files

| File | Description |
|---|---|
| `SVM.py` | SVM implementation and experiment over C values |
| `pulsar_star_dataset.csv` | 8 statistical features of pulsar candidates plus a `Class` label |
| `SVM Implementation for Pulsar Classification.pdf` | Write-up |

**Tech:** Python, NumPy, CVXOPT, pandas, scikit-learn (data split and metrics only)
