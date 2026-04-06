# pde_jet

A JAX library for k-jets of PDE solutions — the collection of partial derivatives of a solution up to order k at a point. Designed for GPU-accelerated mathematical experiments.

## What it does

A **k-jet** of a function u at the origin is the tuple of derivative tensors `(T⁰, T¹, ..., Tᵏ)` where `T^(m)_{i₁...iₘ} = ∂_{i₁}...∂_{iₘ} u(0)`. For harmonic functions (Δu = 0), the tensors for m ≥ 2 are symmetric and trace-free; for eigenfunctions (Δu + λu = 0), the trace of T^m is determined by λ and lower-order tensors.

Currently implemented: harmonic functions and eigenfunctions (Δu + λu = 0) on ℝⁿ.

**Sample application — Kato inequality.** For harmonic u, the sharp constant K in |∇|∇u|| ≤ K|D²u| satisfies K² = (n−1)/n. The library finds this numerically via gradient ascent on the Kato ratio over jet space, and verifies it against the analytic formula.

## Structure

```
pde_jet/
    _tensor.py       # symmetric tensor primitives (symmetrize, trace, sym_outer, ...)
    _harmonics.py    # trace-free projection (Fischer decomposition), harmonic_dim
    _jet.py          # HarmonicJet pytree, constructors
    _eigenfunction.py # EigenfunctionJet (Δu + λu = 0), inverse Fischer reconstruction
    _operators.py    # evaluate_polynomial, gradient_at, hessian_at
    _constraints.py  # constraint projections (fix_u, fix_grad_norm, ...), optimize_ratio
examples/
    kato.py          # Kato inequality: sharp constants K^2=(n-1)/n, K'^2=n/(n+2)
    test_kato.py     # tests for the Kato example (run with: pytest examples/)
tests/               # 115 tests encoding mathematical theorems
FISCHER_DECOMP.md    # mathematical background: symmetric tensor algebra, Fischer decomposition
```

## Running tests

```bash
pytest tests/
```

## Using in Google Colab

Install directly from the public repo (one-time per session):

```python
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/ochodosh/pde_jet.git"
], check=True)
```

After installation:

```python
import jax
import jax.numpy as jnp
from pde_jet import random_harmonic_jet, optimize_kato, kato_analytic

# Sample a random 3-jet of a harmonic function in R^3
key = jax.random.PRNGKey(0)
j = random_harmonic_jet(key, n=3, k=3)

# Find the Kato constant numerically (should converge to 2/3)
result = optimize_kato(n=3, num_steps=500, key=key)
print(f"Numerical K² = {result['best_K2']:.6f}")
print(f"Analytic  K² = {kato_analytic(3):.6f}")
```
