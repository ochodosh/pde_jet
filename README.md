# pde_jet

A JAX library for k-jets of PDE solutions — the collection of partial derivatives of a solution up to order k at a point. Designed for GPU-accelerated mathematical experiments.

## What it does

A **k-jet** of a function u at the origin is the tuple of derivative tensors `(T⁰, T¹, ..., Tᵏ)` where `T^(m)_{i₁...iₘ} = ∂_{i₁}...∂_{iₘ} u(0)`. For harmonic functions (Δu = 0), the tensors for m ≥ 2 are symmetric and trace-free.

Currently implemented: harmonic functions on ℝⁿ.

**Sample application — Kato inequality.** For harmonic u, the sharp constant K in |∇|∇u|| ≤ K|D²u| satisfies K² = (n−1)/n. The library finds this numerically via gradient ascent on the Kato ratio over jet space, and verifies it against the analytic formula.

## Structure

```
pde_jet/
    _tensor.py      # symmetric tensor primitives (symmetrize, trace, sym_outer, ...)
    _harmonics.py   # trace-free projection (Fischer decomposition), harmonic_dim
    _jet.py         # HarmonicJet pytree, constructors
    _operators.py   # evaluate_polynomial, gradient_at, hessian_at, kato_ratio_sq
    _kato.py        # Kato ratio, analytic answer, gradient optimizer
tests/              # 61 tests encoding mathematical theorems
```

## Running tests

```bash
pytest tests/
```

## Using in Google Colab

Since the repo is private, authenticate with a GitHub personal access token (PAT). In a Colab cell:

```python
# 1. Install directly from the private repo (one-time per session)
import subprocess, sys

GITHUB_TOKEN = "ghp_your_token_here"   # or use Colab Secrets (recommended)
repo = "ochodosh/pde_jet"

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    f"git+https://{GITHUB_TOKEN}@github.com/{repo}.git"
], check=True)
```

**Recommended: use Colab Secrets** (the key icon in the left sidebar) to store your token as `GITHUB_TOKEN`, then:

```python
from google.colab import userdata
import subprocess, sys

token = userdata.get("GITHUB_TOKEN")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    f"git+https://{token}@github.com/ochodosh/pde_jet.git"
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
