# pde_jet

A JAX library for k-jets of harmonic functions. Designed for GPU-accelerated mathematical experiments. Work in progress and likely to change!

## What it does

A **k-jet** of a function u at the origin is the tuple of derivative tensors `(T⁰, T¹, ..., Tᵏ)` where `T^(m)_{i₁...iₘ} = ∂_{i₁}...∂_{iₘ} u(0)`. For harmonic functions (Δu = 0), the tensors for m ≥ 2 are fully symmetric and trace-free (STF), which is the infinitely-iterated consequence of Δu = 0.

The jet represents u via its Taylor polynomial:

```
u(x) ≈ T⁰ + T¹ᵢ xⁱ + ½ T²ᵢⱼ xⁱxʲ + ⅙ T³ᵢⱼₖ xⁱxʲxᵏ + ...
```

## Structure

```
pde_jet/
    _tensor.py     # symmetric tensor primitives (symmetrize, trace, ...)
    _harmonics.py  # trace-free projection (Fischer decomposition), harmonic_dim
    _jet.py        # HarmonicJet pytree, constructors
    _operators.py  # evaluate_polynomial
```

## Usage

```python
import jax
import jax.numpy as jnp
import pde_jet as pj

key = jax.random.PRNGKey(0)

# Construct a random harmonic 3-jet in R^3
j = pj.random_harmonic_jet(key, n=3, k=3)

# Evaluate u(x)
x = jnp.array([0.1, 0.2, 0.3])
u = pj.evaluate_polynomial(j, x)

# Derivatives via JAX autodiff
grad_u = jax.grad(pj.evaluate_polynomial, argnums=1)(j, x)   # shape (3,)
hess_u = jax.hessian(pj.evaluate_polynomial, argnums=1)(j, x) # shape (3, 3)

# Gradient w.r.t. the jet tensors themselves
dL_dj = jax.grad(lambda j: some_loss(pj.evaluate_polynomial(j, x)))(j)
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

## Running tests

```bash
pytest tests/
```
