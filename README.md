# pde_jet

A JAX library for k-jets of PDE solutions. Designed for GPU-accelerated mathematical experiments — finding sharp constants in differential inequalities, discovering new PDE quantities via ML, and testing geometric analysis conjectures. Work in progress and likely to change!

## What it does

A **k-jet** of a function u at the origin is the tuple of derivative tensors `(T⁰, T¹, ..., Tᵏ)` where `T^(m)_{i₁...iₘ} = ∂_{i₁}...∂_{iₘ} u(0)`. For harmonic functions (Δu = 0), the tensors for m ≥ 2 are fully symmetric and trace-free (STF), the infinitely-iterated consequence of Δu = 0.

The jet represents u via its Taylor polynomial:

```
u(x) ≈ T⁰ + T¹ᵢ xⁱ + ½ T²ᵢⱼ xⁱxʲ + ⅙ T³ᵢⱼₖ xⁱxʲxᵏ + ...
```

## Structure

```
pde_jet/
    _tensor.py       # symmetric tensor primitives (symmetrize, trace, frobenius_sq, ...)
    _harmonics.py    # trace-free projection (Fischer decomposition), harmonic_dim
    _jet.py          # HarmonicJet pytree, constructors
    _operators.py    # evaluate_polynomial
    _functionals.py  # ∇W and ΔW for scalar functionals of jet data
    _constraints.py  # constraint projections (fix_u, fix_grad_norm, ...)
    _optimize.py     # optimize_ratio, optimize_hierarchical

examples/
    kato.py              # Kato inequality: |∇|∇u|| ≤ K|D²u|, sharp K² = (n-1)/n
    harnack.py           # Gradient Harnack: Bochner identity and Cheng-Yau ratio
    optimizer_comparison.py  # GD vs Adam vs L-BFGS comparison
```

## Usage

### Jet construction and polynomial evaluation

```python
import jax
import jax.numpy as jnp
import pde_jet as pj

key = jax.random.PRNGKey(0)

# Construct a random harmonic 3-jet in R^3
j = pj.random_harmonic_jet(key, n=3, k=3)

# Evaluate u(x) via Taylor polynomial
x = jnp.array([0.1, 0.2, 0.3])
u = pj.evaluate_polynomial(j, x)

# Derivatives via JAX autodiff
grad_u = jax.grad(pj.evaluate_polynomial, argnums=1)(j, x)    # shape (3,)
hess_u = jax.hessian(pj.evaluate_polynomial, argnums=1)(j, x) # shape (3, 3)

# Gradient w.r.t. jet tensors (for optimizing over jets)
dL_dj = jax.grad(lambda j: some_loss(pj.evaluate_polynomial(j, x)))(j)
```

### Functional calculus: ∇W and ΔW at the origin

For `W = f(u, |∇u|², ||D²u||²_F)`, exact chain-rule formulas:

```python
def f(u, s, q):          # W = |∇ log u|² = s / u²
    return s / u**2

grad_W = pj.gradient_of_scalar_functional(f, j)   # shape (n,)
lap_W  = pj.laplacian_of_scalar_functional(f, j)  # scalar, assumes f_q = 0
```

For arbitrary W (including MLP-parameterised), autodiff-based fallback:

```python
def W_fn(eval_fn, x):
    u     = eval_fn(x)
    grad_u = jax.grad(eval_fn)(x)
    return jnp.dot(grad_u, grad_u) / u**2   # |∇ log u|²

grad_W = pj.jet_functional_gradient(W_fn, j)   # shape (n,)
lap_W  = pj.jet_functional_laplacian(W_fn, j)  # scalar
```

### Constrained optimization over jets

```python
from pde_jet import (
    optimize_ratio, optimize_hierarchical,
    fix_u, fix_grad_norm, fix_tensor_frob_norm, clamp_u_nonneg,
)

# Minimize a ratio over harmonic 2-jets with u=1, |g|=1, ||H||=1
result = optimize_ratio(
    ratio_fn,
    n=3, k=2,
    num_steps=500,
    key=jax.random.PRNGKey(0),
    projections=(fix_u(1.0), fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
    lr=0.05,
    num_restarts=16,
    minimize=True,
)
print(result['best_ratio'], result['best_jet'])

# Hierarchical optimizer: fix low levels, optimize high levels first
result = optimize_hierarchical(
    ratio_fn,
    n=3, k=2,
    level_schedule=[([2], 400), ([1, 0], 200)],   # optimize T² first, then T¹, T⁰
    key=jax.random.PRNGKey(0),
    projections=projections,
    minimize=True,
)
```

### Saddle-point problems (outer max, inner min)

For discovering drift vectors or learning W via MLPs, use `extra_params`:

```python
# max_{mlp_params}  min_{jet}  ratio_fn(jet, mlp_params)
result = optimize_ratio(
    ratio_fn,                  # signature: (jet, mlp_params) -> scalar
    n=3, k=2,
    num_steps=500,
    key=key,
    projections=projections,
    minimize=True,             # minimize over jets
    extra_params=mlp_init,     # maximize over MLP weights
    extra_minimize=False,
    extra_lr=1e-3,
)
```

## Experiments

### Kato inequality (examples)

|∇|∇u|| ≤ K|D²u| with sharp `K² = (n-1)/n`, and the higher-order version
|∇|D²u|| ≤ K'|D³u| with `K'² = n/(n+2)`. Both recovered numerically via `optimize_ratio`.

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

86 tests pass (1 skipped: EigenfunctionJet pending restoration).
