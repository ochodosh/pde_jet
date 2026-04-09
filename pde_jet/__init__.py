"""
pde_jet: JAX library for k-jets of harmonic functions.

Public API
----------
Jet construction:
    HarmonicJet         — data structure (JAX pytree); tensors[m] = d^m u(0)
    make_harmonic_jet   — construct with STF constraint enforced
    zero_jet            — all-zero jet
    random_harmonic_jet — random jet with STF constraint enforced

Polynomial evaluation:
    evaluate_polynomial — u(x) ≈ Σ_{m=0}^k (1/m!) T^(m)_{i1...im} x^{i1}...x^{im}

    Derivatives via JAX autodiff:
        grad_u = jax.grad(evaluate_polynomial, argnums=1)(j, x)
        hess_u = jax.hessian(evaluate_polynomial, argnums=1)(j, x)

Tensor and harmonic tools:
    symmetrize        — full symmetrization of a rank-m tensor
    trace             — contract first two indices with delta
    project_tracefree — Fischer projection to the STF (harmonic) subspace
    is_tracefree      — check whether a tensor is numerically trace-free
    harmonic_dim      — dimension of harmonic polynomials of degree m on R^n
"""

from ._harmonics import harmonic_dim, is_tracefree, project_tracefree
from ._jet import HarmonicJet, make_harmonic_jet, random_harmonic_jet, zero_jet
from ._operators import evaluate_polynomial
from ._tensor import symmetrize, trace

__all__ = [
    # Jet
    "HarmonicJet",
    "make_harmonic_jet",
    "zero_jet",
    "random_harmonic_jet",
    # Evaluation
    "evaluate_polynomial",
    # Tensor / harmonic tools
    "symmetrize",
    "trace",
    "project_tracefree",
    "is_tracefree",
    "harmonic_dim",
]
