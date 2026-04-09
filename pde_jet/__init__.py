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

Functional calculus:
    gradient_of_scalar_functional  — ∇W|₀ for W = f(u, |∇u|², ||D²u||²_F) (closed-form)
    laplacian_of_scalar_functional — ΔW|₀ for W = f(u, |∇u|²) (closed-form)
    jet_functional_gradient        — ∇W|₀ for arbitrary W via nested autodiff
    jet_functional_laplacian       — ΔW|₀ for arbitrary W via nested autodiff

Constraint projections:
    replace_tensor          — functional update of one tensor in a jet
    fix_u(c)                — set T⁰ = c
    clamp_u_nonneg(eps)     — set T⁰ = max(T⁰, eps)
    fix_grad_norm(r)        — set |T¹| = r
    project_grad_ball(r)    — set |T¹| ≤ r
    fix_tensor_frob_norm(m,r) — set ||T^m||_F = r
    sphere_tangent_proj(indices) — Riemannian gradient correction for sphere constraints

Optimisation:
    optimize_ratio        — gradient ascent/descent with projections and multiple restarts
    optimize_hierarchical — level-by-level hierarchical optimisation

Tensor and harmonic tools:
    symmetrize        — full symmetrization of a rank-m tensor
    trace             — contract first two indices with delta
    frobenius_sq      — squared Frobenius norm of a tensor
    project_tracefree — Fischer projection to the STF (harmonic) subspace
    is_tracefree      — check whether a tensor is numerically trace-free
    harmonic_dim      — dimension of harmonic polynomials of degree m on R^n
"""

from ._constraints import (
    clamp_u_nonneg,
    fix_grad_norm,
    fix_tensor_frob_norm,
    fix_u,
    project_grad_ball,
    replace_tensor,
    sphere_tangent_proj,
)
from ._functionals import (
    gradient_of_scalar_functional,
    jet_functional_gradient,
    jet_functional_laplacian,
    laplacian_of_scalar_functional,
)
from ._harmonics import harmonic_dim, is_tracefree, project_tracefree
from ._jet import HarmonicJet, make_harmonic_jet, random_harmonic_jet, zero_jet
from ._operators import evaluate_polynomial
from ._optimize import optimize_hierarchical, optimize_ratio
from ._tensor import frobenius_sq, symmetrize, trace

__all__ = [
    # Jet
    "HarmonicJet",
    "make_harmonic_jet",
    "zero_jet",
    "random_harmonic_jet",
    # Evaluation
    "evaluate_polynomial",
    # Functional calculus
    "gradient_of_scalar_functional",
    "laplacian_of_scalar_functional",
    "jet_functional_gradient",
    "jet_functional_laplacian",
    # Constraint projections
    "replace_tensor",
    "fix_u",
    "clamp_u_nonneg",
    "fix_grad_norm",
    "project_grad_ball",
    "fix_tensor_frob_norm",
    "sphere_tangent_proj",
    # Optimisation
    "optimize_ratio",
    "optimize_hierarchical",
    # Tensor / harmonic tools
    "symmetrize",
    "trace",
    "frobenius_sq",
    "project_tracefree",
    "is_tracefree",
    "harmonic_dim",
]
