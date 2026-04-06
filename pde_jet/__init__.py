"""
pde_jet: JAX library for k-jets of PDE solutions.

Public API:
    Jet construction:
        HarmonicJet, make_harmonic_jet, zero_jet, random_harmonic_jet

    Tensor primitives:
        symmetrize, trace, full_trace_k, delta_sym, sym_outer,
        frobenius_sq, contract_vector

    Harmonic tools:
        project_tracefree, is_tracefree, harmonic_dim

    Operators (evaluation at a point):
        evaluate_polynomial, gradient_at, hessian_at

    Constraints and optimization:
        replace_tensor, fix_u, clamp_u_nonneg, fix_grad_norm,
        project_grad_ball, fix_tensor_frob_norm, optimize_ratio
"""

from ._constraints import (
    clamp_u_nonneg,
    fix_grad_norm,
    fix_tensor_frob_norm,
    fix_u,
    optimize_ratio,
    project_grad_ball,
    replace_tensor,
)
from ._harmonics import harmonic_dim, is_tracefree, project_tracefree
from ._jet import HarmonicJet, make_harmonic_jet, random_harmonic_jet, zero_jet
from ._operators import evaluate_polynomial, gradient_at, hessian_at
from ._tensor import (
    contract_vector,
    delta_sym,
    frobenius_sq,
    full_trace_k,
    sym_outer,
    symmetrize,
    trace,
)

__all__ = [
    # Jet
    "HarmonicJet",
    "make_harmonic_jet",
    "zero_jet",
    "random_harmonic_jet",
    # Tensor primitives
    "symmetrize",
    "trace",
    "full_trace_k",
    "delta_sym",
    "sym_outer",
    "frobenius_sq",
    "contract_vector",
    # Harmonic tools
    "project_tracefree",
    "is_tracefree",
    "harmonic_dim",
    # Operators
    "evaluate_polynomial",
    "gradient_at",
    "hessian_at",
    # Constraints
    "replace_tensor",
    "fix_u",
    "clamp_u_nonneg",
    "fix_grad_norm",
    "project_grad_ball",
    "fix_tensor_frob_norm",
    "optimize_ratio",
]
