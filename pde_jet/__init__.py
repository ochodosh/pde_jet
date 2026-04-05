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
        evaluate_polynomial, gradient_at, hessian_at, kato_ratio_sq

    Kato inequality:
        kato_ratio_direct, kato_analytic, kato_optimal_T2, optimize_kato
"""

from ._harmonics import harmonic_dim, is_tracefree, project_tracefree
from ._jet import HarmonicJet, make_harmonic_jet, random_harmonic_jet, zero_jet
from ._kato import kato_analytic, kato_optimal_T2, kato_ratio_direct, optimize_kato
from ._operators import evaluate_polynomial, gradient_at, hessian_at, kato_ratio_sq
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
    "kato_ratio_sq",
    # Kato
    "kato_ratio_direct",
    "kato_analytic",
    "kato_optimal_T2",
    "optimize_kato",
]
