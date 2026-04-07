"""
pde_jet: JAX library for k-jets of PDE solutions.

Public API:
    Jet construction:
        HarmonicJet, make_harmonic_jet, zero_jet, random_harmonic_jet

    Tensor primitives:
        symmetrize, trace, full_trace_k, delta_sym, sym_outer,
        frobenius_sq, contract_vector, contract_matrix

    Harmonic tools:
        project_tracefree, is_tracefree, harmonic_dim

    Operators (evaluation at a point):
        evaluate_polynomial, gradient_at, hessian_at, laplacian_at

    Scalar functional helpers (chain-rule at origin):
        gradient_of_scalar_functional, laplacian_of_scalar_functional

    Constraints and optimization:
        replace_tensor, fix_u, clamp_u_nonneg, fix_grad_norm,
        project_grad_ball, fix_tensor_frob_norm, sphere_tangent_proj,
        optimize_ratio

    optimize_ratio supports three optimizer modes:
        optimizer=None              plain projected gradient ascent/descent
        optimizer=optax.Optimizer   any optax.GradientTransformation (e.g. adam)
        optimizer='lbfgs'           jaxopt.LBFGS (second-order, recommended for
                                    adversarial / high-accuracy settings)

    For sphere-constrained problems (fix_grad_norm, fix_tensor_frob_norm),
    pass tangent_proj_fn=sphere_tangent_proj([m1, m2, ...]) to project the
    Euclidean gradient onto the Riemannian tangent space before each optimizer
    step. Required for correctness with momentum-based optimizers.
"""

from ._eigenfunction import (
    EigenfunctionJet,
    _reproject_eigenfunction as reproject_eigenfunction,
    make_eigenfunction_jet,
    random_eigenfunction_jet,
    zero_eigenfunction_jet,
)
from ._constraints import (
    clamp_u_nonneg,
    fix_grad_norm,
    fix_tensor_frob_norm,
    fix_u,
    optimize_ratio,
    project_grad_ball,
    replace_tensor,
    sphere_tangent_proj,
)
from ._functionals import (
    gradient_of_scalar_functional,
    laplacian_of_scalar_functional,
)
from ._harmonics import harmonic_dim, is_tracefree, project_tracefree
from ._jet import HarmonicJet, make_harmonic_jet, random_harmonic_jet, zero_jet
from ._operators import evaluate_polynomial, gradient_at, hessian_at, laplacian_at
from ._tensor import (
    contract_matrix,
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
    "contract_matrix",
    # Harmonic tools
    "project_tracefree",
    "is_tracefree",
    "harmonic_dim",
    # Operators
    "evaluate_polynomial",
    "gradient_at",
    "hessian_at",
    "laplacian_at",
    # Scalar functional helpers
    "gradient_of_scalar_functional",
    "laplacian_of_scalar_functional",
    # EigenfunctionJet
    "EigenfunctionJet",
    "make_eigenfunction_jet",
    "zero_eigenfunction_jet",
    "random_eigenfunction_jet",
    "reproject_eigenfunction",
    # Constraints and optimization
    "replace_tensor",
    "fix_u",
    "clamp_u_nonneg",
    "fix_grad_norm",
    "project_grad_ball",
    "fix_tensor_frob_norm",
    "sphere_tangent_proj",
    "optimize_ratio",
]
