"""
Constraint projections for harmonic jet optimisation.

A constraint is a factory function returning a HarmonicJet → HarmonicJet
projection.  Projections enforce mathematical conditions on one or more jet
tensors and are applied after each gradient step inside the optimizer.

Factory functions:
    fix_u(c)                  — set T⁰ = c (function value at origin)
    clamp_u_nonneg(eps)       — set T⁰ = max(T⁰, eps)  (u(0) > 0)
    fix_grad_norm(r)          — set |T¹| = r
    project_grad_ball(r)      — set |T¹| ≤ r
    fix_tensor_frob_norm(m,r) — set ||T^m||_F = r

Riemannian gradient:
    sphere_tangent_proj(indices) — project gradient onto sphere tangent spaces
    before any optimizer step.  Required for correctness with momentum-based
    optimizers (Adam, etc.) when norm-fixing projections are active.

Internals:
    _reproject_harmonic(j) — restore STF constraint on all tensors m ≥ 2.
    replace_tensor(j, m, v) — functional update of one tensor in a jet.

Projections are composed by passing them as a tuple to optimize_ratio:
    projections = (fix_u(1.0), fix_grad_norm(1.0))
    optimize_ratio(ratio_fn, n=3, k=2, projections=projections, ...)

Projection order: applied in the order given; they may not commute.
Norm-fixing projections should generally come last.
"""

import jax.numpy as jnp

from ._harmonics import project_tracefree
from ._jet import HarmonicJet
from ._tensor import frobenius_sq, symmetrize


def replace_tensor(j: HarmonicJet, m: int, val: jnp.ndarray) -> HarmonicJet:
    """Return a new jet with tensors[m] replaced by val.

    All other tensors are unchanged.  Preserves the concrete type and all
    metadata (n, k, and lam for EigenfunctionJet).

    Args:
        j:   source jet
        m:   index of the tensor to replace (0 <= m <= j.k)
        val: replacement array; must have shape (j.n,)*m

    Returns:
        Same type as j, with tensors[m] = val.
    """
    new_tensors = tuple(val if i == m else t for i, t in enumerate(j.tensors))
    if hasattr(j, 'lam'):
        return type(j)(new_tensors, j.n, j.k, j.lam)
    return type(j)(new_tensors, j.n, j.k)


# ---------------------------------------------------------------------------
# Constraint factory functions
# ---------------------------------------------------------------------------


def fix_u(c: float):
    """Constraint: set T⁰ = c (fix function value u(0) = c).

    Args:
        c: target value (Python float or JAX scalar)

    Returns:
        Projection HarmonicJet → HarmonicJet.
    """
    def project(j: HarmonicJet) -> HarmonicJet:
        return replace_tensor(j, 0, jnp.asarray(c, dtype=j.tensors[0].dtype))
    return project


def clamp_u_nonneg(eps: float = 1e-6):
    """Constraint: set T⁰ = max(T⁰, eps), enforcing u(0) ≥ eps > 0.

    Args:
        eps: lower bound (default 1e-6)

    Returns:
        Projection HarmonicJet → HarmonicJet.
    """
    def project(j: HarmonicJet) -> HarmonicJet:
        return replace_tensor(j, 0, jnp.maximum(j.tensors[0], eps))
    return project


def fix_grad_norm(r: float, eps: float = 1e-8):
    """Constraint: set |T¹| = r (fix gradient norm at origin).

    Normalises T¹ to Euclidean norm r.  If |T¹| < eps, defaults to r·e₁
    to avoid division by zero.

    Args:
        r:   target norm (> 0)
        eps: zero threshold for |T¹|

    Returns:
        Projection HarmonicJet → HarmonicJet.
    """
    def project(j: HarmonicJet) -> HarmonicJet:
        T1 = j.tensors[1]
        norm = jnp.linalg.norm(T1)
        e1 = jnp.zeros_like(T1).at[0].set(1.0)
        T1_new = jnp.where(norm > eps, r * T1 / norm, r * e1)
        return replace_tensor(j, 1, T1_new)
    return project


def project_grad_ball(r: float):
    """Constraint: set |T¹| ≤ r (project gradient into closed ball of radius r).

    Identity when |T¹| ≤ r; radially clips to the sphere of radius r
    otherwise.

    Args:
        r: ball radius (> 0)

    Returns:
        Projection HarmonicJet → HarmonicJet.
    """
    def project(j: HarmonicJet) -> HarmonicJet:
        T1 = j.tensors[1]
        norm = jnp.linalg.norm(T1)
        scale = jnp.minimum(1.0, r / (norm + 1e-30))
        return replace_tensor(j, 1, T1 * scale)
    return project


def fix_tensor_frob_norm(m: int, r: float, eps: float = 1e-8):
    """Constraint: set ||T^m||_F = r (fix Frobenius norm of m-th tensor).

    Scales T^m to Frobenius norm r.  Trace-freeness is preserved by scaling.

    Args:
        m:   tensor index (0 <= m <= k)
        r:   target Frobenius norm (> 0)
        eps: small value added to denominator to avoid division by zero

    Returns:
        Projection HarmonicJet → HarmonicJet.
    """
    def project(j: HarmonicJet) -> HarmonicJet:
        T = j.tensors[m]
        norm = jnp.sqrt(frobenius_sq(T))
        T_new = T * (r / (norm + eps))
        return replace_tensor(j, m, T_new)
    return project


# ---------------------------------------------------------------------------
# Riemannian gradient correction
# ---------------------------------------------------------------------------


def sphere_tangent_proj(component_indices):
    """Return a tangent_proj_fn projecting gradient components to sphere tangent spaces.

    For each index m in component_indices, T^m is assumed to live on a sphere
    (fixed Frobenius norm, as enforced by fix_grad_norm or
    fix_tensor_frob_norm).  The Riemannian gradient is:

        G_R^m = G^m - (<G^m, T^m>_F / ||T^m||²_F) * T^m

    This must be applied BEFORE any optimizer step so that momentum
    accumulation in Adam etc. stays on the correct tangent space.

    Args:
        component_indices: sequence of tensor indices m whose Frobenius norms
                           are sphere-constrained.

    Returns:
        tangent_proj_fn: callable(j, g_j) → g_j (both are HarmonicJets).
    """
    indices = tuple(component_indices)

    def _proj(j, g_j):
        result = g_j
        for m in indices:
            T = j.tensors[m]
            G = g_j.tensors[m]
            inner = jnp.sum(G * T)
            norm_sq = jnp.sum(T * T)
            G_R = G - (inner / (norm_sq + 1e-30)) * T
            result = replace_tensor(result, m, G_R)
        return result

    return _proj


# ---------------------------------------------------------------------------
# Internal helper: restore PDE constraint after gradient step
# ---------------------------------------------------------------------------


def _reproject_harmonic(j: HarmonicJet) -> HarmonicJet:
    """Re-project all tensors of degree m ≥ 2 to STF (symmetrize + trace-free).

    Called after each gradient step to restore the harmonic constraint.
    Tensors of degree 0 and 1 are unchanged.
    """
    new_tensors = []
    for m, T in enumerate(j.tensors):
        if m >= 2:
            T = project_tracefree(symmetrize(T))
        new_tensors.append(T)
    return HarmonicJet(tuple(new_tensors), j.n, j.k)
