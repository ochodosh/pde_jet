"""
Constraint projections and general constrained optimizer for harmonic jet optimization.

A constraint is a factory function returning a HarmonicJet -> HarmonicJet projection.
The projection enforces a mathematical condition on one or more jet tensors.

Factory functions:
    fix_u(c)                  — set T⁰ = c (function value at origin)
    clamp_u_nonneg(eps)       — set T⁰ = max(T⁰, eps)  (u(0) > 0)
    fix_grad_norm(r)          — set |T¹| = r
    project_grad_ball(r)      — set |T¹| ≤ r
    fix_tensor_frob_norm(m,r) — set ||T^m||_F = r

Projections are composed by passing them as a tuple to optimize_ratio:
    projections = (fix_u(1.0), fix_grad_norm(1.0))
    optimize_ratio(ratio_fn, n=3, k=2, projections=projections, ...)

Scale invariance note: if ratio_fn is scale-invariant in some jet component,
include a norm-fixing projection (fix_grad_norm or fix_tensor_frob_norm) for that
component, otherwise gradient ascent will not converge.

Projection order: projections are applied in the order given. They may not commute.
Norm-fixing projections (fix_grad_norm, fix_tensor_frob_norm) should generally come
last so the final norm is exact.
"""

import jax
import jax.numpy as jnp

from ._harmonics import project_tracefree
from ._jet import HarmonicJet, random_harmonic_jet
from ._tensor import frobenius_sq, symmetrize


def replace_tensor(j: HarmonicJet, m: int, val: jnp.ndarray) -> HarmonicJet:
    """Return a new jet with tensors[m] replaced by val.

    All other tensors are unchanged. This is the only function that directly
    modifies the tensors tuple of a jet. Works for HarmonicJet and
    EigenfunctionJet (preserves the concrete type and all metadata).

    Args:
        j: source jet (HarmonicJet or EigenfunctionJet)
        m: index of the tensor to replace (0 <= m <= j.k)
        val: replacement array; must have shape (j.n,)*m

    Returns:
        Same type as j, with tensors[m] = val, all others from j.
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

    This is a retraction onto {T⁰ ≥ eps}, not a projection onto the open
    set {T⁰ > 0}, but is the natural numerical approximation.

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

    Normalizes T¹ to have Euclidean norm r. If |T¹| < eps, defaults to
    r * e₁ (first standard basis vector) to avoid division by zero.

    Args:
        r: target norm (> 0)
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

    Identity when |T¹| ≤ r; clips to the sphere of radius r otherwise.
    This is the Euclidean projection onto the closed ball {v : |v| ≤ r}.

    Args:
        r: ball radius (> 0)

    Returns:
        Projection HarmonicJet → HarmonicJet.
    """
    def project(j: HarmonicJet) -> HarmonicJet:
        T1 = j.tensors[1]
        norm = jnp.linalg.norm(T1)
        # min(1, r/norm) * T1; add small eps to avoid 0/0 when T1=0
        scale = jnp.minimum(1.0, r / (norm + 1e-30))
        return replace_tensor(j, 1, T1 * scale)
    return project


def fix_tensor_frob_norm(m: int, r: float, eps: float = 1e-8):
    """Constraint: set ||T^m||_F = r (fix Frobenius norm of m-th tensor).

    Scales T^m to have Frobenius norm r. For m ≥ 2, trace-freeness is
    preserved by scaling (it is a linear property).

    Args:
        m: tensor index (0 <= m <= k)
        r: target Frobenius norm (> 0)
        eps: zero threshold added to denominator

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
# Internal helpers
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


# ---------------------------------------------------------------------------
# General constrained optimizer
# ---------------------------------------------------------------------------


def optimize_ratio(
    ratio_fn,
    n: int,
    k: int,
    num_steps: int,
    key: jax.Array,
    projections: tuple = (),
    lr: float = 0.01,
    num_restarts: int = 8,
    dtype=jnp.float32,
    init_fn=None,
    reproject_fn=None,
    minimize: bool = False,
    extra_params=None,
    extra_lr=None,
    extra_minimize: bool = False,
) -> dict:
    """Gradient ascent (or descent) to optimize ratio_fn over constrained k-jets.

    Each step:
      1. Compute gradient of ratio_fn w.r.t. all jet tensors (and extra_params
         if provided), via jax.grad.
      2. Gradient update: j ← j ± lr * grad  (+ for maximize, − for minimize).
      3. Re-project to enforce PDE constraint (STF for harmonic, or custom).
      4. Apply user projections in the given order.

    Multiple random restarts are run in parallel via vmap.

    Scale invariance: if ratio_fn is scale-invariant in any jet component,
    include a norm-fixing projection for that component (e.g., fix_grad_norm,
    fix_tensor_frob_norm) so the optimization is bounded.

    Projection order: projections may not commute. Norm-fixing projections
    (fix_grad_norm, fix_tensor_frob_norm) should typically come last.

    Saddle-point mode: when extra_params is provided, `minimize` and
    `extra_minimize` can be set independently, enabling convex-concave
    saddle-point problems. For example, to solve max_θ min_jet f(jet, θ):

        optimize_ratio(f, ..., extra_params=θ0,
                       minimize=True, extra_minimize=False)

    The jet is updated with sign determined by `minimize`; extra_params with
    sign determined by `extra_minimize`. The returned 'best_ratio' and
    'best_jet' are selected according to `minimize` (best from the jet's
    perspective across restarts).

    Args:
        ratio_fn: jet → scalar (or (jet, params) → scalar when extra_params
                  is provided), the objective to optimize.
                  Must be differentiable via jax.grad.
        n: spatial dimension.
        k: jet order. Jets have tensors T⁰, ..., T^k.
        num_steps: gradient ascent/descent steps per restart.
        key: JAX PRNGKey.
        projections: tuple of (jet → jet), applied after each step.
        lr: learning rate for jet tensors.
        num_restarts: number of random restarts (run in parallel via vmap).
        dtype: JAX dtype for tensor initialization.
        init_fn: callable(key, n, k, dtype) → jet used for initialization.
                 Defaults to random_harmonic_jet (harmonic jets).
                 For eigenfunction jets pass random_eigenfunction_jet
                 wrapped with the desired lam, e.g.:
                     init_fn=lambda key, n, k, dtype: random_eigenfunction_jet(
                         key, n, k, lam=1.0, dtype=dtype)
        reproject_fn: callable(jet) → jet applied after each gradient step to
                      restore the PDE constraint. Defaults to _reproject_harmonic
                      (STF projection for harmonic jets). For eigenfunction jets
                      pass pde_jet.reproject_eigenfunction.
        minimize: if True, minimize ratio_fn over jet tensors instead of
                  maximizing. The returned 'best_ratio' is then the minimum
                  found.
        extra_params: optional JAX pytree of additional parameters passed to
                      ratio_fn as a second argument: ratio_fn(j, params).
                      These are optimized jointly with the jet tensors.
                      If None, ratio_fn is called with only the jet.
        extra_lr: learning rate for extra_params. Defaults to lr.
        extra_minimize: if True, minimize over extra_params; if False (default),
                        maximize over extra_params. Only used when extra_params
                        is provided. Setting minimize=True and extra_minimize=False
                        gives a min_jet max_params saddle point; setting
                        minimize=False and extra_minimize=True gives max_jet
                        min_params. When both flags agree, the behavior is
                        equivalent to joint optimization with a single sign.

    Returns:
        dict with keys:
            'best_ratio': best ratio found across all restarts (scalar).
            'best_jet':   jet achieving the best ratio (HarmonicJet).
            'all_ratios': shape (num_restarts,) array of final ratios.
            'all_jets':   batched jet of all final jets (tensors have a
                          leading num_restarts axis).
        When extra_params is provided, also includes:
            'best_params': extra_params at the best restart.
            'all_params':  batched extra_params across all restarts.
    """
    _init = init_fn if init_fn is not None else (
        lambda key_r, n, k, dtype: random_harmonic_jet(key_r, n, k, dtype=dtype)
    )
    _reproject = reproject_fn if reproject_fn is not None else _reproject_harmonic
    _extra_lr = extra_lr if extra_lr is not None else lr

    # sign: +1 to ascend (maximize), -1 to descend (minimize)
    sign = -1.0 if minimize else 1.0

    has_extra = extra_params is not None

    if has_extra:
        param_sign = -1.0 if extra_minimize else 1.0
        grad_fn = jax.grad(ratio_fn, argnums=(0, 1))

        def _one_restart(key_r):
            j = _init(key_r, n, k, dtype)
            j = _reproject(j)
            for p in projections:
                j = p(j)

            def _step(carry, _):
                j, params = carry
                g_j, g_p = grad_fn(j, params)
                j = jax.tree_util.tree_map(
                    lambda x, dg: x + sign * lr * dg, j, g_j
                )
                params = jax.tree_util.tree_map(
                    lambda p, dp: p + param_sign * _extra_lr * dp, params, g_p
                )
                j = _reproject(j)
                for p_fn in projections:
                    j = p_fn(j)
                return (j, params), None

            (j_final, params_final), _ = jax.lax.scan(
                _step, (j, extra_params), None, length=num_steps
            )
            return ratio_fn(j_final, params_final), j_final, params_final

        keys = jax.random.split(key, num_restarts)
        all_ratios, all_jets, all_params = jax.vmap(_one_restart)(keys)

        best_idx = jnp.argmin(all_ratios) if minimize else jnp.argmax(all_ratios)
        best_ratio = jnp.min(all_ratios) if minimize else jnp.max(all_ratios)
        best_jet = jax.tree_util.tree_map(lambda x: x[best_idx], all_jets)
        best_params = jax.tree_util.tree_map(lambda x: x[best_idx], all_params)

        return {
            'best_ratio': best_ratio,
            'best_jet': best_jet,
            'best_params': best_params,
            'all_ratios': all_ratios,
            'all_jets': all_jets,
            'all_params': all_params,
        }

    else:
        grad_fn = jax.grad(ratio_fn)

        def _one_restart(key_r):
            j = _init(key_r, n, k, dtype)
            j = _reproject(j)
            for p in projections:
                j = p(j)

            def _step(j, _):
                g = grad_fn(j)
                j = jax.tree_util.tree_map(
                    lambda x, dg: x + sign * lr * dg, j, g
                )
                j = _reproject(j)
                for p in projections:
                    j = p(j)
                return j, None

            j_final, _ = jax.lax.scan(_step, j, None, length=num_steps)
            return ratio_fn(j_final), j_final

        keys = jax.random.split(key, num_restarts)
        all_ratios, all_jets = jax.vmap(_one_restart)(keys)

        best_idx = jnp.argmin(all_ratios) if minimize else jnp.argmax(all_ratios)
        best_ratio = jnp.min(all_ratios) if minimize else jnp.max(all_ratios)
        best_jet = jax.tree_util.tree_map(lambda x: x[best_idx], all_jets)

        return {
            'best_ratio': best_ratio,
            'best_jet': best_jet,
            'all_ratios': all_ratios,
            'all_jets': all_jets,
        }
