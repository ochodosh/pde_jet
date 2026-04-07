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

Riemannian gradient:
    sphere_tangent_proj(component_indices) — project gradient onto tangent spaces
    of sphere-constrained components before any optimizer step. Required for
    correctness when using momentum-based optimizers (Adam, etc.).

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
# Riemannian gradient correction
# ---------------------------------------------------------------------------


def sphere_tangent_proj(component_indices):
    """Return a tangent_proj_fn that projects gradient components onto sphere tangent spaces.

    For each index m in component_indices, the corresponding tensor T^m is
    assumed to live on a sphere (||T^m||_F = r, as enforced by fix_grad_norm
    or fix_tensor_frob_norm). The Riemannian gradient of a scalar f at T^m is
    the projection of the Euclidean gradient G^m onto the tangent space of the
    sphere at T^m:

        G_R^m = G^m - (<G^m, T^m>_F / <T^m, T^m>_F) * T^m

    where <·,·>_F is the Frobenius inner product (sum of elementwise products).

    This must be applied BEFORE any optimizer step (before momentum accumulation
    in Adam, etc.) to ensure that moment estimates live in the correct tangent
    space. Applying the Euclidean gradient to a momentum optimizer and then
    projecting to the sphere causes the accumulated moments to point partly
    off-manifold, leading to incorrect descent directions.

    Args:
        component_indices: sequence of tensor indices m (0 ≤ m ≤ k) whose
                           Frobenius norms are sphere-constrained. Typically
                           [1] when using fix_grad_norm, [2] when using
                           fix_tensor_frob_norm(2, ...), or [1, 2] for both.

    Returns:
        tangent_proj_fn: callable(j, g_j) → g_j where j is the current jet
                         and g_j is the Euclidean gradient jet. Returns a new
                         gradient jet with the specified components projected
                         onto their tangent spaces.

    Example:
        projections = (fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0))
        riem_proj = sphere_tangent_proj([1, 2])
        optimize_ratio(f, ..., projections=projections,
                       optimizer=optax.adam(0.01),
                       tangent_proj_fn=riem_proj)
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
    optimizer=None,
    tangent_proj_fn=None,
) -> dict:
    """Gradient ascent (or descent) to optimize ratio_fn over constrained k-jets.

    Each step (gradient-based optimizers):
      1. Compute gradient of ratio_fn w.r.t. all jet tensors via jax.grad.
      2. If tangent_proj_fn is provided: project gradient onto the Riemannian
         tangent space of the constraint manifold (before any momentum update).
      3. Update parameters via the chosen optimizer.
      4. Re-project to enforce PDE constraint (STF for harmonic, or custom).
      5. Apply user projections in the given order.

    Multiple random restarts are run in parallel via vmap.

    Optimizer options
    -----------------
    optimizer=None (default):
        Plain gradient ascent/descent with fixed learning rate lr.
        Step: j ← j ± lr * grad.

    optimizer=optax.GradientTransformation (e.g. optax.adam(lr)):
        Use the provided optax optimizer. The lr parameter is ignored; learning
        rate must be baked into the optimizer. For maximization, the gradient is
        negated before passing to the optimizer (optax minimizes by convention).
        Optimizer state is threaded through the scan carry.

    optimizer='lbfgs':
        Use jaxopt.LBFGS (requires jaxopt to be installed). The objective is
        wrapped as f(project(x)) so that the L-BFGS line search operates in
        the unconstrained ambient space while the evaluation always uses
        projected parameters. Not supported with extra_params.
        num_steps maps to maxiter in jaxopt.LBFGS.

    Riemannian gradient
    -------------------
    When norm-fixing projections are active (fix_grad_norm, fix_tensor_frob_norm),
    the parameter lives on a sphere. Momentum-based optimizers (Adam, etc.)
    accumulate gradient moments; if those moments include radial components that
    the projection will remove, the accumulated direction is wrong. Pass
    tangent_proj_fn=sphere_tangent_proj([m1, m2, ...]) to project each
    gradient onto the correct tangent space before the optimizer step.

    Saddle-point mode: when extra_params is provided, `minimize` and
    `extra_minimize` can be set independently, enabling convex-concave
    saddle-point problems. For example, to solve max_θ min_jet f(jet, θ):

        optimize_ratio(f, ..., extra_params=θ0,
                       minimize=True, extra_minimize=False)

    The jet is updated with sign determined by `minimize`; extra_params with
    sign determined by `extra_minimize`. When using an optax optimizer, it
    applies to the jet update only; extra_params always use plain gradient
    steps with extra_lr.

    Args:
        ratio_fn: jet → scalar (or (jet, params) → scalar when extra_params
                  is provided), the objective to optimize.
                  Must be differentiable via jax.grad.
        n: spatial dimension.
        k: jet order. Jets have tensors T⁰, ..., T^k.
        num_steps: gradient steps per restart (or maxiter for L-BFGS).
        key: JAX PRNGKey.
        projections: tuple of (jet → jet), applied after each step.
        lr: learning rate for jet tensors (plain GD only; ignored with optax).
        num_restarts: number of random restarts (run in parallel via vmap).
        dtype: JAX dtype for tensor initialization.
        init_fn: callable(key, n, k, dtype) → jet used for initialization.
                 Defaults to random_harmonic_jet.
        reproject_fn: callable(jet) → jet applied after each gradient step to
                      restore the PDE constraint. Defaults to _reproject_harmonic.
        minimize: if True, minimize ratio_fn over jet tensors instead of
                  maximizing.
        extra_params: optional JAX pytree of additional parameters passed to
                      ratio_fn as a second argument: ratio_fn(j, params).
                      Not supported with optimizer='lbfgs'.
        extra_lr: learning rate for extra_params (plain GD). Defaults to lr.
        extra_minimize: if True, minimize over extra_params.
        optimizer: None for plain GD, an optax.GradientTransformation for
                   optax-based optimization, or 'lbfgs' for jaxopt L-BFGS.
        tangent_proj_fn: callable(j, grad_j) → grad_j that projects the
                         Euclidean gradient onto the Riemannian tangent space
                         of the constraint manifold. Applied before each
                         optimizer step. Use sphere_tangent_proj(indices) to
                         construct this for sphere-constrained components.

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
    use_optax = optimizer is not None and optimizer != 'lbfgs'

    # -----------------------------------------------------------------------
    # L-BFGS path
    # -----------------------------------------------------------------------
    if optimizer == 'lbfgs':
        if has_extra:
            raise ValueError(
                "optimizer='lbfgs' does not support extra_params. "
                "Use a gradient-based optimizer for saddle-point problems."
            )
        try:
            import jaxopt
        except ImportError:
            raise ImportError(
                "jaxopt is required for optimizer='lbfgs'. "
                "Install with: pip install jaxopt"
            )

        # Wrap objective as f(project(x)): L-BFGS operates in the unconstrained
        # ambient space; projections are applied before every evaluation.
        # The gradient of this composed objective is J_project^T * grad_f,
        # which equals the Riemannian gradient for sphere projections.
        def projected_obj(j_params):
            j = _reproject(j_params)
            for p_fn in projections:
                j = p_fn(j)
            val = ratio_fn(j)
            # jaxopt.LBFGS minimizes; negate for maximization
            return -val if not minimize else val

        solver = jaxopt.LBFGS(fun=projected_obj, maxiter=num_steps)

        def _one_restart(key_r):
            j0 = _init(key_r, n, k, dtype)
            j0 = _reproject(j0)
            for p_fn in projections:
                j0 = p_fn(j0)
            lbfgs_result = solver.run(j0)
            # Final projection to ensure constraint is exactly satisfied
            j_final = _reproject(lbfgs_result.params)
            for p_fn in projections:
                j_final = p_fn(j_final)
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

    # -----------------------------------------------------------------------
    # Gradient-based paths (plain GD or optax)
    # -----------------------------------------------------------------------
    if use_optax:
        try:
            import optax as _optax
        except ImportError:
            raise ImportError(
                "optax is required for optimizer argument. "
                "Install with: pip install optax"
            )

    if has_extra:
        param_sign = -1.0 if extra_minimize else 1.0
        grad_fn = jax.grad(ratio_fn, argnums=(0, 1))

        if use_optax:
            def _one_restart(key_r):
                j = _init(key_r, n, k, dtype)
                j = _reproject(j)
                for p_fn in projections:
                    j = p_fn(j)
                opt_state = optimizer.init(j)

                def _step(carry, _):
                    j, params, opt_state = carry
                    g_j, g_p = grad_fn(j, params)
                    if tangent_proj_fn is not None:
                        g_j = tangent_proj_fn(j, g_j)
                    # optax minimizes: negate gradient for maximization
                    g_for_opt = jax.tree_util.tree_map(
                        lambda dg: -sign * dg, g_j
                    )
                    updates, new_opt_state = optimizer.update(
                        g_for_opt, opt_state, j
                    )
                    j = _optax.apply_updates(j, updates)
                    # extra_params always use plain GD with extra_lr
                    params = jax.tree_util.tree_map(
                        lambda p, dp: p + param_sign * _extra_lr * dp,
                        params, g_p,
                    )
                    j = _reproject(j)
                    for p_fn in projections:
                        j = p_fn(j)
                    return (j, params, new_opt_state), None

                (j_final, params_final, _), _ = jax.lax.scan(
                    _step, (j, extra_params, opt_state), None, length=num_steps
                )
                return ratio_fn(j_final, params_final), j_final, params_final

        else:
            def _one_restart(key_r):
                j = _init(key_r, n, k, dtype)
                j = _reproject(j)
                for p_fn in projections:
                    j = p_fn(j)

                def _step(carry, _):
                    j, params = carry
                    g_j, g_p = grad_fn(j, params)
                    if tangent_proj_fn is not None:
                        g_j = tangent_proj_fn(j, g_j)
                    j = jax.tree_util.tree_map(
                        lambda x, dg: x + sign * lr * dg, j, g_j
                    )
                    params = jax.tree_util.tree_map(
                        lambda p, dp: p + param_sign * _extra_lr * dp,
                        params, g_p,
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

        if use_optax:
            def _one_restart(key_r):
                j = _init(key_r, n, k, dtype)
                j = _reproject(j)
                for p_fn in projections:
                    j = p_fn(j)
                opt_state = optimizer.init(j)

                def _step(carry, _):
                    j, opt_state = carry
                    g = grad_fn(j)
                    if tangent_proj_fn is not None:
                        g = tangent_proj_fn(j, g)
                    g_for_opt = jax.tree_util.tree_map(
                        lambda dg: -sign * dg, g
                    )
                    updates, new_opt_state = optimizer.update(
                        g_for_opt, opt_state, j
                    )
                    j = _optax.apply_updates(j, updates)
                    j = _reproject(j)
                    for p_fn in projections:
                        j = p_fn(j)
                    return (j, new_opt_state), None

                (j_final, _), _ = jax.lax.scan(
                    _step, (j, opt_state), None, length=num_steps
                )
                return ratio_fn(j_final), j_final

        else:
            def _one_restart(key_r):
                j = _init(key_r, n, k, dtype)
                j = _reproject(j)
                for p_fn in projections:
                    j = p_fn(j)

                def _step(j, _):
                    g = grad_fn(j)
                    if tangent_proj_fn is not None:
                        g = tangent_proj_fn(j, g)
                    j = jax.tree_util.tree_map(
                        lambda x, dg: x + sign * lr * dg, j, g
                    )
                    j = _reproject(j)
                    for p_fn in projections:
                        j = p_fn(j)
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
