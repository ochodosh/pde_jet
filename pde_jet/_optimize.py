"""
Constrained optimisation over harmonic jets.

Two public functions:

    optimize_ratio(ratio_fn, ...)
        General-purpose gradient ascent/descent over a k-jet with arbitrary
        constraint projections.  Supports plain GD, optax optimizers, and
        jaxopt L-BFGS.  Multiple restarts run in parallel via vmap.
        Restored from the pre-refactor codebase; API is unchanged.

    optimize_hierarchical(ratio_fn, ...)
        Hierarchical optimisation over jet levels: first optimise the highest
        levels with lower levels fixed, then progressively release lower
        levels.  Motivated by PDE problems where higher-order jets are
        determined by (or closely coupled to) lower-order ones.

        level_schedule: list of (optimize_levels, num_steps) pairs.
        Example: [([2], 500), ([1], 300), ([0], 200)]
          - Round 1: fix T⁰, T¹; optimise T² for 500 steps.
          - Round 2: fix T⁰; optimise T¹ for 300 steps (T² free throughout).
          - Round 3: optimise T⁰ for 200 steps (T¹, T² free throughout).

        Freezing is implemented by zeroing the gradient for frozen levels
        before each update step.  This has negligible overhead for the jet
        sizes used in practice.
"""

import jax
import jax.numpy as jnp

from ._constraints import _reproject_harmonic, replace_tensor
from ._jet import HarmonicJet, random_harmonic_jet


# ---------------------------------------------------------------------------
# optimize_ratio — general constrained optimiser
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
    """Gradient ascent/descent to optimise ratio_fn over constrained harmonic k-jets.

    Each iteration:
      1. Compute gradient of ratio_fn w.r.t. jet tensors via jax.grad.
      2. Apply tangent_proj_fn (Riemannian correction) if provided.
      3. Update parameters via the chosen optimizer.
      4. Re-project to restore the PDE constraint (STF by default).
      5. Apply user projections in order.

    Multiple restarts run in parallel via vmap.

    Optimizer options
    -----------------
    optimizer=None:
        Plain GD with fixed learning rate lr.

    optimizer=optax.GradientTransformation:
        Optax-based optimisation.  lr is ignored; embed the learning rate in
        the optimizer.  Optax minimises by convention; for maximisation the
        gradient is negated before the optimizer step.

    optimizer='lbfgs':
        jaxopt.LBFGS (requires jaxopt).  Not compatible with extra_params.

    Saddle-point mode
    -----------------
    When extra_params is not None, ratio_fn is called as ratio_fn(j, params).
    minimize controls the sign for jet updates; extra_minimize controls the
    sign for extra_params updates (always plain GD with extra_lr).

    Example — max_params min_jet f(j, params):
        optimize_ratio(f, ..., extra_params=p0,
                       minimize=True, extra_minimize=False)

    Args:
        ratio_fn:       jet → scalar (or (jet, params) → scalar).
        n:              spatial dimension.
        k:              jet order.
        num_steps:      gradient steps per restart (or maxiter for L-BFGS).
        key:            JAX PRNGKey.
        projections:    tuple of (jet → jet) applied after each step.
        lr:             learning rate for plain GD (ignored with optax).
        num_restarts:   number of independent random restarts (parallel).
        dtype:          JAX dtype for tensor initialisation.
        init_fn:        callable(key, n, k, dtype) → jet; defaults to
                        random_harmonic_jet.
        reproject_fn:   callable(jet) → jet to restore PDE constraint;
                        defaults to _reproject_harmonic (STF projection).
        minimize:       if True, minimise ratio_fn over jet tensors.
        extra_params:   optional JAX pytree of auxiliary parameters.
        extra_lr:       learning rate for extra_params; defaults to lr.
        extra_minimize: if True, minimise over extra_params.
        optimizer:      None, optax.GradientTransformation, or 'lbfgs'.
        tangent_proj_fn:callable(j, grad_j) → grad_j projecting the
                        Euclidean gradient onto the Riemannian tangent space.
                        Use sphere_tangent_proj([m1, m2, ...]) to construct.

    Returns:
        dict with keys:
            'best_ratio': best value found across restarts (scalar).
            'best_jet':   jet achieving the best value.
            'all_ratios': shape (num_restarts,) array of final values.
            'all_jets':   batched jet with a leading num_restarts axis.
        When extra_params is provided, also includes:
            'best_params': extra_params at the best restart.
            'all_params':  batched extra_params.
    """
    _init = init_fn if init_fn is not None else (
        lambda key_r, n, k, dtype: random_harmonic_jet(key_r, n, k, dtype=dtype)
    )
    _reproject = reproject_fn if reproject_fn is not None else _reproject_harmonic
    _extra_lr = extra_lr if extra_lr is not None else lr

    sign = -1.0 if minimize else 1.0
    has_extra = extra_params is not None
    use_optax = optimizer is not None and optimizer != 'lbfgs'

    # -----------------------------------------------------------------------
    # L-BFGS path
    # -----------------------------------------------------------------------
    if optimizer == 'lbfgs':
        if has_extra:
            raise ValueError(
                "optimizer='lbfgs' does not support extra_params."
            )
        try:
            import jaxopt
        except ImportError:
            raise ImportError(
                "jaxopt is required for optimizer='lbfgs'. "
                "Install with: pip install jaxopt"
            )

        def projected_obj(j_params):
            j = _reproject(j_params)
            for p_fn in projections:
                j = p_fn(j)
            val = ratio_fn(j)
            return val if minimize else -val

        solver = jaxopt.LBFGS(fun=projected_obj, maxiter=num_steps)

        def _one_restart(key_r):
            j0 = _init(key_r, n, k, dtype)
            j0 = _reproject(j0)
            for p_fn in projections:
                j0 = p_fn(j0)
            lbfgs_result = solver.run(j0)
            j_final = _reproject(lbfgs_result.params)
            for p_fn in projections:
                j_final = p_fn(j_final)
            return ratio_fn(j_final), j_final

        keys = jax.random.split(key, num_restarts)
        all_ratios, all_jets = jax.vmap(_one_restart)(keys)

        best_idx = jnp.argmin(all_ratios) if minimize else jnp.argmax(all_ratios)
        best_ratio = all_ratios[best_idx]
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
                    g_for_opt = jax.tree_util.tree_map(
                        lambda dg: -sign * dg, g_j
                    )
                    updates, new_opt_state = optimizer.update(
                        g_for_opt, opt_state, j
                    )
                    j = _optax.apply_updates(j, updates)
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
        best_ratio = all_ratios[best_idx]
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
        best_ratio = all_ratios[best_idx]
        best_jet = jax.tree_util.tree_map(lambda x: x[best_idx], all_jets)
        return {
            'best_ratio': best_ratio,
            'best_jet': best_jet,
            'all_ratios': all_ratios,
            'all_jets': all_jets,
        }


# ---------------------------------------------------------------------------
# optimize_hierarchical — level-by-level optimisation
# ---------------------------------------------------------------------------


def optimize_hierarchical(
    ratio_fn,
    n: int,
    k: int,
    level_schedule: list,
    key: jax.Array,
    projections: tuple = (),
    lr: float = 0.01,
    num_restarts: int = 8,
    minimize: bool = True,
    reproject_fn=None,
    dtype=jnp.float32,
    init_fn=None,
) -> dict:
    """Hierarchical optimisation over jet levels.

    Optimises ratio_fn by cycling through jet levels from high to low.
    At each stage, only the specified levels are updated; all other levels
    are frozen by zeroing their gradient components before the update step.

    Motivation: for nonlinear PDEs the higher-order jet data is tightly
    coupled to lower-order data, so optimising high levels first (with low
    levels fixed) finds a good initialisation for the lower-level search.
    For harmonic functions (all levels free) this is a curriculum strategy
    that often avoids poor local optima.

    Freezing is implemented by zeroing the gradient for frozen levels before
    the update.  This adds negligible overhead compared to full-gradient
    steps and is jit-compatible.

    Args:
        ratio_fn:       jet → scalar, the objective.
        n:              spatial dimension.
        k:              jet order.
        level_schedule: list of (optimize_levels, num_steps) pairs, e.g.
                        [([2], 500), ([1], 300), ([0], 200)].
                        optimize_levels is a list of integer tensor indices.
        key:            JAX PRNGKey.
        projections:    tuple of (jet → jet) applied after each step.
        lr:             fixed learning rate (plain GD only).
        num_restarts:   independent random restarts run in parallel.
        minimize:       if True, minimise ratio_fn; if False, maximise.
        reproject_fn:   restore PDE constraint after each step; defaults to
                        _reproject_harmonic.
        dtype:          JAX dtype.
        init_fn:        random initialisation; defaults to random_harmonic_jet.

    Returns:
        dict with keys:
            'best_ratio': best value found across restarts (scalar).
            'best_jet':   jet achieving the best value.
            'all_ratios': shape (num_restarts,) final values.
            'all_jets':   batched jet with a leading num_restarts axis.
    """
    _init = init_fn if init_fn is not None else (
        lambda key_r, n, k, dtype: random_harmonic_jet(key_r, n, k, dtype=dtype)
    )
    _reproject = reproject_fn if reproject_fn is not None else _reproject_harmonic

    sign = -1.0 if minimize else 1.0
    grad_fn = jax.grad(ratio_fn)

    def _freeze_grad(g_j: HarmonicJet, optimize_levels: tuple) -> HarmonicJet:
        """Zero out gradient components for levels not in optimize_levels."""
        new_tensors = []
        for m, G in enumerate(g_j.tensors):
            new_tensors.append(G if m in optimize_levels else jnp.zeros_like(G))
        return HarmonicJet(tuple(new_tensors), g_j.n, g_j.k)

    def _run_stage(j: HarmonicJet, optimize_levels: tuple, num_steps: int) -> HarmonicJet:
        """Run num_steps of gradient descent, freezing all but optimize_levels."""
        def _step(j, _):
            g = grad_fn(j)
            g = _freeze_grad(g, optimize_levels)
            j = jax.tree_util.tree_map(
                lambda x, dg: x + sign * lr * dg, j, g
            )
            j = _reproject(j)
            for p_fn in projections:
                j = p_fn(j)
            return j, None

        j_final, _ = jax.lax.scan(_step, j, None, length=num_steps)
        return j_final

    def _one_restart(key_r):
        j = _init(key_r, n, k, dtype)
        j = _reproject(j)
        for p_fn in projections:
            j = p_fn(j)
        for (opt_levels, n_steps) in level_schedule:
            j = _run_stage(j, tuple(opt_levels), n_steps)
        return ratio_fn(j), j

    keys = jax.random.split(key, num_restarts)
    all_ratios, all_jets = jax.vmap(_one_restart)(keys)

    best_idx = jnp.argmin(all_ratios) if minimize else jnp.argmax(all_ratios)
    best_ratio = all_ratios[best_idx]
    best_jet = jax.tree_util.tree_map(lambda x: x[best_idx], all_jets)
    return {
        'best_ratio': best_ratio,
        'best_jet': best_jet,
        'all_ratios': all_ratios,
        'all_jets': all_jets,
    }
