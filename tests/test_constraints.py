"""
Tests for the constraints module.

Each test encodes a mathematical property of the projection functions.
"""

import jax
import jax.numpy as jnp
import pytest

from pde_jet import (
    clamp_u_nonneg,
    fix_grad_norm,
    fix_tensor_frob_norm,
    fix_u,
    optimize_ratio,
    project_grad_ball,
    random_harmonic_jet,
    replace_tensor,
)
from pde_jet._tensor import frobenius_sq

from examples.kato import (
    higher_kato_analytic,
    higher_kato_ratio_from_jet,
    kato_analytic,
    kato_ratio_from_jet,
)


# ---------------------------------------------------------------------------
# replace_tensor
# ---------------------------------------------------------------------------


def test_replace_tensor_changes_only_target():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=3)
    new_val = jnp.ones((3, 3))
    j2 = replace_tensor(j, 2, new_val)
    assert jnp.allclose(j2.tensors[2], new_val)
    # All other tensors unchanged
    for m in [0, 1, 3]:
        assert jnp.allclose(j2.tensors[m], j.tensors[m])


def test_replace_tensor_all_orders():
    key = jax.random.PRNGKey(1)
    j = random_harmonic_jet(key, n=3, k=3)
    for m in range(4):
        new_val = jnp.zeros((3,) * m)
        j2 = replace_tensor(j, m, new_val)
        assert jnp.allclose(j2.tensors[m], new_val)
        for other in range(4):
            if other != m:
                assert jnp.allclose(j2.tensors[other], j.tensors[other])


def test_replace_tensor_preserves_n_k():
    key = jax.random.PRNGKey(2)
    j = random_harmonic_jet(key, n=4, k=2)
    j2 = replace_tensor(j, 0, jnp.array(5.0))
    assert j2.n == j.n
    assert j2.k == j.k


# ---------------------------------------------------------------------------
# fix_u
# ---------------------------------------------------------------------------


def test_fix_u_sets_t0_exactly():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    for c in [-2.0, 0.0, 1.0, 3.5]:
        j2 = fix_u(c)(j)
        assert jnp.allclose(j2.tensors[0], jnp.array(c))


def test_fix_u_leaves_other_tensors_unchanged():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    j2 = fix_u(1.0)(j)
    assert jnp.allclose(j2.tensors[1], j.tensors[1])
    assert jnp.allclose(j2.tensors[2], j.tensors[2])


def test_fix_u_is_idempotent():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    proj = fix_u(2.0)
    j2 = proj(proj(j))
    assert jnp.allclose(j2.tensors[0], jnp.array(2.0))


# ---------------------------------------------------------------------------
# clamp_u_nonneg
# ---------------------------------------------------------------------------


def test_clamp_u_nonneg_leaves_positive_unchanged():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    # Force T⁰ to be a large positive value
    j = replace_tensor(j, 0, jnp.array(5.0))
    j2 = clamp_u_nonneg(eps=1e-6)(j)
    assert jnp.allclose(j2.tensors[0], jnp.array(5.0))


def test_clamp_u_nonneg_raises_negative():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    eps = 1e-4
    j = replace_tensor(j, 0, jnp.array(-1.0))
    j2 = clamp_u_nonneg(eps=eps)(j)
    assert jnp.allclose(j2.tensors[0], jnp.array(eps))


def test_clamp_u_nonneg_raises_below_eps():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    eps = 1e-3
    j = replace_tensor(j, 0, jnp.array(1e-6))  # < eps
    j2 = clamp_u_nonneg(eps=eps)(j)
    assert jnp.allclose(j2.tensors[0], jnp.array(eps))


def test_clamp_u_nonneg_other_tensors_unchanged():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    j = replace_tensor(j, 0, jnp.array(-1.0))
    j2 = clamp_u_nonneg()(j)
    assert jnp.allclose(j2.tensors[1], j.tensors[1])
    assert jnp.allclose(j2.tensors[2], j.tensors[2])


# ---------------------------------------------------------------------------
# fix_grad_norm
# ---------------------------------------------------------------------------


def test_fix_grad_norm_sets_norm():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    for r in [0.5, 1.0, 2.0]:
        j2 = fix_grad_norm(r)(j)
        norm = jnp.linalg.norm(j2.tensors[1])
        assert jnp.allclose(norm, r, atol=1e-6)


def test_fix_grad_norm_preserves_direction():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    T1 = j.tensors[1]
    j2 = fix_grad_norm(2.0)(j)
    T1_new = j2.tensors[1]
    # Direction preserved: T1_new / |T1_new| == T1 / |T1|
    hat_old = T1 / jnp.linalg.norm(T1)
    hat_new = T1_new / jnp.linalg.norm(T1_new)
    assert jnp.allclose(hat_old, hat_new, atol=1e-6)


def test_fix_grad_norm_fallback_on_zero():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    j = replace_tensor(j, 1, jnp.zeros(3))
    j2 = fix_grad_norm(1.5)(j)
    norm = jnp.linalg.norm(j2.tensors[1])
    assert jnp.allclose(norm, 1.5, atol=1e-6)
    # Should be r * e1
    expected = jnp.array([1.5, 0.0, 0.0])
    assert jnp.allclose(j2.tensors[1], expected, atol=1e-6)


def test_fix_grad_norm_other_tensors_unchanged():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    j2 = fix_grad_norm(1.0)(j)
    assert jnp.allclose(j2.tensors[0], j.tensors[0])
    assert jnp.allclose(j2.tensors[2], j.tensors[2])


# ---------------------------------------------------------------------------
# project_grad_ball
# ---------------------------------------------------------------------------


def test_project_grad_ball_identity_inside():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    # Scale T1 to norm 0.5, then project with r=1.0
    T1_small = j.tensors[1] / jnp.linalg.norm(j.tensors[1]) * 0.5
    j = replace_tensor(j, 1, T1_small)
    j2 = project_grad_ball(r=1.0)(j)
    assert jnp.allclose(j2.tensors[1], j.tensors[1], atol=1e-6)


def test_project_grad_ball_clips_outside():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    # Scale T1 to norm 3.0, project with r=1.0
    T1_large = j.tensors[1] / jnp.linalg.norm(j.tensors[1]) * 3.0
    j = replace_tensor(j, 1, T1_large)
    j2 = project_grad_ball(r=1.0)(j)
    assert jnp.allclose(jnp.linalg.norm(j2.tensors[1]), 1.0, atol=1e-6)


def test_project_grad_ball_preserves_direction_when_clipping():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    T1_large = j.tensors[1] / jnp.linalg.norm(j.tensors[1]) * 3.0
    j = replace_tensor(j, 1, T1_large)
    j2 = project_grad_ball(r=1.0)(j)
    hat_old = T1_large / jnp.linalg.norm(T1_large)
    hat_new = j2.tensors[1] / jnp.linalg.norm(j2.tensors[1])
    assert jnp.allclose(hat_old, hat_new, atol=1e-6)


def test_project_grad_ball_is_idempotent():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    proj = project_grad_ball(r=1.0)
    j2 = proj(j)
    j3 = proj(j2)
    assert jnp.allclose(j2.tensors[1], j3.tensors[1], atol=1e-6)


# ---------------------------------------------------------------------------
# fix_tensor_frob_norm
# ---------------------------------------------------------------------------


def test_fix_tensor_frob_norm_sets_norm():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=3)
    for m in range(4):
        for r in [0.5, 1.0, 2.0]:
            j2 = fix_tensor_frob_norm(m, r)(j)
            norm = jnp.sqrt(frobenius_sq(j2.tensors[m]))
            assert jnp.allclose(norm, r, atol=1e-5), f"m={m}, r={r}, got norm={norm}"


def test_fix_tensor_frob_norm_preserves_stf_for_m2():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    j2 = fix_tensor_frob_norm(2, 1.0)(j)
    T2 = j2.tensors[2]
    # Trace must still be zero (scaling preserves trace-freeness)
    trace_val = jnp.trace(T2)
    assert jnp.allclose(trace_val, 0.0, atol=1e-5)


def test_fix_tensor_frob_norm_other_tensors_unchanged():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=3)
    j2 = fix_tensor_frob_norm(2, 1.0)(j)
    assert jnp.allclose(j2.tensors[0], j.tensors[0])
    assert jnp.allclose(j2.tensors[1], j.tensors[1])
    assert jnp.allclose(j2.tensors[3], j.tensors[3])


# ---------------------------------------------------------------------------
# kato_ratio_from_jet and higher_kato_ratio_from_jet
# ---------------------------------------------------------------------------


def test_kato_ratio_from_jet_matches_direct():
    from examples.kato import kato_ratio_direct

    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    via_jet = kato_ratio_from_jet(j)
    direct = kato_ratio_direct(j.tensors[1], j.tensors[2])
    assert jnp.allclose(via_jet, direct, atol=1e-6)


def test_higher_kato_ratio_from_jet_matches_direct():
    from examples.kato import higher_kato_ratio_direct

    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=3)
    via_jet = higher_kato_ratio_from_jet(j)
    direct = higher_kato_ratio_direct(j.tensors[2], j.tensors[3])
    assert jnp.allclose(via_jet, direct, atol=1e-6)


def test_kato_ratio_from_jet_is_differentiable():
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    # jax.grad should not raise
    g = jax.grad(kato_ratio_from_jet)(j)
    assert g.n == j.n
    assert g.k == j.k


# ---------------------------------------------------------------------------
# optimize_ratio — convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_optimize_ratio_kato_converges(n):
    """optimize_ratio with standard projections finds K^2 ≈ (n-1)/n."""
    key = jax.random.PRNGKey(42)
    result = optimize_ratio(
        kato_ratio_from_jet,
        n=n,
        k=2,
        num_steps=500,
        key=key,
        projections=(fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        lr=0.01,
        num_restarts=8,
    )
    analytic = kato_analytic(n)
    assert result['best_ratio'] <= analytic + 1e-4, (
        f"n={n}: best_ratio={result['best_ratio']} exceeds analytic {analytic}"
    )
    assert result['best_ratio'] >= analytic - 0.02, (
        f"n={n}: best_ratio={result['best_ratio']} too far below analytic {analytic}"
    )


@pytest.mark.parametrize("n", [2, 3])
def test_optimize_ratio_higher_kato_converges(n):
    """optimize_ratio with standard projections finds higher K^2 ≈ n/(n+2)."""
    key = jax.random.PRNGKey(42)
    result = optimize_ratio(
        higher_kato_ratio_from_jet,
        n=n,
        k=3,
        num_steps=500,
        key=key,
        projections=(fix_tensor_frob_norm(2, 1.0), fix_tensor_frob_norm(3, 1.0)),
        lr=0.01,
        num_restarts=8,
    )
    analytic = higher_kato_analytic(n)
    assert result['best_ratio'] <= analytic + 1e-4
    assert result['best_ratio'] >= analytic - 0.02


def test_optimize_ratio_returns_all_ratios():
    key = jax.random.PRNGKey(0)
    result = optimize_ratio(
        kato_ratio_from_jet,
        n=3,
        k=2,
        num_steps=100,
        key=key,
        projections=(fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        num_restarts=6,
    )
    assert result['all_ratios'].shape == (6,)
    assert jnp.allclose(result['best_ratio'], jnp.max(result['all_ratios']))


# ---------------------------------------------------------------------------
# optimize_ratio — constraints change the answer
# ---------------------------------------------------------------------------


def test_fix_u_zero_drives_ratio_to_zero():
    """A ratio scaled by min(u,1) with fix_u(0) should be driven to 0."""
    def ratio_fn(j):
        return kato_ratio_from_jet(j) * jnp.minimum(j.tensors[0], 1.0)

    key = jax.random.PRNGKey(0)
    result = optimize_ratio(
        ratio_fn,
        n=3,
        k=2,
        num_steps=200,
        key=key,
        projections=(fix_u(0.0), fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        num_restarts=4,
    )
    # With T⁰ = 0, ratio_fn = kato * min(0, 1) = 0
    assert jnp.allclose(result['best_ratio'], 0.0, atol=1e-5)


def test_fix_u_large_gives_kato_ratio():
    """A ratio scaled by min(u,1) with fix_u(2.0): min(2,1)=1 so ratio = kato ratio."""
    def ratio_fn(j):
        return kato_ratio_from_jet(j) * jnp.minimum(j.tensors[0], 1.0)

    key = jax.random.PRNGKey(0)
    result = optimize_ratio(
        ratio_fn,
        n=3,
        k=2,
        num_steps=500,
        key=key,
        projections=(fix_u(2.0), fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        lr=0.01,
        num_restarts=8,
    )
    analytic = kato_analytic(3)
    # ratio_fn = kato_ratio * min(2, 1) = kato_ratio; should converge to (n-1)/n
    assert result['best_ratio'] >= analytic - 0.02


def test_project_grad_ball_keeps_t1_inside_ball():
    """With project_grad_ball(r), T¹ must stay inside the ball throughout training."""
    r = 0.5
    norms_at_final = []

    key = jax.random.PRNGKey(0)
    # Run optimization and check final T1 norms
    from pde_jet._jet import random_harmonic_jet as _rhj
    from pde_jet._constraints import _reproject_harmonic

    def ratio_fn(j):
        # Scale-sensitive: |T¹|^2 * kato_ratio so the gradient pushes T¹ outward
        return frobenius_sq(j.tensors[1]) * kato_ratio_from_jet(j)

    result = optimize_ratio(
        ratio_fn,
        n=3,
        k=2,
        num_steps=300,
        key=key,
        projections=(project_grad_ball(r), fix_tensor_frob_norm(2, 1.0)),
        num_restarts=4,
    )
    # The best ratio must be ≤ r^2 * kato_analytic (since |T¹|^2 ≤ r^2)
    assert result['best_ratio'] <= r ** 2 * kato_analytic(3) + 1e-4
