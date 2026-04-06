"""
Tests for EigenfunctionJet: jets of solutions to Δu + λu = 0.

Each test encodes a mathematical property of the eigenfunction jet.

Core constraint: tr(T^m) = −λ T^{m−2}  for m ≥ 2.
At λ=0 this reduces to trace-free (harmonic) tensors.
"""

import jax
import jax.numpy as jnp
import pytest

from pde_jet import (
    EigenfunctionJet,
    make_eigenfunction_jet,
    random_eigenfunction_jet,
    zero_eigenfunction_jet,
    optimize_ratio,
    fix_tensor_frob_norm,
    fix_grad_norm,
    random_harmonic_jet,
)
from pde_jet._tensor import trace, frobenius_sq
from pde_jet._harmonics import project_tracefree
from pde_jet._eigenfunction import _reproject_eigenfunction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_trace_constraint(j: EigenfunctionJet, atol: float = 1e-5):
    """Assert tr(T^m) = -lam * T^{m-2} for all m in 2..k."""
    for m in range(2, j.k + 1):
        tr_m = trace(j.tensors[m])                # shape (n,)*(m-2)
        expected = -j.lam * j.tensors[m - 2]
        assert jnp.allclose(tr_m, expected, atol=atol), (
            f"Trace constraint violated at m={m}: max error "
            f"{float(jnp.max(jnp.abs(tr_m - expected))):.2e}"
        )


# ---------------------------------------------------------------------------
# EigenfunctionJet pytree
# ---------------------------------------------------------------------------


def test_pytree_flatten_unflatten():
    key = jax.random.PRNGKey(0)
    j = random_eigenfunction_jet(key, n=3, k=3, lam=1.0)
    children, aux = jax.tree_util.tree_flatten(j)
    j2 = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(j), children
    )
    assert j2.n == j.n
    assert j2.k == j.k
    assert j2.lam == j.lam
    for a, b in zip(j.tensors, j2.tensors):
        assert jnp.allclose(a, b)


def test_vmap_over_eigenfunction_jets():
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    jets = jax.vmap(
        lambda key: random_eigenfunction_jet(key, n=3, k=2, lam=1.0)
    )(keys)
    assert jets.tensors[0].shape == (5,)
    assert jets.tensors[2].shape == (5, 3, 3)


# ---------------------------------------------------------------------------
# Trace constraint: tr(T^m) = -lam * T^{m-2}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.0, 1.0, -2.0, 0.5])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_trace_constraint_make_jet(n, lam):
    """make_eigenfunction_jet enforces tr(T^m) = -lam * T^{m-2} for m=2,3,4."""
    key = jax.random.PRNGKey(0)
    # Build arbitrary tensors
    tensors = tuple(
        jax.random.normal(jax.random.PRNGKey(m), (n,) * m) for m in range(5)
    )
    j = make_eigenfunction_jet(tensors, n=n, k=4, lam=lam)
    _check_trace_constraint(j, atol=1e-5)


def test_trace_constraint_m2_explicit():
    """For m=2: tr(T^2) = -lam * T^0 exactly."""
    n = 3
    lam = 2.0
    key = jax.random.PRNGKey(7)
    tensors = tuple(jax.random.normal(jax.random.PRNGKey(m), (n,) * m) for m in range(3))
    j = make_eigenfunction_jet(tensors, n=n, k=2, lam=lam)
    tr_T2 = jnp.trace(j.tensors[2])      # scalar
    expected = -lam * j.tensors[0]        # scalar
    assert jnp.allclose(tr_T2, expected, atol=1e-5)


def test_trace_constraint_m3_explicit():
    """For m=3: tr(T^3)_k = -lam * T^1_k exactly (vector equation)."""
    n = 3
    lam = 1.5
    key = jax.random.PRNGKey(8)
    tensors = tuple(jax.random.normal(jax.random.PRNGKey(m), (n,) * m) for m in range(4))
    j = make_eigenfunction_jet(tensors, n=n, k=3, lam=lam)
    tr_T3 = trace(j.tensors[3])           # shape (3,): trace of rank-3 tensor
    expected = -lam * j.tensors[1]
    assert jnp.allclose(tr_T3, expected, atol=1e-5)


def test_trace_constraint_m4_explicit():
    """For m=4: tr(T^4) = -lam * T^2, and tr^2(T^4) = lam^2 * T^0."""
    n = 3
    lam = 1.0
    tensors = tuple(jax.random.normal(jax.random.PRNGKey(m), (n,) * m) for m in range(5))
    j = make_eigenfunction_jet(tensors, n=n, k=4, lam=lam)

    # First trace
    tr_T4 = trace(j.tensors[4])           # shape (3, 3)
    assert jnp.allclose(tr_T4, -lam * j.tensors[2], atol=1e-5), (
        f"tr(T^4) != -lam*T^2; max error "
        f"{float(jnp.max(jnp.abs(tr_T4 + lam * j.tensors[2]))):.2e}"
    )

    # Second trace: tr(tr(T^4)) = -lam * tr(T^2) = lam^2 * T^0
    tr2_T4 = trace(tr_T4)                 # scalar
    expected2 = lam ** 2 * j.tensors[0]
    assert jnp.allclose(tr2_T4, expected2, atol=1e-5)


@pytest.mark.parametrize("lam", [-3.0, 0.0, 1.0, 2.5])
def test_trace_constraint_random_jet(lam):
    """random_eigenfunction_jet satisfies the trace constraint."""
    key = jax.random.PRNGKey(42)
    j = random_eigenfunction_jet(key, n=3, k=4, lam=lam)
    _check_trace_constraint(j, atol=1e-5)


# ---------------------------------------------------------------------------
# λ=0 reduces to harmonic (STF)
# ---------------------------------------------------------------------------


def test_lam_zero_reduces_to_stf():
    """At λ=0 the eigenfunction jet is trace-free (same as harmonic)."""
    n = 3
    tensors = tuple(jax.random.normal(jax.random.PRNGKey(m), (n,) * m) for m in range(5))
    j = make_eigenfunction_jet(tensors, n=n, k=4, lam=0.0)
    for m in range(2, 5):
        # Each tensor should equal project_tracefree(symmetrize(tensors[m]))
        expected = project_tracefree(
            jax.numpy.array(tensors[m] + jnp.transpose(tensors[m], (1, 0) + tuple(range(2, m))))
            / 2 if m == 2 else tensors[m]
        )
        # Simpler: just check trace = 0
        tr = trace(j.tensors[m])
        assert jnp.allclose(tr, jnp.zeros_like(tr), atol=1e-5), (
            f"At lam=0, T^{m} should be trace-free; max trace = "
            f"{float(jnp.max(jnp.abs(tr))):.2e}"
        )


# ---------------------------------------------------------------------------
# Idempotence: projecting twice gives the same result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.5])
def test_projection_idempotent(lam):
    """_reproject_eigenfunction is idempotent."""
    key = jax.random.PRNGKey(0)
    j = random_eigenfunction_jet(key, n=3, k=3, lam=lam)
    j2 = _reproject_eigenfunction(j)
    j3 = _reproject_eigenfunction(j2)
    for m in range(4):
        assert jnp.allclose(j2.tensors[m], j3.tensors[m], atol=1e-5)


# ---------------------------------------------------------------------------
# STF free data is preserved
# ---------------------------------------------------------------------------


def test_stf_part_preserved_by_projection():
    """The STF part of each tensor is preserved by the eigenfunction projection.

    The projection only adds trace-carrying (non-STF) corrections; the STF
    free data P_TF(T^m) is unchanged.
    """
    from pde_jet._harmonics import project_tracefree
    from pde_jet._tensor import symmetrize

    n, lam = 3, 2.0
    tensors = tuple(jax.random.normal(jax.random.PRNGKey(m + 10), (n,) * m) for m in range(4))
    j = make_eigenfunction_jet(tensors, n=n, k=3, lam=lam)

    for m in range(2, 4):
        stf_input = project_tracefree(symmetrize(tensors[m]))
        stf_output = project_tracefree(j.tensors[m])
        assert jnp.allclose(stf_input, stf_output, atol=1e-5), (
            f"STF part changed at m={m}; max diff "
            f"{float(jnp.max(jnp.abs(stf_input - stf_output))):.2e}"
        )


# ---------------------------------------------------------------------------
# zero_eigenfunction_jet
# ---------------------------------------------------------------------------


def test_zero_jet_is_zero():
    j = zero_eigenfunction_jet(n=3, k=3, lam=1.5)
    for T in j.tensors:
        assert jnp.allclose(T, jnp.zeros_like(T))


def test_zero_jet_satisfies_constraint():
    """Zero jet trivially satisfies tr(0) = -lam * 0 = 0."""
    j = zero_eigenfunction_jet(n=3, k=4, lam=2.0)
    _check_trace_constraint(j)


# ---------------------------------------------------------------------------
# JAX differentiability
# ---------------------------------------------------------------------------


def test_grad_through_eigenfunction_jet():
    """jax.grad can differentiate through a ratio function w.r.t. an EigenfunctionJet."""
    from pde_jet._tensor import frobenius_sq

    key = jax.random.PRNGKey(0)
    j = random_eigenfunction_jet(key, n=3, k=2, lam=1.0)

    def ratio(j):
        return frobenius_sq(j.tensors[2]) / (frobenius_sq(j.tensors[1]) + 1e-8)

    g = jax.grad(ratio)(j)
    assert isinstance(g, EigenfunctionJet)
    assert g.tensors[2].shape == (3, 3)


# ---------------------------------------------------------------------------
# optimize_ratio with EigenfunctionJet
# ---------------------------------------------------------------------------


def test_optimize_ratio_eigenfunction_kato_lam0():
    """At lam=0 (harmonic), optimize_ratio with EigenfunctionJet converges to (n-1)/n.

    The Kato bound (n-1)/n is the sharp constant for harmonic functions (lam=0),
    where T^2 is trace-free.
    """
    from examples.kato import kato_ratio_from_jet, kato_analytic

    n = 3
    key = jax.random.PRNGKey(0)
    result = optimize_ratio(
        kato_ratio_from_jet,
        n=n, k=2, num_steps=400, key=key,
        projections=(fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        lr=0.01, num_restarts=8,
        init_fn=lambda key, n, k, dtype: random_eigenfunction_jet(
            key, n, k, lam=0.0, dtype=dtype
        ),
        reproject_fn=_reproject_eigenfunction,
    )
    analytic = kato_analytic(n)
    assert result['best_ratio'] <= analytic + 1e-4
    assert result['best_ratio'] >= analytic - 0.02


def test_optimize_ratio_eigenfunction_kato_lam_nonzero():
    """optimize_ratio runs correctly on non-harmonic eigenfunction jets.

    For lam≠0, T^2 has nonzero trace (-lam*u), so the Kato ratio
    |T2 T̂1|^2/||T2||^2 is NOT bounded by (n-1)/n — it can reach up to 1
    by Cauchy-Schwarz. We just verify the optimizer runs and returns a
    valid ratio in [0, 1].
    """
    from examples.kato import kato_ratio_from_jet

    n = 3
    lam = 1.0
    key = jax.random.PRNGKey(0)
    result = optimize_ratio(
        kato_ratio_from_jet,
        n=n, k=2, num_steps=200, key=key,
        projections=(fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        lr=0.01, num_restarts=4,
        init_fn=lambda key, n, k, dtype: random_eigenfunction_jet(
            key, n, k, lam=lam, dtype=dtype
        ),
        reproject_fn=_reproject_eigenfunction,
    )
    assert 0.0 <= float(result['best_ratio']) <= 1.0 + 1e-4


def test_optimize_ratio_backward_compat():
    """optimize_ratio without init_fn/reproject_fn still works (backward compat)."""
    from examples.kato import kato_ratio_from_jet

    key = jax.random.PRNGKey(1)
    result = optimize_ratio(
        kato_ratio_from_jet,
        n=3, k=2, num_steps=200, key=key,
        projections=(fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        num_restarts=4,
    )
    assert 'best_ratio' in result
    assert result['all_ratios'].shape == (4,)
