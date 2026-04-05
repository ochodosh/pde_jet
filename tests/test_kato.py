"""
Tests for _kato.py.

Each test encodes a theorem about the Kato inequality for harmonic functions.
The ground truth is: K_n^2 = (n-1)/n for harmonic functions on R^n.
"""

import jax
import jax.numpy as jnp
import pytest

from pde_jet._harmonics import project_tracefree
from pde_jet._kato import (
    kato_analytic,
    kato_optimal_T2,
    kato_ratio_direct,
    optimize_kato,
)
from pde_jet._tensor import frobenius_sq, symmetrize

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# kato_analytic
# ---------------------------------------------------------------------------


def test_kato_analytic_known_values():
    """K_n^2 = (n-1)/n: check known dimensions."""
    assert kato_analytic(2) == pytest.approx(0.5)
    assert kato_analytic(3) == pytest.approx(2.0 / 3)
    assert kato_analytic(4) == pytest.approx(0.75)
    assert kato_analytic(10) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# kato_ratio_direct
# ---------------------------------------------------------------------------


def test_kato_ratio_scale_invariant():
    """K^2(alpha*T1, beta*T2) = K^2(T1, T2) for nonzero scalars.

    Proof: Both numerator and denominator scale by beta^2, and T1_hat = T1/|T1|
    is invariant under positive scaling of T1. Negative alpha only flips T1_hat,
    which doesn't change |T2 T1_hat|^2.
    """
    key = jax.random.PRNGKey(0)
    n = 3
    T1 = jax.random.normal(key, (n,))
    T2 = project_tracefree(symmetrize(jax.random.normal(jax.random.PRNGKey(1), (n, n))))
    K2_base = kato_ratio_direct(T1, T2)
    for alpha, beta in [(2.0, 3.0), (-1.0, 0.5), (100.0, 0.01)]:
        K2_scaled = kato_ratio_direct(alpha * T1, beta * T2)
        assert jnp.allclose(K2_base, K2_scaled, atol=1e-6), (
            f"Scale invariance failed for alpha={alpha}, beta={beta}"
        )


def test_kato_ratio_bounded_above_by_analytic():
    """K^2(T1, T2) <= (n-1)/n for all T1, T2 (T2 trace-free symmetric).

    This tests the Kato inequality itself. The constant (n-1)/n is the best
    possible (it is achieved by kato_optimal_T2).
    """
    key = jax.random.PRNGKey(2)
    for n in [2, 3, 4]:
        bound = kato_analytic(n)
        for i in range(20):
            k1, k2, key = jax.random.split(key, 3)
            T1 = jax.random.normal(k1, (n,))
            T2 = project_tracefree(symmetrize(jax.random.normal(k2, (n, n))))
            K2 = kato_ratio_direct(T1, T2)
            assert float(K2) <= bound + 1e-5, (
                f"Kato violated for n={n}: K^2={float(K2):.6f} > {bound:.6f}"
            )


def test_kato_ratio_zero_T1():
    """kato_ratio_direct returns 0 when T1 = 0."""
    n = 3
    T1 = jnp.zeros(n)
    T2 = jnp.diag(jnp.array([1.0, -0.5, -0.5]))
    assert jnp.allclose(kato_ratio_direct(T1, T2), jnp.zeros(()))


def test_kato_ratio_zero_T2():
    """kato_ratio_direct returns 0 when T2 = 0."""
    n = 3
    T1 = jnp.array([1.0, 0.0, 0.0])
    T2 = jnp.zeros((n, n))
    assert jnp.allclose(kato_ratio_direct(T1, T2), jnp.zeros(()))


# ---------------------------------------------------------------------------
# kato_optimal_T2
# ---------------------------------------------------------------------------


def test_kato_optimal_T2_achieves_maximum():
    """kato_optimal_T2(n) achieves K^2 = (n-1)/n with T1 = e1.

    This is the key numerical verification of the analytic formula.
    """
    for n in [2, 3, 4, 5]:
        T1 = jnp.zeros(n).at[0].set(1.0)  # e1
        T2 = kato_optimal_T2(n, dtype=jnp.float64)
        K2 = kato_ratio_direct(T1, T2)
        expected = kato_analytic(n)
        assert jnp.allclose(K2, jnp.array(expected), atol=1e-10), (
            f"kato_optimal_T2({n}) gives K^2={float(K2):.12f}, expected {expected:.12f}"
        )


def test_kato_optimal_T2_frobenius_norm_one():
    """kato_optimal_T2(n) has Frobenius norm 1."""
    for n in [2, 3, 4, 5]:
        T2 = kato_optimal_T2(n, dtype=jnp.float64)
        assert jnp.allclose(frobenius_sq(T2), jnp.ones(()), atol=1e-10), (
            f"||kato_optimal_T2({n})||_F != 1"
        )


def test_kato_optimal_T2_is_tracefree():
    """kato_optimal_T2(n) is trace-free (tr = sum of diagonal = 0)."""
    for n in [2, 3, 4, 5]:
        T2 = kato_optimal_T2(n, dtype=jnp.float64)
        tr = jnp.trace(T2)
        assert jnp.allclose(tr, jnp.zeros(()), atol=1e-10), (
            f"kato_optimal_T2({n}) has trace {float(tr):.2e}"
        )


# ---------------------------------------------------------------------------
# optimize_kato: numerical convergence to the analytic bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_optimize_kato_converges(n):
    """Gradient ascent converges to K_n^2 = (n-1)/n within tolerance.

    This test is slower (runs the optimizer) but is the key end-to-end
    validation of the library.
    """
    result = optimize_kato(
        n=n,
        num_steps=500,
        key=jax.random.PRNGKey(n),
        lr=0.05,
        num_restarts=8,
    )
    expected = kato_analytic(n)
    best = float(result['best_K2'])
    assert best >= expected - 1e-3, (
        f"Optimizer did not converge for n={n}: best K^2={best:.6f}, expected {expected:.6f}"
    )
    assert best <= expected + 1e-4, (
        f"Optimizer exceeded analytic bound for n={n}: best K^2={best:.6f}, expected {expected:.6f}"
    )
