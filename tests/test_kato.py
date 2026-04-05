"""
Tests for _kato.py.

Each test encodes a theorem about the Kato inequality for harmonic functions.

Original Kato:  |grad |grad u||  <= K  |D^2 u|,  K^2 = (n-1)/n
Higher Kato:    |grad |D^2 u||   <= K' |D^3 u|,  K'^2 = n/(n+2)

Both constants are sharp for harmonic functions on R^n.
"""

import jax
import jax.numpy as jnp
import pytest

from pde_jet._harmonics import project_tracefree
from pde_jet._kato import (
    higher_kato_analytic,
    higher_kato_ratio_direct,
    kato_analytic,
    kato_optimal_T2,
    kato_ratio_direct,
    optimize_higher_kato,
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


# ===========================================================================
# Higher Kato: |grad |D^2 u|| vs |D^3 u|, sharp constant K'^2 = n/(n+2)
# ===========================================================================


# ---------------------------------------------------------------------------
# higher_kato_analytic
# ---------------------------------------------------------------------------


def test_higher_kato_analytic_known_values():
    """K'^2 = n/(n+2): check known dimensions.

    Derivation summary: the contraction map L_{T2}: STF^3 -> R^n defined by
    L(B)_k = sum_{ij} T2_{ij} B_{ijk} has (LL*)_{kk} = (1/3)||T2||^2 + (2n/3(n+2)) lambda_k^2.
    Maximizing over T3 (operator norm) then over T2 (STF, unit Frobenius norm)
    gives n/(n+2).
    """
    assert higher_kato_analytic(2) == pytest.approx(0.5)
    assert higher_kato_analytic(3) == pytest.approx(3.0 / 5)
    assert higher_kato_analytic(4) == pytest.approx(4.0 / 6)
    assert higher_kato_analytic(10) == pytest.approx(10.0 / 12)


def test_higher_kato_strictly_smaller_than_kato_for_n_geq_3():
    """For n >= 3: n/(n+2) < (n-1)/n  (higher Kato is a strictly tighter bound).

    For n = 2: n/(n+2) = 1/2 = (n-1)/n  (the two constants coincide).

    Proof: n/(n+2) < (n-1)/n iff n^2 < (n-1)(n+2) = n^2+n-2 iff 0 < n-2.
    """
    assert higher_kato_analytic(2) == pytest.approx(kato_analytic(2))
    for n in [3, 4, 5, 10]:
        assert higher_kato_analytic(n) < kato_analytic(n), (
            f"Higher Kato constant {higher_kato_analytic(n):.4f} should be < "
            f"Kato constant {kato_analytic(n):.4f} for n={n}"
        )


# ---------------------------------------------------------------------------
# higher_kato_ratio_direct
# ---------------------------------------------------------------------------


def test_higher_kato_ratio_scale_invariant():
    """R^2(alpha*T2, beta*T3) = R^2(T2, T3) for nonzero scalars."""
    key = jax.random.PRNGKey(10)
    n = 3
    T2 = project_tracefree(symmetrize(jax.random.normal(key, (n, n))))
    T3 = project_tracefree(symmetrize(jax.random.normal(jax.random.PRNGKey(11), (n, n, n))))
    R2_base = higher_kato_ratio_direct(T2, T3)
    for alpha, beta in [(3.0, 0.5), (-2.0, 1.0), (0.01, 100.0)]:
        R2_scaled = higher_kato_ratio_direct(alpha * T2, beta * T3)
        assert jnp.allclose(R2_base, R2_scaled, atol=1e-6), (
            f"Scale invariance failed for alpha={alpha}, beta={beta}"
        )


def test_higher_kato_ratio_bounded_above():
    """R^2(T2, T3) <= n/(n+2) for all T2 in STF^2, T3 in STF^3.

    This tests the higher Kato inequality itself. Verifies that the analytic
    bound n/(n+2) is correct for random STF tensor pairs.
    """
    key = jax.random.PRNGKey(12)
    for n in [2, 3, 4]:
        bound = higher_kato_analytic(n)
        for i in range(20):
            k2, k3, key = jax.random.split(key, 3)
            T2 = project_tracefree(symmetrize(jax.random.normal(k2, (n, n))))
            T3 = project_tracefree(symmetrize(jax.random.normal(k3, (n, n, n))))
            R2 = higher_kato_ratio_direct(T2, T3)
            assert float(R2) <= bound + 1e-5, (
                f"Higher Kato violated for n={n}: R^2={float(R2):.6f} > {bound:.6f}"
            )


def test_higher_kato_ratio_zero_inputs():
    """Returns 0 when T2 = 0 or T3 = 0."""
    n = 3
    T2 = jnp.diag(jnp.array([2.0, -1.0, -1.0]))
    T3 = project_tracefree(symmetrize(jax.random.normal(jax.random.PRNGKey(99), (n, n, n))))
    assert jnp.allclose(higher_kato_ratio_direct(jnp.zeros((n, n)), T3), jnp.zeros(()))
    assert jnp.allclose(higher_kato_ratio_direct(T2, jnp.zeros((n, n, n))), jnp.zeros(()))


def test_higher_kato_ratio_n2_is_constant():
    """For n=2, R^2 = 1/2 for ALL nonzero T2 in STF^2, T3 in STF^3.

    This is a special property of n=2: the LL* eigenvalues are all equal
    (both lambda_k^2 = 1/2 for ||T2||=1), so the ratio is the same for
    every nonzero pair. In higher dimensions the ratio varies and only
    reaches the maximum n/(n+2) at the optimum.

    Proof sketch: for n=2, STF^2 and STF^3 are each 2-dimensional, and
    L_{T2}(L_{T2})* = (1/2) I regardless of which unit T2 in STF^2 is chosen.
    """
    n = 2
    bound = higher_kato_analytic(n)   # = 1/2
    key = jax.random.PRNGKey(20)
    for i in range(10):
        k2, k3, key = jax.random.split(key, 3)
        T2 = project_tracefree(symmetrize(jax.random.normal(k2, (n, n))))
        T3 = project_tracefree(symmetrize(jax.random.normal(k3, (n, n, n))))
        # Only test when both are nonzero
        if float(frobenius_sq(T2)) > 1e-8 and float(frobenius_sq(T3)) > 1e-8:
            R2 = higher_kato_ratio_direct(T2, T3)
            assert jnp.allclose(R2, jnp.array(bound), atol=1e-5), (
                f"n=2 higher Kato ratio should be exactly 1/2, got {float(R2):.6f}"
            )


def test_higher_kato_explicit_n3():
    """Verify R^2 against a hand-computed example for n=3.

    Take T2 = diag(2, -1, -1) (unnormalized, trace-free) and
    T3 from u = 2x1^3 - 3x1(x2^2+x3^2), which is harmonic.

    T3: T3_{111}=12, T3_{122}=T3_{212}=T3_{221}=-6,
        T3_{133}=T3_{313}=T3_{331}=-6, all other entries = 0.

    Contraction: sum_{ij} T2_{ij} T3_{ijk} = 2*T3_{11k} - T3_{22k} - T3_{33k}.
    k=1: 2*12 - (-6) - (-6) = 36.  k=2,3: 0.

    ||T2||^2 = 4+1+1 = 6.  ||T3||^2 = 12^2 + 3*6^2 + 3*6^2 = 144+108+108 = 360.
    R^2 = 36^2 / (6 * 360) = 1296/2160 = 3/5 = n/(n+2) for n=3.

    This example achieves the sharp constant, confirming the analytic answer.
    """
    n = 3
    T2 = jnp.diag(jnp.array([2.0, -1.0, -1.0]))

    T3 = jnp.zeros((n, n, n))
    T3 = T3.at[0, 0, 0].set(12.0)
    T3 = T3.at[0, 1, 1].set(-6.0); T3 = T3.at[1, 0, 1].set(-6.0); T3 = T3.at[1, 1, 0].set(-6.0)
    T3 = T3.at[0, 2, 2].set(-6.0); T3 = T3.at[2, 0, 2].set(-6.0); T3 = T3.at[2, 2, 0].set(-6.0)

    R2 = higher_kato_ratio_direct(T2, T3)
    expected = higher_kato_analytic(n)  # 3/5
    assert jnp.allclose(R2, jnp.array(expected), atol=1e-6), (
        f"Explicit n=3 example: R^2={float(R2):.6f}, expected {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# optimize_higher_kato: numerical convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_optimize_higher_kato_converges(n):
    """Gradient ascent converges to K'^2 = n/(n+2) within tolerance."""
    result = optimize_higher_kato(
        n=n,
        num_steps=600,
        key=jax.random.PRNGKey(n + 100),
        lr=0.05,
        num_restarts=8,
    )
    expected = higher_kato_analytic(n)
    best = float(result['best_K2'])
    assert best >= expected - 1e-3, (
        f"Optimizer did not converge for n={n}: best R^2={best:.6f}, expected {expected:.6f}"
    )
    assert best <= expected + 1e-4, (
        f"Optimizer exceeded analytic bound for n={n}: best R^2={best:.6f}, expected {expected:.6f}"
    )
