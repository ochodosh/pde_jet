"""
Tests for _jet.py: HarmonicJet data structure and constructors.
"""

import jax
import jax.numpy as jnp
import pytest

from pde_jet._harmonics import is_tracefree
from pde_jet._jet import HarmonicJet, make_harmonic_jet, random_harmonic_jet, zero_jet
from pde_jet._operators import evaluate_polynomial

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Construction and harmonic constraint
# ---------------------------------------------------------------------------


def test_make_harmonic_jet_enforces_stf():
    """make_harmonic_jet projects m>=2 tensors to STF.

    The invariant is: every tensors[m] for m >= 2 is trace-free.
    """
    key = jax.random.PRNGKey(0)
    for n in [2, 3, 4]:
        for k in [2, 3, 4]:
            tensors = tuple(
                jax.random.normal(jax.random.PRNGKey(n * 100 + k * 10 + m), (n,) * m)
                for m in range(k + 1)
            )
            j = make_harmonic_jet(tensors, n, k)
            for m in range(2, k + 1):
                assert is_tracefree(j.tensors[m], atol=1e-5), (
                    f"tensors[{m}] not STF for n={n}, k={k}"
                )


def test_make_harmonic_jet_shapes():
    """tensors[m] has shape (n,)*m for each m."""
    n, k = 3, 3
    tensors = tuple(jax.random.normal(jax.random.PRNGKey(m), (n,) * m) for m in range(k + 1))
    j = make_harmonic_jet(tensors, n, k)
    for m in range(k + 1):
        assert j.tensors[m].shape == (n,) * m, (
            f"tensors[{m}] has wrong shape {j.tensors[m].shape}"
        )


def test_zero_jet():
    """zero_jet: all tensors are zero arrays of the right shape."""
    for n in [2, 3]:
        for k in [0, 1, 2, 3]:
            j = zero_jet(n, k)
            assert j.n == n and j.k == k
            for m in range(k + 1):
                assert j.tensors[m].shape == (n,) * m
                assert jnp.allclose(j.tensors[m], jnp.zeros((n,) * m))


def test_random_harmonic_jet_stf():
    """random_harmonic_jet produces STF tensors for m >= 2."""
    key = jax.random.PRNGKey(1)
    for n in [2, 3]:
        for k in [2, 3]:
            j = random_harmonic_jet(key, n, k)
            for m in range(2, k + 1):
                assert is_tracefree(j.tensors[m], atol=1e-5), (
                    f"random jet tensors[{m}] not STF for n={n}, k={k}"
                )


# ---------------------------------------------------------------------------
# JAX pytree correctness
# ---------------------------------------------------------------------------


def test_harmonic_jet_is_valid_pytree():
    """jax.tree_util.tree_leaves(j) returns the tensor arrays."""
    j = zero_jet(3, 2)
    leaves = jax.tree_util.tree_leaves(j)
    # Should contain the tensors; n and k are aux, not leaves.
    assert len(leaves) == 3, f"Expected 3 leaves (k+1=3), got {len(leaves)}"
    for leaf in leaves:
        assert isinstance(leaf, jnp.ndarray)


def test_harmonic_jet_jit_identity():
    """jax.jit(identity)(j) returns a jet equal to j."""
    j = random_harmonic_jet(jax.random.PRNGKey(2), n=3, k=2)

    @jax.jit
    def identity(jet):
        return jet

    j_out = identity(j)
    assert j_out.n == j.n
    assert j_out.k == j.k
    for m in range(j.k + 1):
        assert jnp.allclose(j_out.tensors[m], j.tensors[m])


def test_harmonic_jet_vmap():
    """vmap over a batch of jets works correctly.

    We create a batch of jets by stacking their leaves, vmap over evaluation,
    and verify the result matches sequential evaluation.
    """
    n, k = 3, 2
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    jets = [random_harmonic_jet(k_, n, k) for k_ in keys]

    # Stack leaves to form a "batched jet".
    batched_tensors = tuple(
        jnp.stack([j.tensors[m] for j in jets], axis=0)
        for m in range(k + 1)
    )
    batched_jet = HarmonicJet(batched_tensors, n, k)

    x = jnp.ones(n) * 0.1
    vals_vmap = jax.vmap(evaluate_polynomial, in_axes=(0, None))(batched_jet, x)
    vals_seq = jnp.array([evaluate_polynomial(j, x) for j in jets])
    assert jnp.allclose(vals_vmap, vals_seq, atol=1e-5), (
        f"vmap and sequential evaluation differ: {vals_vmap} vs {vals_seq}"
    )


# ---------------------------------------------------------------------------
# evaluate_polynomial: correctness on known harmonic polynomials
# ---------------------------------------------------------------------------


def test_evaluate_polynomial_known_quadratic():
    """Evaluate the jet of u(x) = x1^2 - (1/n)|x|^2 at several points.

    For n=3: u(x) = x1^2 - (1/3)(x1^2 + x2^2 + x3^2) = (2/3)x1^2 - (1/3)x2^2 - (1/3)x3^2.
    This is harmonic: Delta u = 2 - 2 = 0. ✓

    Jet at origin:
        T^(0) = u(0) = 0
        T^(1) = grad u(0) = 0
        T^(2)_{ij} = d_i d_j u(0) = diag(4/3, -2/3, -2/3) for n=3...

    Wait: d_{11} u = 2 - 2/3 = 4/3, d_{22} u = -2/3, d_{33} u = -2/3.
    So T^(2) = diag(4/3, -2/3, -2/3). tr = 4/3 - 2/3 - 2/3 = 0. ✓

    evaluate_polynomial(j, x) = (1/2) T^(2)_{ij} x_i x_j
                                = (1/2)(4/3 x1^2 - 2/3 x2^2 - 2/3 x3^2)

    But the actual u(x) = 2/3 x1^2 - 1/3 x2^2 - 1/3 x3^2
                        = (1/2) * 4/3 * x1^2 + (1/2)*(-2/3)*x2^2 + (1/2)*(-2/3)*x3^2. ✓
    """
    n = 3
    T0 = jnp.zeros(())
    T1 = jnp.zeros(n)
    T2 = jnp.diag(jnp.array([4.0 / 3, -2.0 / 3, -2.0 / 3]))
    tensors = (T0, T1, T2)
    j = make_harmonic_jet(tensors, n, k=2)

    def u_exact(x):
        return 2.0 / 3 * x[0] ** 2 - 1.0 / 3 * x[1] ** 2 - 1.0 / 3 * x[2] ** 2

    test_points = [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([1.0, 1.0, 1.0]),
        jnp.array([0.5, -0.3, 0.7]),
    ]
    for x in test_points:
        val = evaluate_polynomial(j, x)
        expected = u_exact(x)
        assert jnp.allclose(val, expected, atol=1e-6), (
            f"evaluate_polynomial({x}) = {val}, expected {expected}"
        )


def test_evaluate_polynomial_cubic_harmonic():
    """Evaluate the jet of u(x) = x1^3 - 3 x1 x2^2 (harmonic in R^2).

    Delta u = 6 x1 - 6 x1 = 0. ✓

    Jet at origin (k=3):
        T^(0) = 0
        T^(1) = (0, 0)
        T^(2)_{ij} = d_i d_j u(0) = 0 (all 2nd derivatives vanish at 0)
        T^(3)_{ijk} = d_i d_j d_k u(0):
            d_{111} u = 6, d_{122} = d_{212} = d_{221} = -6, others = 0.
            As fully symmetric tensor: T^(3)_{111} = 6, T^(3)_{122} = T^(3)_{212} = T^(3)_{221} = -6 (divided by symmetry? No — T^(m) stores the actual derivative, NOT divided by m!)

    Wait: T^(m)_{i1...im} = d_{i1}...d_{im} u(0). So T^(3)_{111} = d_1 d_1 d_1 u = 6.
    T^(3)_{122} = d_1 d_2 d_2 u = -6. Similarly for permutations:
    T^(3)_{212} = -6, T^(3)_{221} = -6. Others = 0.

    evaluate_polynomial at degree 3: (1/3!) T^(3)_{ijk} x_i x_j x_k
        = (1/6)[T_{111} x1^3 + 3*T_{122}(x1 x2^2) + 3*T_{112}(x1^2 x2) + ...]

    By symmetry of contraction: sum_{ijk} T_{ijk} x_i x_j x_k = 6*x1^3 + 3*(-6)*x1*x2^2
    (the factor 3 comes from the 3 permutations of (1,2,2): (122),(212),(221))
    = 6 x1^3 - 18 x1 x2^2

    So evaluate_polynomial = (1/6)(6 x1^3 - 18 x1 x2^2) = x1^3 - 3 x1 x2^2 = u(x). ✓
    """
    n = 2
    T0 = jnp.zeros(())
    T1 = jnp.zeros(n)
    T2 = jnp.zeros((n, n))
    # T^(3)_{ijk}: only nonzero entries are T_{111}=6, T_{122}=T_{212}=T_{221}=-6
    T3 = jnp.zeros((n, n, n))
    T3 = T3.at[0, 0, 0].set(6.0)
    T3 = T3.at[0, 1, 1].set(-6.0)
    T3 = T3.at[1, 0, 1].set(-6.0)
    T3 = T3.at[1, 1, 0].set(-6.0)
    # T3 is already symmetric (by inspection) and trace-free (tr on (0,1): T_{ii k} = T_{00k}+T_{11k} = 0 for all k ✓)
    tensors = (T0, T1, T2, T3)
    j = make_harmonic_jet(tensors, n, k=3)

    def u_exact(x):
        return x[0] ** 3 - 3 * x[0] * x[1] ** 2

    test_points = [
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, 1.0]),
        jnp.array([2.0, -1.0]),
    ]
    for x in test_points:
        val = evaluate_polynomial(j, x)
        expected = u_exact(x)
        assert jnp.allclose(val, expected, atol=1e-6), (
            f"Cubic: evaluate_polynomial({x}) = {val:.6f}, expected {expected:.6f}"
        )
