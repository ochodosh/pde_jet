"""
Tests for _harmonics.py.

Each test encodes a mathematical theorem about the trace-free projection.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pde_jet._harmonics import _stf_coeff, harmonic_dim, is_tracefree, project_tracefree
from pde_jet._tensor import frobenius_sq, symmetrize, trace

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# harmonic_dim
# ---------------------------------------------------------------------------


def test_harmonic_dim_known_values():
    """Check H(n,m) against known values.

    H(n, 0) = 1 for all n.
    H(n, 1) = n.
    H(3, 2) = 5  (traceless symmetric 3x3 matrices: 6 - 1 = 5)
    H(3, 3) = 7  (C(5,3) - C(3,1) = 10 - 3 = 7)
    H(2, m) = 2 for all m >= 1  (2D: cos(m*theta) and sin(m*theta))
    """
    assert harmonic_dim(3, 0) == 1
    assert harmonic_dim(3, 1) == 3
    assert harmonic_dim(3, 2) == 5
    assert harmonic_dim(3, 3) == 7
    assert harmonic_dim(4, 2) == 9   # C(5,2) - C(3,0) = 10 - 1 = 9
    for n in [2, 3, 4, 5]:
        assert harmonic_dim(n, 0) == 1
        assert harmonic_dim(n, 1) == n
    for m in range(1, 6):
        assert harmonic_dim(2, m) == 2


# ---------------------------------------------------------------------------
# _stf_coeff
# ---------------------------------------------------------------------------


def test_stf_coeff_verified_cases():
    """Check b(m,s,n) against values derived independently from first principles.

    Derivation: b(m,s,n) is determined by requiring that the trace of P_TF(T)
    vanishes. For s=1: tracing the first-order correction gives the equation
        tr(T) + b(m,1,n) * (n + 2m - 4) / C(m,2) * tr(T) = 0
    => b(m,1,n) = -C(m,2) / (n + 2m - 4) = -m(m-1) / (2(n+2m-4))
    """
    tol = 1e-10
    # m=2, s=1: b = -C(2,2) / n = -1/n
    for n in [2, 3, 4, 10]:
        assert abs(_stf_coeff(2, 1, n) - (-1.0 / n)) < tol, f"Failed m=2,s=1,n={n}"
    # m=3, s=1: b = -C(3,2) / (n+2) = -3/(n+2)
    for n in [2, 3, 4]:
        assert abs(_stf_coeff(3, 1, n) - (-3.0 / (n + 2))) < tol
    # m=4, s=1: b = -C(4,2) / (n+4) = -6/(n+4)
    for n in [2, 3]:
        assert abs(_stf_coeff(4, 1, n) - (-6.0 / (n + 4))) < tol
    # m=4, s=2: b = C(4,4)*3!! / ((n+4)(n+2)) = 3/((n+4)(n+2))
    for n in [2, 3]:
        assert abs(_stf_coeff(4, 2, n) - (3.0 / ((n + 4) * (n + 2)))) < tol
    # s=0 always 1
    for m in range(6):
        assert _stf_coeff(m, 0, 3) == 1.0


# ---------------------------------------------------------------------------
# project_tracefree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,m", [
    (2, 2), (3, 2), (4, 2),
    (2, 3), (3, 3), (4, 3),
    (2, 4), (3, 4),
    (3, 5),
])
def test_project_tracefree_is_tracefree(n, m):
    """P_TF(T) is trace-free for random symmetric T."""
    T = symmetrize(jax.random.normal(jax.random.PRNGKey(n * 100 + m), (n,) * m))
    P = project_tracefree(T)
    assert is_tracefree(P, atol=1e-5), (
        f"P_TF(T) not trace-free for n={n}, m={m}; "
        f"trace norm = {float(jnp.linalg.norm(trace(P))):.2e}"
    )


@pytest.mark.parametrize("n,m", [
    (2, 2), (3, 2), (4, 2),
    (3, 3), (4, 3),
    (3, 4),
])
def test_project_tracefree_idempotent(n, m):
    """P_TF(P_TF(T)) = P_TF(T): projecting twice gives the same result."""
    T = symmetrize(jax.random.normal(jax.random.PRNGKey(n + m), (n,) * m))
    P = project_tracefree(T)
    PP = project_tracefree(P)
    assert jnp.allclose(P, PP, atol=1e-5), (
        f"P_TF not idempotent for n={n}, m={m}; "
        f"max diff = {float(jnp.max(jnp.abs(P - PP))):.2e}"
    )


def test_project_tracefree_identity_is_zero():
    """P_TF(I) = 0 for m=2: the identity is purely trace, zero harmonic part.

    Proof: I_{ij} = delta_{ij} = delta_{ij} * 1 (a pure-trace tensor).
    Its STF part is I - (1/n)*I*tr(I) = I - I = 0.
    """
    for n in [2, 3, 4]:
        I = jnp.eye(n)
        P = project_tracefree(I)
        assert jnp.allclose(P, jnp.zeros((n, n)), atol=1e-6), (
            f"P_TF(I) != 0 for n={n}; got max |entry| = {float(jnp.max(jnp.abs(P))):.2e}"
        )


def test_project_tracefree_e1_outer_e1():
    """P_TF(e1 otimes e1) = diag(1-1/n, -1/n, ..., -1/n) for m=2.

    Derivation: e1 otimes e1 has tr = 1, so
        P_TF(e1 e1^T)_{ij} = (e1 e1^T)_{ij} - (1/n) delta_{ij} * 1
                            = delta_{i1} delta_{j1} - (1/n) delta_{ij}
    Diagonal: entry (1,1) = 1 - 1/n, entries (k,k) for k>1 = -1/n.
    Off-diagonal: 0.
    """
    for n in [2, 3, 4]:
        e1 = jnp.zeros(n).at[0].set(1.0)
        T = jnp.outer(e1, e1)
        P = project_tracefree(T)
        expected_diag = jnp.array([1.0 - 1.0 / n] + [-1.0 / n] * (n - 1))
        expected = jnp.diag(expected_diag)
        assert jnp.allclose(P, expected, atol=1e-6), (
            f"P_TF(e1 e1^T) wrong for n={n}"
        )


def test_project_tracefree_stf_fixed_point():
    """If T is already trace-free, P_TF(T) = T.

    Proof: T^STF is in the image of P_TF and P_TF is a projector, so it fixes T.
    """
    for n in [2, 3, 4]:
        # Construct a known STF tensor for m=2: e1 e1^T - (1/n) I
        e1 = jnp.zeros(n).at[0].set(1.0)
        T = jnp.outer(e1, e1) - jnp.eye(n) / n
        P = project_tracefree(T)
        assert jnp.allclose(P, T, atol=1e-6), (
            f"STF tensor not fixed by P_TF for n={n}"
        )


def test_project_tracefree_decreases_norm():
    """||P_TF(T)||^2_F <= ||T||^2_F: orthogonal projection decreases norm.

    Proof: P_TF is an orthogonal projection (w.r.t. Frobenius inner product),
    so ||P_TF(T)||^2 = ||T||^2 - ||T - P_TF(T)||^2 <= ||T||^2.
    """
    key = jax.random.PRNGKey(42)
    for n, m in [(3, 2), (3, 3), (4, 2)]:
        T = symmetrize(jax.random.normal(key, (n,) * m))
        P = project_tracefree(T)
        assert float(frobenius_sq(P)) <= float(frobenius_sq(T)) + 1e-6, (
            f"Projection increased norm for n={n}, m={m}"
        )


def test_project_tracefree_dimension_via_rank():
    """The rank of P_TF (as a linear map on symmetric tensors) = harmonic_dim(n,m).

    We represent P_TF as a matrix acting on the vectorized symmetric tensors,
    then check its rank numerically.
    """
    for n, m in [(3, 2), (3, 3), (2, 2), (2, 3)]:
        dim_full = n ** m
        # Build the matrix of P_TF by applying it to each standard basis vector.
        basis_matrix = []
        for idx in range(dim_full):
            e = jnp.zeros(dim_full).at[idx].set(1.0).reshape((n,) * m)
            e_sym = symmetrize(e)
            P_e = project_tracefree(e_sym)
            basis_matrix.append(P_e.flatten())
        M = jnp.stack(basis_matrix, axis=1)  # shape (dim_full, dim_full)
        rank = int(jnp.linalg.matrix_rank(M, tol=1e-4))
        expected = harmonic_dim(n, m)
        assert rank == expected, (
            f"rank(P_TF) = {rank} but harmonic_dim({n},{m}) = {expected}"
        )
