"""
Tests for _tensor.py.

Each test encodes a mathematical theorem, not just a numerical check.
"""

import itertools

import jax
import jax.numpy as jnp
import pytest

from pde_jet._tensor import (
    contract_vector,
    delta_sym,
    frobenius_sq,
    full_trace_k,
    sym_outer,
    symmetrize,
    trace,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# symmetrize
# ---------------------------------------------------------------------------


def test_symmetrize_idempotent():
    """Sym(Sym(T)) = Sym(T): symmetrization is a projector."""
    key = jax.random.PRNGKey(0)
    for n, m in [(3, 2), (3, 3), (4, 3), (2, 4)]:
        T = jax.random.normal(key, (n,) * m)
        S = symmetrize(T)
        SS = symmetrize(S)
        assert jnp.allclose(S, SS, atol=1e-6), f"Failed for n={n}, m={m}"


def test_symmetrize_invariant_under_all_permutations():
    """Sym(T) is invariant under all axis permutations."""
    key = jax.random.PRNGKey(1)
    for n, m in [(3, 3), (4, 2), (2, 4)]:
        T = jax.random.normal(key, (n,) * m)
        S = symmetrize(T)
        for perm in itertools.permutations(range(m)):
            S_perm = jnp.transpose(S, perm)
            assert jnp.allclose(S, S_perm, atol=1e-6), (
                f"Not invariant under permutation {perm} for n={n}, m={m}"
            )


def test_symmetrize_rank0_rank1():
    """Scalars and vectors are unchanged by symmetrize."""
    s = jnp.array(3.14)
    v = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(symmetrize(s), s)
    assert jnp.allclose(symmetrize(v), v)


def test_symmetrize_already_symmetric():
    """If T is already symmetric, Sym(T) = T."""
    # Construct a symmetric rank-2 tensor explicitly.
    A = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert jnp.allclose(symmetrize(A), A, atol=1e-6)


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


def test_trace_linearity():
    """tr(alpha*A + beta*B) = alpha*tr(A) + beta*tr(B)."""
    key = jax.random.PRNGKey(2)
    n, m = 3, 3
    A = jax.random.normal(key, (n,) * m)
    B = jax.random.normal(jax.random.PRNGKey(3), (n,) * m)
    alpha, beta = 2.5, -1.3
    lhs = trace(alpha * A + beta * B)
    rhs = alpha * trace(A) + beta * trace(B)
    assert jnp.allclose(lhs, rhs, atol=1e-6)


def test_trace_rank2_is_scalar_sum():
    """tr(A) for rank-2 = sum of diagonal = standard matrix trace."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert jnp.allclose(trace(A), jnp.array(5.0))


def test_trace_rank3_shape():
    """tr(T) for rank-3 on R^n has shape (n,)."""
    T = jax.random.normal(jax.random.PRNGKey(4), (3, 3, 3))
    t = trace(T)
    assert t.shape == (3,), f"Expected shape (3,), got {t.shape}"


def test_full_trace_k_zero():
    """full_trace_k(T, 0) = T."""
    T = jax.random.normal(jax.random.PRNGKey(5), (3, 3, 3))
    assert jnp.allclose(full_trace_k(T, 0), T)


def test_full_trace_k_twice():
    """full_trace_k(T, 2) for rank-4 on R^n gives a scalar."""
    n = 3
    T = jax.random.normal(jax.random.PRNGKey(6), (n, n, n, n))
    result = full_trace_k(T, 2)
    assert result.shape == (), f"Expected scalar, got shape {result.shape}"


# ---------------------------------------------------------------------------
# delta_sym
# ---------------------------------------------------------------------------


def test_delta_sym_s0_is_one():
    """delta_sym(n, 0) = 1 (scalar)."""
    d = delta_sym(3, 0)
    assert d.shape == ()
    assert jnp.allclose(d, jnp.ones(()))


def test_delta_sym_s1_is_identity():
    """delta_sym(n, 1) = identity matrix."""
    for n in [2, 3, 4]:
        d = delta_sym(n, 1)
        assert d.shape == (n, n)
        assert jnp.allclose(d, jnp.eye(n), atol=1e-6)


def test_delta_sym_is_symmetric():
    """delta_sym(n, s) is invariant under all axis permutations."""
    for n, s in [(3, 2), (2, 2), (3, 1)]:
        d = delta_sym(n, s)
        for perm in itertools.permutations(range(2 * s)):
            assert jnp.allclose(d, jnp.transpose(d, perm), atol=1e-6), (
                f"delta_sym({n},{s}) not invariant under permutation {perm}"
            )


def test_delta_sym_trace():
    """tr(delta_sym(n, 1)) = n (trace of identity = dimension)."""
    for n in [2, 3, 5]:
        d = delta_sym(n, 1)
        assert jnp.allclose(trace(d), jnp.array(float(n)), atol=1e-6)


# ---------------------------------------------------------------------------
# sym_outer
# ---------------------------------------------------------------------------


def test_sym_outer_is_symmetric():
    """sym_outer(A, B) is invariant under all permutations of p+q axes."""
    key = jax.random.PRNGKey(7)
    A = symmetrize(jax.random.normal(key, (3, 3)))
    B = symmetrize(jax.random.normal(jax.random.PRNGKey(8), (3, 3, 3)))
    AB = sym_outer(A, B)
    for perm in itertools.permutations(range(5)):
        assert jnp.allclose(AB, jnp.transpose(AB, perm), atol=1e-6), (
            f"sym_outer not invariant under permutation {perm}"
        )


def test_sym_outer_scalar():
    """sym_outer(scalar, T) = scalar * T."""
    T = jax.random.normal(jax.random.PRNGKey(9), (3, 3))
    s = jnp.array(2.5)
    result = sym_outer(s, T)
    assert jnp.allclose(result, 2.5 * T, atol=1e-6)


# ---------------------------------------------------------------------------
# frobenius_sq
# ---------------------------------------------------------------------------


def test_frobenius_sq_rank1():
    """||v||^2 = sum(v^2) for a vector."""
    v = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(frobenius_sq(v), jnp.array(14.0))


def test_frobenius_sq_rank2_identity():
    """||I_n||^2_F = n (sum of all entries of identity^2)."""
    for n in [2, 3, 4]:
        assert jnp.allclose(frobenius_sq(jnp.eye(n)), jnp.array(float(n)))


# ---------------------------------------------------------------------------
# contract_vector
# ---------------------------------------------------------------------------


def test_contract_vector_rank2_contracts_first_axis():
    """contract_vector(A, v, axis=0) contracts the first index: result_j = sum_i A_{ij} v_i = (A^T v)_j.

    This is the convention used throughout the library: contract_vector always
    contracts the specified axis with v. For a non-symmetric matrix this differs
    from A @ v (which contracts the second index).
    """
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    v = jnp.array([5.0, 6.0])
    # contract_vector(A, v, axis=0) = A^T @ v
    assert jnp.allclose(contract_vector(A, v, axis=0), A.T @ v)
    # contract_vector(A, v, axis=1) = A @ v
    assert jnp.allclose(contract_vector(A, v, axis=1), A @ v)


def test_contract_vector_reduces_rank():
    """Contracting a rank-m tensor with a vector gives rank-(m-1)."""
    T = jax.random.normal(jax.random.PRNGKey(10), (3, 3, 3))
    v = jnp.ones(3)
    result = contract_vector(T, v)
    assert result.shape == (3, 3), f"Expected (3,3), got {result.shape}"
