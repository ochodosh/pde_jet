"""
Symmetric tensor primitives.

All functions operate on jnp.ndarray of shape (n,)*m representing a rank-m
tensor on R^n. Tensors are stored in full redundant form (n^m entries), not
as arrays of independent components.

Norm convention throughout this library:
    ||T||^2_F = sum of ALL entries T_{i1...im}^2  (full-array Frobenius)

For a symmetric tensor this counts each independent component multiple times
(once per permutation), but the convention is used consistently everywhere so
all ratios are correct. Sampling on the "unit sphere" must use this same norm.
"""

import itertools
import math

import jax.numpy as jnp


def symmetrize(T: jnp.ndarray) -> jnp.ndarray:
    """Full symmetrization of a rank-m tensor.

    Mathematical definition:
        Sym(T)_{i1...im} = (1/m!) * sum_{sigma in S_m} T_{i_{sigma(1)}...i_{sigma(m)}}

    Args:
        T: shape (n,)*m

    Returns:
        Fully symmetric tensor of the same shape.
    """
    m = T.ndim
    if m == 0:
        return T
    if m == 1:
        return T
    perms = list(itertools.permutations(range(m)))
    return sum(jnp.transpose(T, perm) for perm in perms) / math.factorial(m)


def trace(T: jnp.ndarray) -> jnp.ndarray:
    """Contract the first two indices of a rank-m tensor (m >= 2).

    Mathematical definition:
        (tr T)_{i3...im} = sum_i T_{i i i3 ... im} = delta^{ab} T_{ab i3...im}

    Args:
        T: shape (n,)*m with m >= 2

    Returns:
        Shape (n,)*(m-2). For m=2, returns a scalar (shape ()).
    """
    assert T.ndim >= 2, f"trace requires rank >= 2, got rank {T.ndim}"
    return jnp.trace(T, axis1=0, axis2=1)


def full_trace_k(T: jnp.ndarray, k: int) -> jnp.ndarray:
    """Apply trace k times, contracting k pairs of indices.

    Mathematical definition:
        tr^k(T) = T with k pairs of indices contracted with delta.
    Result has rank m - 2k.

    Args:
        T: shape (n,)*m
        k: number of traces to take; 0 <= k <= m//2

    Returns:
        Shape (n,)*(m - 2k). For k=0 returns T unchanged.
    """
    for _ in range(k):
        T = trace(T)
    return T


def delta_sym(n: int, s: int) -> jnp.ndarray:
    """Fully symmetrized product of s copies of the identity (delta tensor).

    Mathematical definition:
        (delta^{otimes s})_{i1...i_{2s}} = Sym(delta_{i1 i2} * delta_{i3 i4} * ... * delta_{i_{2s-1} i_{2s}})

    For s=0, returns a scalar 1.0.
    For s=1, returns the n x n identity matrix.

    Args:
        n: dimension
        s: number of delta factors (result has rank 2s)

    Returns:
        Shape (n,)*(2s). For s=0, shape ().
    """
    if s == 0:
        return jnp.ones(())
    # Build the outer product of s identity matrices, then symmetrize.
    # Start with shape (n, n, n, n, ..., n, n) = (n,)*(2s)
    result = jnp.eye(n)
    for _ in range(s - 1):
        result = jnp.tensordot(result, jnp.eye(n), axes=0)
    return symmetrize(result)


def sym_outer(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Fully symmetrized outer product of two symmetric tensors.

    Mathematical definition:
        Sym(A otimes B)_{i1...i_{p+q}} = symmetrize over all p+q indices of
        A_{i1...ip} B_{i_{p+1}...i_{p+q}}

    Note: the symmetrization is over ALL p+q indices jointly, not just within
    each factor. This is essential for the correctness of project_tracefree.

    Args:
        A: shape (n,)*p
        B: shape (n,)*q

    Returns:
        Fully symmetric tensor of shape (n,)*(p+q).
    """
    if A.ndim == 0:
        return B * A
    if B.ndim == 0:
        return A * B
    outer = jnp.tensordot(A, B, axes=0)  # shape (n,)*(p+q)
    return symmetrize(outer)


def frobenius_sq(T: jnp.ndarray) -> jnp.ndarray:
    """Squared Frobenius norm: sum of all entries squared.

    ||T||^2_F = sum_{i1...im} T_{i1...im}^2

    For a symmetric rank-m tensor this counts each independent component
    m!/prod(alpha_i!) times (once per permutation of equal indices), but the
    convention is consistent throughout the library.

    Args:
        T: shape (n,)*m (any rank)

    Returns:
        Scalar.
    """
    return jnp.sum(T ** 2)


def contract_vector(T: jnp.ndarray, v: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Contract one index of T with vector v.

    Mathematical definition:
        (T . v)_{i2...im} = sum_{i1} T_{i1 i2 ... im} v_{i1}

    Args:
        T: shape (n,)*m with m >= 1
        v: shape (n,)
        axis: which axis of T to contract with v (default 0)

    Returns:
        Shape (n,)*(m-1).
    """
    return jnp.tensordot(T, v, axes=([axis], [0]))


def contract_matrix(T: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    """Contract the first two indices of T with matrix M.

    Mathematical definition:
        (T : M)_{i3...im} = sum_{i,j} T_{ij i3...im} M_{ij}

    Useful for expressions like T^(3)_{ijk} M_{ij} (one free index k) or
    T^(2)_{ij} M_{ij} (scalar double contraction).

    Args:
        T: shape (n,)*m with m >= 2
        M: shape (n, n)

    Returns:
        Shape (n,)*(m-2). For m=2 returns a scalar (shape ()).
    """
    assert T.ndim >= 2, f"contract_matrix requires rank >= 2, got rank {T.ndim}"
    return jnp.tensordot(M, T, axes=([0, 1], [0, 1]))
