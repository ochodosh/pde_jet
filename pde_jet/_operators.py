"""
Operators on HarmonicJet: polynomial evaluation, gradient, Hessian.

All functions take a single HarmonicJet j (representing the k-jet of a harmonic
function u at the origin in R^n) and an evaluation point x in R^n. They return
the value of some expression involving u at x, computed via the Taylor polynomial:

    p_k(x) = sum_{m=0}^{k} (1/m!) * T^(m)_{i1...im} x_{i1} ... x_{im}

This approximates u(x) up to order O(|x|^{k+1}).

Batch usage: write functions for single jets; use jax.vmap for batches.
"""

import math

import jax.numpy as jnp

from ._jet import HarmonicJet
from ._tensor import contract_vector, trace


def _contract_m_times(T: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Contract all m indices of rank-m tensor T with vector x.

    Computes T_{i1...im} x_{i1} ... x_{im} = scalar.

    Args:
        T: shape (n,)*m
        x: shape (n,)

    Returns:
        Scalar.
    """
    result = T
    for _ in range(T.ndim):
        result = contract_vector(result, x, axis=0)
    return result


def evaluate_polynomial(j: HarmonicJet, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the degree-k Taylor polynomial of u at point x.

    p_k(x) = sum_{m=0}^{k} (1/m!) T^(m)_{i1...im} x_{i1} ... x_{im}

    Args:
        j: HarmonicJet
        x: shape (n,)

    Returns:
        Scalar approximation to u(x).
    """
    result = jnp.zeros((), dtype=j.tensors[0].dtype)
    for m, T in enumerate(j.tensors):
        result = result + _contract_m_times(T, x) / math.factorial(m)
    return result


def gradient_at(j: HarmonicJet, x: jnp.ndarray) -> jnp.ndarray:
    """Gradient of the Taylor polynomial at x.

    d_i p_k(x) = sum_{m=1}^{k} (1/(m-1)!) T^(m)_{i i2...im} x_{i2} ... x_{im}

    This approximates (grad u)(x) to order O(|x|^k).

    Args:
        j: HarmonicJet
        x: shape (n,)

    Returns:
        Shape (n,).
    """
    n = j.n
    result = jnp.zeros((n,), dtype=j.tensors[0].dtype)
    for m in range(1, j.k + 1):
        T = j.tensors[m]              # shape (n,)*m
        # Contract the last (m-1) indices with x, leaving the first index free.
        # T^(m)_{i i2...im} x_{i2}...x_{im} = contract_vector applied (m-1) times
        # from the RIGHT (axis=-1 each time after the first dimension is free).
        # Equivalent: transpose so free index is last, contract from front (m-1) times.
        # Simpler: contract indices 1,...,m-1 with x.
        contracted = T
        for _ in range(m - 1):
            contracted = contract_vector(contracted, x, axis=1)
        # contracted now has shape (n,) — the free index i.
        result = result + contracted / math.factorial(m - 1)
    return result


def hessian_at(j: HarmonicJet, x: jnp.ndarray) -> jnp.ndarray:
    """Hessian of the Taylor polynomial at x.

    d_i d_j p_k(x) = sum_{m=2}^{k} (1/(m-2)!) T^(m)_{ij i3...im} x_{i3}...x_{im}

    This approximates (D^2 u)(x) to order O(|x|^{k-1}).

    Args:
        j: HarmonicJet
        x: shape (n,)

    Returns:
        Shape (n, n). For a harmonic function this matrix is trace-free (at x=0
        this is exact; at x != 0 it is approximate to the Taylor order).
    """
    n = j.n
    result = jnp.zeros((n, n), dtype=j.tensors[0].dtype)
    for m in range(2, j.k + 1):
        T = j.tensors[m]              # shape (n,)*m
        # Contract indices 2,...,m-1 (0-indexed) with x, leaving axes 0 and 1 free.
        contracted = T
        for _ in range(m - 2):
            contracted = contract_vector(contracted, x, axis=2)
        # contracted now has shape (n, n).
        result = result + contracted / math.factorial(m - 2)
    return result


def laplacian_at(j: HarmonicJet, x: jnp.ndarray) -> jnp.ndarray:
    """Laplacian of the Taylor polynomial at x.

    Δp_k(x) = sum_{m=2}^{k} (1/(m-2)!) tr(T^(m))_{i3...im} x_{i3}...x_{im}

    where tr contracts the first two indices: tr(T)_{i3...im} = T_{ii i3...im}.

    Derivation: applying Δ to the m-th term (1/m!) T_{i1...im} x_{i1}...x_{im}
    and summing over the Laplacian index yields m(m-1)/m! = 1/(m-2)! copies of
    tr(T) contracted with x^{m-2}.

    For harmonic jets, tr(T^(m)) = 0 for all m >= 2 (STF constraint), so this
    returns 0 identically. The function is included for completeness and for use
    with non-harmonic polynomial data.

    Args:
        j: HarmonicJet (or any jet-like object with .tensors and .k)
        x: shape (n,)

    Returns:
        Scalar.
    """
    result = jnp.zeros((), dtype=j.tensors[0].dtype)
    for m in range(2, j.k + 1):
        tr_T = trace(j.tensors[m])          # shape (n,)*(m-2); scalar when m=2
        result = result + _contract_m_times(tr_T, x) / math.factorial(m - 2)
    return result


