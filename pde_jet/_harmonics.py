"""
Harmonic polynomial tools: trace-free projection and dimension formulas.

A homogeneous harmonic polynomial of degree m on R^n is characterized by its
coefficient tensor, which is a fully symmetric trace-free (STF) tensor of rank m.
"Trace-free" means every contraction of two indices with delta_{ij} vanishes.

The Fischer decomposition: every symmetric rank-m tensor T decomposes uniquely as
    T = T^STF + Sym(delta otimes R)
where T^STF is STF and R is a symmetric rank-(m-2) tensor. The function
project_tracefree computes the T^STF part.

Projection formula derivation:
    P_TF(T) = sum_{s=0}^{floor(m/2)} b(m,s,n) * sym_outer(delta^{otimes s}, tr^s(T))

where the coefficients b(m,s,n) are determined by the trace-cancellation condition.
Setting the trace of P_TF(T) to zero gives the triangular system (solved below):

    b(m, 0, n) = 1
    b(m, s, n) = (-1)^s * C(m,2s) * (2s-1)!! / prod_{j=0}^{s-1}(n + 2m - 4 - 2j)

The product in the denominator is (n+2m-4)(n+2m-6)...(n+2m-2s-2).

Verified special cases (derived independently from the trace condition):
    m=2: b(2,1,n) = -1/n                            (standard traceless part)
    m=3: b(3,1,n) = -3/(n+2)
    m=4: b(4,1,n) = -6/(n+4), b(4,2,n) = 3/((n+4)(n+2))
"""

import math

import jax.numpy as jnp

from ._tensor import delta_sym, full_trace_k, sym_outer, symmetrize, trace


def harmonic_dim(n: int, m: int) -> int:
    """Dimension of the space of homogeneous harmonic polynomials of degree m on R^n.

    Mathematical definition:
        H(n, 0) = 1
        H(n, 1) = n
        H(n, m) = C(n+m-1, m) - C(n+m-3, m-2)   for m >= 2

    This equals the dimension of STF^m(R^n), the space of fully symmetric
    trace-free rank-m tensors on R^n.

    Args:
        n: spatial dimension (>= 1)
        m: polynomial degree (>= 0)

    Returns:
        Python int.
    """
    if m == 0:
        return 1
    if m == 1:
        return n
    return math.comb(n + m - 1, m) - math.comb(n + m - 3, m - 2)


def _stf_coeff(m: int, s: int, n: int) -> float:
    """Coefficient b(m, s, n) in the STF projection formula.

    b(m, 0, n) = 1
    b(m, s, n) = (-1)^s * C(m, 2s) * (2s-1)!! / prod_{j=0}^{s-1}(n + 2m - 4 - 2j)

    where (2s-1)!! = 1 * 3 * 5 * ... * (2s-1), with (-1)!! := 1 (so s=1 gives 1).

    This is a Python-time computation returning a float; never called on traced values.

    Args:
        m: tensor rank
        s: correction order (0 <= s <= m//2)
        n: spatial dimension

    Returns:
        Python float.
    """
    if s == 0:
        return 1.0
    # C(m, 2s) * (2s-1)!!
    binom = math.comb(m, 2 * s)
    double_fact = math.prod(range(1, 2 * s, 2))  # 1 * 3 * ... * (2s-1)
    # prod_{j=0}^{s-1} (n + 2m - 4 - 2j)
    denom = math.prod(n + 2 * m - 4 - 2 * j for j in range(s))
    return ((-1) ** s) * binom * double_fact / denom


def project_tracefree(T: jnp.ndarray) -> jnp.ndarray:
    """Project a fully symmetric tensor to its trace-free (harmonic) part.

    Computes the STF component in the Fischer decomposition:
        T = P_TF(T) + Sym(delta otimes R)

    Formula:
        P_TF(T) = sum_{s=0}^{floor(m/2)} b(m,s,n) * sym_outer(delta^{otimes s}, tr^s(T))

    The loop is over the Python integer m//2, not a traced value. All array
    operations inside are JAX and jit-compatible.

    Args:
        T: shape (n,)*m, assumed fully symmetric (call symmetrize first if needed)

    Returns:
        Shape (n,)*m, fully symmetric and trace-free.
    """
    m = T.ndim
    n = T.shape[0] if m > 0 else 1

    if m <= 1:
        # Scalars and vectors have no trace constraint.
        return T

    result = jnp.zeros_like(T)
    for s in range(m // 2 + 1):
        coeff = _stf_coeff(m, s, n)
        tr_s = full_trace_k(T, s)          # shape (n,)*(m-2s), or scalar if m=2s
        d_s = delta_sym(n, s)              # shape (n,)*(2s), or scalar if s=0
        term = sym_outer(d_s, tr_s)        # shape (n,)*m
        result = result + coeff * term

    # Final symmetrize enforces exact symmetry against floating-point drift.
    return symmetrize(result)


def is_tracefree(T: jnp.ndarray, atol: float = 1e-5) -> bool:
    """Check whether T is numerically trace-free.

    A tensor is trace-free if every contraction of two indices with delta_{ij}
    vanishes. We check only the first trace (by symmetry of T, all traces are
    permutations of the same values).

    Args:
        T: shape (n,)*m with m >= 2
        atol: absolute tolerance

    Returns:
        Python bool.
    """
    if T.ndim < 2:
        return True
    t = trace(T)
    return bool(jnp.allclose(t, jnp.zeros_like(t), atol=atol))
