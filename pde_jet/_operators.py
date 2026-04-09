"""
Polynomial evaluation for a HarmonicJet.

Given the k-jet j of a harmonic function u at the origin in R^n, the Taylor
polynomial approximation is:

    p_k(x) = sum_{m=0}^{k} (1/m!) T^(m)_{i1...im} x_{i1} ... x_{im}

This approximates u(x) up to O(|x|^{k+1}).

Since HarmonicJet is a JAX pytree (tensors are leaves), the full power of JAX
autodiff applies directly:

    u_val  = evaluate_polynomial(j, x)
    grad_u = jax.grad(evaluate_polynomial, argnums=1)(j, x)    # shape (n,)
    hess_u = jax.hessian(evaluate_polynomial, argnums=1)(j, x) # shape (n, n)
    dL_dj  = jax.grad(lambda j: loss(evaluate_polynomial(j, x)))(j)
"""

import math

import jax.numpy as jnp

from ._jet import HarmonicJet


def evaluate_polynomial(j: HarmonicJet, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the degree-k Taylor polynomial of u at point x.

    p_k(x) = sum_{m=0}^{k} (1/m!) T^(m)_{i1...im} x_{i1} ... x_{im}

    The loop is over the static Python integer j.k, so this function is
    jit- and vmap-compatible. Since j is a JAX pytree, jax.grad and
    jax.hessian work with respect to both j (jet tensors) and x.

    Args:
        j: HarmonicJet with k-th order derivative data.
        x: shape (n,), evaluation point.

    Returns:
        Scalar approximation to u(x).
    """
    result = jnp.zeros((), dtype=j.tensors[0].dtype)
    for m, T in enumerate(j.tensors):
        # Contract all m indices: T_{i1...im} x_{i1} ... x_{im}
        Tx = T
        for _ in range(m):
            Tx = jnp.tensordot(Tx, x, axes=([0], [0]))
        result = result + Tx / math.factorial(m)
    return result
