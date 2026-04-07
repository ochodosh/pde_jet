"""
Chain-rule helpers for scalar functionals of jet invariants at the origin.

For a harmonic function u with k-jet j, three natural scalar invariants at
the origin (x = 0) are:

    u          = T⁰                       (function value)
    grad_sq    = |T¹|²                    (gradient norm squared)
    hess_frob_sq = ||T²||²_F             (Hessian Frobenius norm squared)

Given W = f(u, grad_sq, hess_frob_sq), this module provides:

    gradient_of_scalar_functional(f, j)   — ∇W at x=0 (shape n)
    laplacian_of_scalar_functional(f, j)  — ΔW at x=0 (scalar)

Both are derived from first principles using the chain rule and the
harmonicity constraint (STF tensors). f must be JAX-differentiable.

Mathematical derivations (at x = 0):
--------------------------------------
Let u₀ = T⁰, g = T¹, H = T², s = |g|², q = ||H||²_F.

Gradient ∇W|₀:
    ∂_k W = f_u (∂_k u) + f_s (∂_k s) + f_q (∂_k q)

    At x=0:
        ∂_k u   = g_k
        ∂_k s   = 2 Σ_j g_j H_{jk}  = 2(H g)_k
        ∂_k q   = 2 Σ_{ij} H_{ij} (∂_k H_{ij})|₀
                = 2 Σ_{ij} H_{ij} T³_{ijk}
                = 2 [contract_matrix(T³, H)]_k        (requires k ≥ 3)

    → ∇W|₀ = f_u g + 2 f_s H@g + 2 f_q contract_matrix(T³, H)

    When k < 3 the T³ term is absent (no third-order data); if f is
    independent of q this term vanishes anyway.

Laplacian ΔW|₀ (for f = f(u, s) only; see note below):
    ΔW = f_uu|∇u|² + 2 f_us ∇u·∇s + f_ss|∇s|² + f_u Δu + f_s Δs

    At x=0 for harmonic u:
        Δu     = 0          (STF: tr(T²) = 0)
        ∇u·∇s  = 2 g^T H g
        |∇s|²  = 4 |Hg|²
        Δs     = 2||H||²_F  (via Δ(|∇u|²) = 2|D²u|², T³ trace-free)

    → ΔW|₀ = f_uu s + 4 f_us (g·Hg) + 4 f_ss |Hg|² + 2 f_s ||H||²_F

    Note on the q-dependence: computing ΔW for the hess_frob_sq variable
    requires Δ(||D²u||²_F)|₀, which involves T³ (via ∇(||H||²_F)) and T⁴
    (via the Laplacian of ||H(x)||²_F). These formulas require k ≥ 4 and
    are substantially more complex. laplacian_of_scalar_functional therefore
    provides the closed-form result under the assumption that f_q = f_sq =
    f_qq ≡ 0 (i.e., f effectively depends only on u and grad_sq). When f
    does depend on hess_frob_sq, users must provide the additional terms.
"""

import inspect

import jax
import jax.numpy as jnp

from ._jet import HarmonicJet
from ._tensor import contract_matrix, frobenius_sq


def _ensure_3args(f):
    """Wrap a 2-argument callable f(u, s) into f(u, s, q) for uniform use.

    If f already accepts 3 (or more) positional arguments, it is returned
    unchanged. If it accepts exactly 2, it is wrapped so the third argument q
    is accepted but silently ignored. This lets callers pass f(u, s) directly
    without an artificial lambda.

    Arity detection uses inspect.signature; if the signature cannot be
    determined (e.g., for built-in C functions) f is returned as-is.
    """
    try:
        n = len(inspect.signature(f).parameters)
    except (ValueError, TypeError):
        return f
    if n == 2:
        return lambda u, s, q: f(u, s)
    return f


def gradient_of_scalar_functional(f, j: HarmonicJet) -> jnp.ndarray:
    """Gradient of W = f(u, |∇u|², ||D²u||²_F) at the jet center (x=0).

    Computes ∇W|_{x=0} using the formula derived from the chain rule and the
    jet's first-, second-, and (when k ≥ 3) third-order derivative data.

    Mathematical formula:
        ∇W|₀ = f_u g + 2 f_s H@g + 2 f_q contract_matrix(T³, H)

    where g = T¹, H = T², s = |g|², q = ||H||²_F.
    The T³ term is included only when j.k ≥ 3.

    Args:
        f: callable (u, grad_sq[, hess_frob_sq]) -> scalar, JAX-differentiable.
           Accepts either 2 arguments (u, grad_sq) or 3 arguments
           (u, grad_sq, hess_frob_sq). 2-argument functions are wrapped
           automatically; the hess_frob_sq derivative is then zero.
        j: HarmonicJet with k >= 2.

    Returns:
        Shape (n,) gradient vector.
    """
    f = _ensure_3args(f)
    u0 = j.tensors[0]
    g = j.tensors[1]
    H = j.tensors[2]
    s = jnp.dot(g, g)
    q = frobenius_sq(H)

    f_u = jax.grad(f, argnums=0)(u0, s, q)
    f_s = jax.grad(f, argnums=1)(u0, s, q)
    f_q = jax.grad(f, argnums=2)(u0, s, q)

    Hg = H @ g
    result = f_u * g + 2.0 * f_s * Hg

    if j.k >= 3:
        result = result + 2.0 * f_q * contract_matrix(j.tensors[3], H)

    return result


def laplacian_of_scalar_functional(f, j: HarmonicJet) -> jnp.ndarray:
    """Laplacian of W = f(u, |∇u|², ||D²u||²_F) at the jet center (x=0).

    Computes ΔW|_{x=0} under the assumption that f does not depend on its
    third argument (hess_frob_sq) — i.e., f_q = f_sq = f_qq = 0. The q
    argument is still forwarded to f so that the same function object can be
    reused with gradient_of_scalar_functional, but only the (u, grad_sq)
    partial derivatives enter the formula.

    Mathematical formula (for harmonic u, using STF constraint):
        ΔW|₀ = f_uu s + 4 f_us (g·Hg) + 4 f_ss |Hg|² + 2 f_s ||H||²_F

    where g = T¹, H = T², s = |g|², Hg = H@g.

    See module docstring for the full derivation.

    Args:
        f: callable (u, grad_sq[, hess_frob_sq]) -> scalar, JAX-differentiable.
           Accepts either 2 arguments (u, grad_sq) or 3 arguments
           (u, grad_sq, hess_frob_sq). 2-argument functions are wrapped
           automatically. Only the (u, grad_sq) partial derivatives enter
           the Laplacian formula; hess_frob_sq dependence is ignored.
        j: HarmonicJet with k >= 2.

    Returns:
        Scalar ΔW at the origin.
    """
    f = _ensure_3args(f)
    u0 = j.tensors[0]
    g = j.tensors[1]
    H = j.tensors[2]
    s = jnp.dot(g, g)
    q = frobenius_sq(H)

    f_s  = jax.grad(f, argnums=1)(u0, s, q)
    f_uu = jax.grad(jax.grad(f, argnums=0), argnums=0)(u0, s, q)
    f_us = jax.grad(jax.grad(f, argnums=0), argnums=1)(u0, s, q)
    f_ss = jax.grad(jax.grad(f, argnums=1), argnums=1)(u0, s, q)

    Hg = H @ g
    return (f_uu * s
            + 4.0 * f_us * jnp.dot(g, Hg)
            + 4.0 * f_ss * jnp.dot(Hg, Hg)
            + 2.0 * f_s  * frobenius_sq(H))
