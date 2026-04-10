"""
Functional calculus for scalar quantities derived from a HarmonicJet.

Two interfaces are provided:

1. Closed-form (fast):  gradient_of_scalar_functional, laplacian_of_scalar_functional
   For W = f(u, |∇u|², ||D²u||²_F).  Derived from first principles via the
   chain rule and the harmonicity (STF) constraint.  f must be JAX-differentiable.

2. Autodiff-based (general):  jet_functional_gradient, jet_functional_laplacian
   For W defined as any function of the Taylor polynomial of u.  Handles MLP-
   parameterised W without requiring analytic chain-rule formulas.  Uses nested
   JAX autodiff; ~100× more compute than the closed-form version for simple f,
   but correct for any differentiable W.

Closed-form derivations (at x = 0):
--------------------------------------
Let u₀ = T⁰, g = T¹, H = T², s = |g|², q = ||H||²_F.

Gradient ∇W|₀ for W = f(u, s, q):
    ∂_k W = f_u (∂_k u) + f_s (∂_k s) + f_q (∂_k q)

    At x=0:
        ∂_k u = g_k
        ∂_k s = 2(Hg)_k
        ∂_k q = 2 [contract_matrix(T³, H)]_k   (requires k ≥ 3)

    → ∇W|₀ = f_u g + 2 f_s H@g + 2 f_q contract_matrix(T³, H)

Laplacian ΔW|₀ for W = f(u, s) (q-independent):
    ΔW = f_uu|∇u|² + 2 f_us ∇u·∇s + f_ss|∇s|² + f_u Δu + f_s Δs

    At x=0 for harmonic u (Δu = 0, Δs = 2||H||²_F):
        → ΔW|₀ = f_uu s + 4 f_us (g·Hg) + 4 f_ss |Hg|² + 2 f_s ||H||²_F

    Note: the q-dependent Laplacian requires T³ and T⁴ terms (much more
    complex). Use jet_functional_laplacian for q-dependent f.
"""

import jax
import jax.numpy as jnp

from ._jet import HarmonicJet
from ._operators import evaluate_polynomial
from ._tensor import contract_matrix, frobenius_sq


# ---------------------------------------------------------------------------
# Closed-form interface: W = f(u, |∇u|², ||D²u||²_F)
# ---------------------------------------------------------------------------


def gradient_of_scalar_functional(f, j: HarmonicJet) -> jnp.ndarray:
    """Gradient of W = f(u, |∇u|², ||D²u||²_F) at the jet centre (x=0).

    Uses exact chain-rule formulas; no numerical differentiation over x.

    Mathematical formula:
        ∇W|₀ = f_u g + 2 f_s H@g + 2 f_q contract_matrix(T³, H)

    where g = T¹, H = T², s = |g|², q = ||H||²_F.
    The T³ term is included only when j.k ≥ 3.

    Args:
        f: callable (u, grad_sq, hess_frob_sq) -> scalar, JAX-differentiable.
        j: HarmonicJet with k >= 2.

    Returns:
        Shape (n,) gradient vector.
    """
    u0 = j.tensors[0]
    g = j.tensors[1]
    H = j.tensors[2]
    s = jnp.dot(g, g)
    q = frobenius_sq(H)

    f_u, f_s, f_q = jax.grad(f, argnums=(0, 1, 2))(u0, s, q)

    Hg = H @ g
    result = f_u * g + 2.0 * f_s * Hg

    if j.k >= 3:
        result = result + 2.0 * f_q * contract_matrix(j.tensors[3], H)

    return result


def laplacian_of_scalar_functional(f, j: HarmonicJet) -> jnp.ndarray:
    """Laplacian of W = f(u, |∇u|², ||D²u||²_F) at the jet centre (x=0).

    Computes ΔW|₀ using exact chain-rule formulas.  Requires that f does
    not depend on hess_frob_sq (i.e. f_q = f_sq = f_qq = 0); the q argument
    is accepted so the same f can be passed to both gradient_ and laplacian_
    functions, but q-partial derivatives are not computed.

    Mathematical formula (harmonic u, STF constraint):
        ΔW|₀ = f_uu s + 4 f_us (g·Hg) + 4 f_ss |Hg|² + 2 f_s ||H||²_F

    where g = T¹, H = T², s = |g|², Hg = H@g.

    For q-dependent f, use jet_functional_laplacian instead.

    Args:
        f: callable (u, grad_sq, hess_frob_sq) -> scalar, JAX-differentiable.
           The q argument is forwarded but q-derivatives are not used.
        j: HarmonicJet with k >= 2.

    Returns:
        Scalar ΔW at the origin.
    """
    u0 = j.tensors[0]
    g = j.tensors[1]
    H = j.tensors[2]
    s = jnp.dot(g, g)
    q = frobenius_sq(H)

    f_s = jax.grad(f, argnums=1)(u0, s, q)
    hess = jax.hessian(f, argnums=(0, 1))(u0, s, q)
    f_uu, f_us, f_ss = hess[0][0], hess[0][1], hess[1][1]

    Hg = H @ g
    return (f_uu * s
            + 4.0 * f_us * jnp.dot(g, Hg)
            + 4.0 * f_ss * jnp.dot(Hg, Hg)
            + 2.0 * f_s  * frobenius_sq(H))


# ---------------------------------------------------------------------------
# Autodiff-based interface: W defined via the Taylor polynomial
# ---------------------------------------------------------------------------


def jet_functional_gradient(W_fn, j: HarmonicJet) -> jnp.ndarray:
    """Gradient of W at x=0 where W is defined via the Taylor polynomial.

    W_fn(eval_fn, x) must return a scalar given:
        eval_fn: callable x -> u(x) using the Taylor polynomial of j
        x:       evaluation point (shape (n,))

    Example for W = |∇u|²:
        W_fn = lambda eval_fn, x: jnp.dot(jax.grad(eval_fn)(x), jax.grad(eval_fn)(x))

    Example for W = |∇ log u|² = |∇u|²/u²:
        W_fn = lambda eval_fn, x: jnp.dot(jax.grad(eval_fn)(x), jax.grad(eval_fn)(x)) / eval_fn(x)**2

    Uses nested JAX autodiff (differentiating W_fn(., x) w.r.t. x), which is
    correct for any differentiable W_fn but ~100× slower than the closed-form
    version for simple functionals.  Use this when W is parameterised by an
    MLP or involves q = ||D²u||²_F.

    Args:
        W_fn: callable (eval_fn, x) -> scalar.
        j:    HarmonicJet.

    Returns:
        Shape (n,) gradient of W at x=0.
    """
    x0 = jnp.zeros(j.n, dtype=j.tensors[0].dtype)

    def W_at(x):
        return W_fn(lambda z: evaluate_polynomial(j, z), x)

    return jax.grad(W_at)(x0)


def jet_functional_grad_and_laplacian(W_fn, j: HarmonicJet):
    """Gradient and Laplacian of W at x=0, computed together.

    Same interface as jet_functional_gradient / jet_functional_laplacian, but
    builds the polynomial closure once and reuses it for both computations.

    Args:
        W_fn: callable (eval_fn, x) -> scalar.
        j:    HarmonicJet.

    Returns:
        (grad_W, lap_W): shape (n,) gradient and scalar Laplacian.
    """
    x0 = jnp.zeros(j.n, dtype=j.tensors[0].dtype)

    def W_at(x):
        return W_fn(lambda z: evaluate_polynomial(j, z), x)

    grad_W = jax.grad(W_at)(x0)
    lap_W = jnp.trace(jax.hessian(W_at)(x0))
    return grad_W, lap_W


def jet_functional_laplacian(W_fn, j: HarmonicJet) -> jnp.ndarray:
    """Laplacian of W at x=0 where W is defined via the Taylor polynomial.

    Same interface as jet_functional_gradient.  Computes ΔW = tr(∇²W) at x=0
    via nested JAX autodiff.

    Example for W = |∇ log u|²:
        W_fn = lambda eval_fn, x: jnp.dot(jax.grad(eval_fn)(x), jax.grad(eval_fn)(x)) / eval_fn(x)**2

    Args:
        W_fn: callable (eval_fn, x) -> scalar.
        j:    HarmonicJet.

    Returns:
        Scalar ΔW at the origin.
    """
    x0 = jnp.zeros(j.n, dtype=j.tensors[0].dtype)

    def W_at(x):
        return W_fn(lambda z: evaluate_polynomial(j, z), x)

    return jnp.trace(jax.hessian(W_at)(x0))
