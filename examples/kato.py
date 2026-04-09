r"""
Kato inequality example: sharp constants for harmonic functions on R^n.

This module demonstrates the pde_jet library applied to two classical
Kato-type inequalities:

Original Kato:
    |∇|∇u|| ≤ K |D²u|,  sharp constant K² = (n−1)/n

Higher Kato:
    |∇|D²u|| ≤ K' |D³u|,  sharp constant K'² = n/(n+2)

Both constants are found analytically (closed-form) and numerically
(gradient ascent via optimize_ratio from pde_jet).

Usage:
    from examples.kato import kato_analytic, optimize_kato
    # or run the demo:
    python examples/kato.py
"""

import jax
import jax.numpy as jnp

import jax

from pde_jet import (
    evaluate_polynomial,
    fix_grad_norm,
    fix_tensor_frob_norm,
    frobenius_sq,
    optimize_ratio,
    project_tracefree,
    symmetrize,
)
from pde_jet._jet import HarmonicJet


# ---------------------------------------------------------------------------
# Original Kato: |grad |grad u|| vs |D^2 u|
# ---------------------------------------------------------------------------


def kato_analytic(n: int) -> float:
    """The exact Kato constant squared: K_n^2 = (n-1)/n.

    Args:
        n: spatial dimension (>= 2)

    Returns:
        Python float.
    """
    assert n >= 2
    return (n - 1) / n


def kato_ratio_direct(
    T1: jnp.ndarray, T2: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """Kato ratio K^2 = |T2 hat_T1|^2 / ||T2||^2_F.

    Args:
        T1: shape (n,), gradient
        T2: shape (n, n), Hessian (symmetric)
        eps: zero threshold

    Returns:
        Scalar in [0, (n-1)/n].
    """
    T1_norm_sq = jnp.sum(T1 ** 2)
    T2_norm_sq = frobenius_sq(T2)
    safe = (T1_norm_sq > eps ** 2) & (T2_norm_sq > eps ** 2)
    T1_hat = T1 / jnp.sqrt(T1_norm_sq + eps ** 2)
    numerator = jnp.sum((T2 @ T1_hat) ** 2)
    ratio = numerator / (T2_norm_sq + eps ** 2)
    return jnp.where(safe, ratio, jnp.zeros(()))


def kato_ratio_from_jet(j: HarmonicJet) -> jnp.ndarray:
    """Kato ratio computed from a HarmonicJet (requires k >= 2)."""
    return kato_ratio_direct(j.tensors[1], j.tensors[2])


def kato_ratio_at_point(
    j: HarmonicJet, x: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """Kato ratio |grad|grad u||^2 / |D^2 u|^2 at point x (via Taylor approx).

    Uses the Taylor polynomial of u to evaluate the Kato ratio at a point
    x != 0. The result approximates the true ratio to order O(|x|^{k-1}).

    Args:
        j: HarmonicJet (k >= 2)
        x: shape (n,), evaluation point
        eps: threshold for degenerate cases

    Returns:
        Scalar in [0, (n-1)/n].
    """
    g = jax.grad(evaluate_polynomial, argnums=1)(j, x)
    H = jax.hessian(evaluate_polynomial, argnums=1)(j, x)
    g_norm_sq = jnp.sum(g ** 2)
    H_norm_sq = frobenius_sq(H)
    safe = (g_norm_sq > eps ** 2) & (H_norm_sq > eps ** 2)
    g_hat = g / jnp.sqrt(g_norm_sq + eps ** 2)
    numerator = jnp.sum((H @ g_hat) ** 2)
    ratio = numerator / (H_norm_sq + eps ** 2)
    return jnp.where(safe, ratio, jnp.zeros(()))


def kato_optimal_T2(n: int, dtype=jnp.float32) -> jnp.ndarray:
    """The trace-free symmetric matrix achieving K^2 = (n-1)/n with T1 = e1.

    T2* = diag(1, -1/(n-1), ..., -1/(n-1)) normalized to Frobenius norm 1.

    Args:
        n: spatial dimension (>= 2)
        dtype: JAX dtype

    Returns:
        Shape (n, n), trace-free symmetric, Frobenius norm = 1.
    """
    diag_vals = jnp.array([1.0] + [-1.0 / (n - 1)] * (n - 1), dtype=dtype)
    T2_unnorm = jnp.diag(diag_vals)
    norm = jnp.sqrt(frobenius_sq(T2_unnorm))
    return T2_unnorm / norm


def optimize_kato(
    n: int,
    num_steps: int,
    key: jax.Array,
    lr: float = 0.01,
    num_restarts: int = 8,
    dtype=jnp.float32,
) -> dict:
    """Gradient ascent to find the maximum Kato ratio numerically.

    Equivalent to optimize_ratio(kato_ratio_from_jet, n=n, k=2,
    projections=(fix_grad_norm(1.), fix_tensor_frob_norm(2, 1.)), ...).

    Returns:
        dict with keys 'best_K2', 'analytic_K2', 'all_K2'.
    """
    from pde_jet import fix_grad_norm, fix_tensor_frob_norm

    result = optimize_ratio(
        kato_ratio_from_jet,
        n=n,
        k=2,
        num_steps=num_steps,
        key=key,
        projections=(fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0)),
        lr=lr,
        num_restarts=num_restarts,
        dtype=dtype,
    )
    return {
        'best_K2': result['best_ratio'],
        'analytic_K2': kato_analytic(n),
        'all_K2': result['all_ratios'],
    }


# ---------------------------------------------------------------------------
# Higher Kato: |grad |D^2 u|| vs |D^3 u|
# ---------------------------------------------------------------------------


def higher_kato_analytic(n: int) -> float:
    r"""Sharp constant for the higher Kato inequality: K'^2 = n/(n+2).

    For harmonic u on R^n, |∇|D²u|| ≤ K'|D³u| with K'^2 = n/(n+2).

    For n=2: K'^2 = 1/2 (same as original Kato; ratio is CONSTANT).
    For n>=3: n/(n+2) < (n-1)/n (higher Kato is strictly tighter).

    Args:
        n: spatial dimension (>= 2)
    """
    assert n >= 2
    return n / (n + 2)


def higher_kato_ratio_direct(
    T2: jnp.ndarray, T3: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    r"""Higher Kato ratio R^2 = sum_k <T2, T3_{..k}>^2 / (||T2||^2 * ||T3||^2).

    Args:
        T2: shape (n, n), Hessian (symmetric, trace-free for harmonic u)
        T3: shape (n, n, n), 3rd derivative tensor (symmetric, trace-free)
        eps: zero threshold

    Returns:
        Scalar in [0, n/(n+2)].
    """
    T2_norm_sq = frobenius_sq(T2)
    T3_norm_sq = frobenius_sq(T3)
    safe = (T2_norm_sq > eps ** 2) & (T3_norm_sq > eps ** 2)
    contraction = jnp.einsum('ij,ijk->k', T2, T3)
    numerator = jnp.sum(contraction ** 2)
    ratio = numerator / ((T2_norm_sq + eps ** 2) * (T3_norm_sq + eps ** 2))
    return jnp.where(safe, ratio, jnp.zeros(()))


def higher_kato_ratio_from_jet(j: HarmonicJet) -> jnp.ndarray:
    """Higher Kato ratio computed from a HarmonicJet (requires k >= 3)."""
    return higher_kato_ratio_direct(j.tensors[2], j.tensors[3])


def optimize_higher_kato(
    n: int,
    num_steps: int,
    key: jax.Array,
    lr: float = 0.01,
    num_restarts: int = 8,
    dtype=jnp.float32,
) -> dict:
    """Gradient ascent to find the maximum higher Kato ratio numerically.

    Returns:
        dict with keys 'best_K2', 'analytic_K2', 'all_K2'.
    """
    from pde_jet import fix_tensor_frob_norm

    result = optimize_ratio(
        higher_kato_ratio_from_jet,
        n=n,
        k=3,
        num_steps=num_steps,
        key=key,
        projections=(fix_tensor_frob_norm(2, 1.0), fix_tensor_frob_norm(3, 1.0)),
        lr=lr,
        num_restarts=num_restarts,
        dtype=dtype,
    )
    return {
        'best_K2': result['best_ratio'],
        'analytic_K2': higher_kato_analytic(n),
        'all_K2': result['all_ratios'],
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    print("=== Original Kato inequality ===")
    for n in [2, 3, 4]:
        result = optimize_kato(n=n, num_steps=500, key=key, lr=0.05)
        print(
            f"  n={n}: numeric K^2 = {float(result['best_K2']):.6f},"
            f" analytic = {result['analytic_K2']:.6f}"
        )

    print("\n=== Higher Kato inequality ===")
    for n in [2, 3, 4]:
        result = optimize_higher_kato(n=n, num_steps=500, key=key, lr=0.05)
        print(
            f"  n={n}: numeric K'^2 = {float(result['best_K2']):.6f},"
            f" analytic = {result['analytic_K2']:.6f}"
        )
