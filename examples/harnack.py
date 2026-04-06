r"""
Gradient Harnack experiment: Bochner-Weitzenböck and the Cheng-Yau approach.

This module demonstrates the pde_jet library applied to differential Harnack
inequalities for positive harmonic functions on R^n.

Background:
    For a positive harmonic function u (Δu = 0), the Bochner-Weitzenböck
    identity gives:

        Δ(|∇u|²) = 2|D²u|²_F                                (1)

    (the gradient energy is superharmonic in ℝⁿ since the right side is ≥ 0).
    Equation (1) follows from the chain rule applied to the jet at any point:
    setting W = f(u, s) = s (where s = |∇u|²), one has

        ΔW|₀ = f_s * 2||H||²_F = 2||H||²_F

    since f_uu = f_us = f_ss = 0.

Cheng-Yau gradient ratio:
    A natural quantity in Cheng-Yau-type estimates is

        ratio(j) = |∇W|² / (W · ΔW)    with W = |∇u|²

    At the jet origin with normalization |g| = 1, ||H||_F = 1:

        W   = |g|² = 1
        ΔW  = 2||H||²_F = 2          (Bochner identity)
        ∇W  = 2 H g                  (chain rule: f_s = 1)
        |∇W|² = 4|Hg|²

    So ratio = 4|Hg|² / (1 · 2) = 2|Hg|².

    The supremum of this ratio over all harmonic jets with |g| = 1, ||H||_F = 1
    is the Kato constant:

        sup ratio = 2 * (n-1)/n   (same as K² = (n-1)/n, scaled by 2)

    The infimum is 0 (achieved when Hg = 0, i.e., g is in the null space of H).

This example also verifies the Bochner identity (1) numerically by checking
that laplacian_of_scalar_functional with f = s reproduces 2||H||²_F.

Usage:
    from examples.harnack import harnack_analytic_sup, optimize_harnack
    # or run the demo:
    python examples/harnack.py
"""

import jax
import jax.numpy as jnp

from pde_jet import (
    fix_grad_norm,
    fix_tensor_frob_norm,
    frobenius_sq,
    gradient_of_scalar_functional,
    laplacian_of_scalar_functional,
    optimize_ratio,
    random_harmonic_jet,
)
from pde_jet._jet import HarmonicJet


# ---------------------------------------------------------------------------
# The functional W = |∇u|² = f(u, s, q) = s
# ---------------------------------------------------------------------------

def _f_grad_sq(u, s, q):
    """f(u, s, q) = s = |∇u|². Independent of u and q."""
    return s


# ---------------------------------------------------------------------------
# Analytic results
# ---------------------------------------------------------------------------


def harnack_analytic_sup(n: int) -> float:
    """Supremum of the Cheng-Yau gradient ratio: 2(n-1)/n.

    For harmonic u on R^n with |∇u| = 1 and ||D²u||_F = 1 at the origin:

        sup_j  |∇(|∇u|²)|² / (|∇u|² · Δ(|∇u|²)) = 2(n-1)/n

    This equals twice the Kato constant K² = (n-1)/n.

    Args:
        n: spatial dimension (>= 2)

    Returns:
        Python float.
    """
    assert n >= 2
    return 2.0 * (n - 1) / n


def bochner_constant() -> float:
    """The Bochner-Weitzenböck constant: Δ(|∇u|²) = 2||D²u||²_F.

    For any harmonic function in flat R^n, the Laplacian of the gradient
    energy is exactly 2 times the squared Frobenius norm of the Hessian.
    Returns the constant 2 (independent of n).
    """
    return 2.0


# ---------------------------------------------------------------------------
# Ratio function
# ---------------------------------------------------------------------------


def harnack_ratio(j: HarmonicJet, eps: float = 1e-8) -> jnp.ndarray:
    """Cheng-Yau gradient ratio |∇W|² / (W · ΔW) where W = |∇u|².

    Requires the jet to satisfy |g| = 1 and ||H||_F = 1 (enforced via
    the projections in optimize_harnack). Under these normalizations:

        W  = 1,    ΔW = 2,    |∇W|² = 4|Hg|²

    so ratio = 2|Hg|² ∈ [0, 2(n-1)/n].

    Args:
        j: HarmonicJet with k >= 2.
        eps: threshold to avoid division by zero.

    Returns:
        Scalar in [0, 2(n-1)/n].
    """
    g = j.tensors[1]
    H = j.tensors[2]
    W = frobenius_sq(g)                         # |∇u|² at origin
    dW = laplacian_of_scalar_functional(_f_grad_sq, j)   # Δ(|∇u|²) = 2||H||²_F
    grad_W = gradient_of_scalar_functional(_f_grad_sq, j)  # ∇(|∇u|²) = 2Hg
    grad_W_sq = jnp.dot(grad_W, grad_W)        # |∇W|²
    denom = W * dW
    safe = denom > eps ** 2
    return jnp.where(safe, grad_W_sq / (denom + eps ** 2), jnp.zeros(()))


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def optimize_harnack(
    n: int,
    num_steps: int,
    key: jax.Array,
    lr: float = 0.01,
    num_restarts: int = 8,
    dtype=jnp.float32,
) -> dict:
    """Find the supremum of the Cheng-Yau gradient ratio numerically.

    Optimizes ratio = |∇(|∇u|²)|² / (|∇u|² · Δ|∇u|²) over harmonic 2-jets
    with |g| = 1 and ||H||_F = 1.

    Also finds the infimum (minimize=True) which is 0 (g in null space of H).

    Returns:
        dict with keys:
            'sup_ratio':    supremum of ratio (numeric).
            'inf_ratio':    infimum of ratio (numeric, should be ~0).
            'analytic_sup': 2(n-1)/n.
            'bochner_const': ratio ΔW / ||H||²_F (should be 2 exactly).
            'best_jet_sup': HarmonicJet achieving the supremum.
            'best_jet_inf': HarmonicJet achieving the infimum.
    """
    projections = (fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0))

    result_sup = optimize_ratio(
        harnack_ratio,
        n=n, k=2,
        num_steps=num_steps,
        key=key,
        projections=projections,
        lr=lr,
        num_restarts=num_restarts,
        dtype=dtype,
        minimize=False,
    )

    result_inf = optimize_ratio(
        harnack_ratio,
        n=n, k=2,
        num_steps=num_steps,
        key=jax.random.fold_in(key, 1),
        projections=projections,
        lr=lr,
        num_restarts=num_restarts,
        dtype=dtype,
        minimize=True,
    )

    # Verify Bochner identity at the best jet.
    j_best = result_sup['best_jet']
    H_frob_sq = float(frobenius_sq(j_best.tensors[2]))
    dW = float(laplacian_of_scalar_functional(_f_grad_sq, j_best))
    bochner_numeric = dW / (H_frob_sq + 1e-12)

    return {
        'sup_ratio': result_sup['best_ratio'],
        'inf_ratio': result_inf['best_ratio'],
        'analytic_sup': harnack_analytic_sup(n),
        'bochner_const': bochner_numeric,
        'best_jet_sup': result_sup['best_jet'],
        'best_jet_inf': result_inf['best_jet'],
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    print("=== Cheng-Yau gradient ratio: sup |∇W|²/(W·ΔW) for W = |∇u|² ===")
    print(f"    (should equal 2(n-1)/n, twice the Kato constant)\n")

    for n in [2, 3, 4]:
        result = optimize_harnack(n=n, num_steps=500, key=key, lr=0.05)
        print(
            f"  n={n}: "
            f"sup = {float(result['sup_ratio']):.6f}  (analytic 2(n-1)/n = {result['analytic_sup']:.6f}),  "
            f"inf = {float(result['inf_ratio']):.6f}  (analytic 0),  "
            f"Bochner ΔW/||H||² = {result['bochner_const']:.6f}  (exact 2)"
        )
        j_sup = result['best_jet_sup']
        g = j_sup.tensors[1]
        H = j_sup.tensors[2]
        Hg = H @ g
        print(
            f"       best jet: g = {jnp.round(g, 3)}, "
            f"|Hg| = {float(jnp.linalg.norm(Hg)):.4f}, "
            f"g·Hg = {float(jnp.dot(g, Hg)):.4f}"
        )
