"""
Optimizer comparison for the Kato inequality.

Compares the accuracy of five optimizer configurations in finding
the sharp Kato constant K² = (n-1)/n for harmonic functions:

    1. GD           — plain gradient ascent (current default)
    2. GD+Riem      — gradient ascent with Riemannian gradient correction
    3. Adam         — optax.adam, no Riemannian correction
    4. Adam+Riem    — optax.adam with Riemannian gradient correction
    5. L-BFGS       — jaxopt.LBFGS (second-order, via projected objective)

For each configuration we report:
    - best K² found across restarts
    - error = analytic K² - best K²  (should be ≥ 0; negative means overshoot)
    - variance across restarts (spread of all_ratios)

Each optimizer is given the same number of gradient steps per restart
(num_steps) and the same number of restarts (num_restarts).

Note on step budget: L-BFGS performs line searches (multiple function
evaluations per step), so its wall-clock cost per step is higher.
num_steps here means maxiter (L-BFGS steps), not function evaluations.

Observed results (300 steps, 16 restarts, n=2..5, GD lr=0.05, Adam lr=0.02):
    - GD and GD+Riem converged to float32 noise (~1e-7 error, ~1e-14 variance)
      on all restarts for all n. Riemannian correction made no visible difference
      here; the step size is small enough that off-manifold drift is negligible.
    - Adam (no Riem) degraded with n: best K² missed by ~4e-6 at n=4, ~6e-6
      at n=5, with several restarts stuck well below the optimum (variance ~1e-7).
    - Adam+Riem partially reduced variance but did not eliminate the best-case gap.
    - L-BFGS matched GD accuracy on best K² while collapsing restart variance to
      ~1e-13 at n=4,5 — essentially all restarts landed at the same solution.

These results are specific to the Kato landscape (smooth, compact, low-dimensional).
Re-run this script if using different ratio functions, higher jet orders, or
float64, as the relative optimizer performance can change.

Usage:
    python examples/optimizer_comparison.py
    # or from the repo root:
    python -m examples.optimizer_comparison
"""

import jax
import jax.numpy as jnp
import optax

from examples.kato import kato_analytic, kato_ratio_from_jet
from pde_jet import (
    fix_grad_norm,
    fix_tensor_frob_norm,
    optimize_ratio,
    sphere_tangent_proj,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_STEPS = 300
NUM_RESTARTS = 16
ADAM_LR = 0.02
GD_LR = 0.05
DIMENSIONS = [2, 3, 4, 5]

# Kato projections: T¹ on S^{n-1}, T² on unit STF sphere
PROJECTIONS = (fix_grad_norm(1.0), fix_tensor_frob_norm(2, 1.0))

# Riemannian correction for both sphere-constrained components
RIEM_PROJ = sphere_tangent_proj([1, 2])


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------


def run_comparison(n: int, key: jax.Array) -> dict:
    """Run all five optimizer configurations for Kato in dimension n.

    Returns dict mapping optimizer name -> dict with keys:
        'best': best K² found
        'error': analytic - best
        'var': variance of all_ratios across restarts
    """
    analytic = kato_analytic(n)
    results = {}

    configs = [
        ("GD",        dict(optimizer=None,          tangent_proj_fn=None,      lr=GD_LR)),
        ("GD+Riem",   dict(optimizer=None,          tangent_proj_fn=RIEM_PROJ, lr=GD_LR)),
        ("Adam",      dict(optimizer=optax.adam(ADAM_LR), tangent_proj_fn=None,      lr=GD_LR)),
        ("Adam+Riem", dict(optimizer=optax.adam(ADAM_LR), tangent_proj_fn=RIEM_PROJ, lr=GD_LR)),
        ("L-BFGS",    dict(optimizer='lbfgs',       tangent_proj_fn=None,      lr=GD_LR)),
    ]

    for name, kwargs in configs:
        key, subkey = jax.random.split(key)
        result = optimize_ratio(
            kato_ratio_from_jet,
            n=n,
            k=2,
            num_steps=NUM_STEPS,
            key=subkey,
            projections=PROJECTIONS,
            num_restarts=NUM_RESTARTS,
            **kwargs,
        )
        best = float(result['best_ratio'])
        all_r = result['all_ratios']
        results[name] = {
            'best': best,
            'error': analytic - best,
            'var': float(jnp.var(all_r)),
            'min_restart': float(jnp.min(all_r)),
            'max_restart': float(jnp.max(all_r)),
        }

    return analytic, results


def print_table(n: int, analytic: float, results: dict) -> None:
    header = f"\n  n={n}  analytic K² = {analytic:.6f}"
    print(header)
    print("  " + "-" * 74)
    fmt = "  {:<12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}"
    print(fmt.format("optimizer", "best K²", "error", "var", "min(K²)", "max(K²)"))
    print("  " + "-" * 74)
    for name, r in results.items():
        print(fmt.format(
            name,
            f"{r['best']:.6f}",
            f"{r['error']:+.2e}",
            f"{r['var']:.2e}",
            f"{r['min_restart']:.6f}",
            f"{r['max_restart']:.6f}",
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Kato constant K² = (n-1)/n: optimizer comparison ===")
    print(f"  Steps per restart: {NUM_STEPS},  Restarts: {NUM_RESTARTS}")
    print(f"  GD lr={GD_LR},  Adam lr={ADAM_LR}")

    key = jax.random.PRNGKey(0)
    for n in DIMENSIONS:
        key, subkey = jax.random.split(key)
        analytic, results = run_comparison(n, subkey)
        print_table(n, analytic, results)

    print("\n  Columns: best K² (max across restarts), error (analytic - best),")
    print("  var (variance of final K² across restarts), min/max K² per restart.")
    print("  Negative error = optimizer overshot analytic (numerical noise).")
