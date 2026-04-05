r"""
Kato inequality: |grad |grad u|| <= K |D^2 u| for harmonic functions on R^n.

Mathematical background:
    At any point x where grad u != 0, the ratio is:
        K^2(T1, T2) = |T2 * hat_T1|^2 / ||T2||^2_F
    where T1 = grad u (1-jet), T2 = D^2 u (2-jet, STF), hat_T1 = T1/|T1|.

    The sharp constant satisfies:
        K_n^2 = sup K^2(T1, T2) over all T1 in R^n \ {0},
                                    T2 in STF^2(R^n) \ {0}
             = (n - 1) / n

    Proof: By O(n) invariance fix hat_T1 = e1. Then:
        K^2 = ||T2 e1||^2 / ||T2||^2_F
    Maximize over trace-free symmetric T2 with ||T2||_F = 1. This is equivalent
    to finding the largest eigenvalue of the map T2 -> T2 e1 from the STF unit
    sphere to R^n. By a Lagrange multiplier calculation, the maximum is (n-1)/n,
    achieved by T2 = diag(1, -1/(n-1), ..., -1/(n-1)) (normalized).

    Verification: for n=2, K^2 = 1/2. For n=3, K^2 = 2/3. For n -> inf, K -> 1.
"""

import jax
import jax.numpy as jnp

from ._harmonics import project_tracefree
from ._tensor import frobenius_sq


def kato_analytic(n: int) -> float:
    """The exact Kato constant squared: K_n^2 = (n-1)/n.

    Args:
        n: spatial dimension (>= 2)

    Returns:
        Python float.
    """
    assert n >= 2, f"Kato inequality is trivial for n < 2; got n={n}"
    return (n - 1) / n


def kato_ratio_direct(
    T1: jnp.ndarray, T2: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """Kato ratio K^2 = |T2 hat_T1|^2 / ||T2||^2_F.

    Works directly with gradient T1 (shape (n,)) and Hessian T2 (shape (n,n)).
    T2 need not be pre-projected; any symmetric matrix is accepted.

    The ratio is scale-invariant: K^2(alpha*T1, beta*T2) = K^2(T1, T2) for
    nonzero scalars alpha, beta.

    Returns 0.0 if |T1| < eps or ||T2||_F < eps.

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


def kato_optimal_T2(n: int, dtype=jnp.float32) -> jnp.ndarray:
    """The trace-free symmetric matrix achieving K^2 = (n-1)/n with T1 = e1.

    T2* = diag(1, -1/(n-1), ..., -1/(n-1)) / ||diag(1, -1/(n-1), ..., -1/(n-1))||_F

    Derivation:
        With T1 = e1, we want to maximize ||T2 e1||^2 / ||T2||^2_F, i.e., the
        squared first-column norm of T2 normalized by its Frobenius norm.
        The maximum is achieved when T2 has the largest possible first eigenvalue
        subject to trace = 0 and ||T2||_F = 1. The optimizer is a diagonal matrix
        with lambda_1 = sqrt((n-1)/n) and lambda_2 = ... = lambda_n = -1/sqrt(n(n-1)).

        Unnormalized: d = diag(1, -1/(n-1), ..., -1/(n-1)).
        Frobenius norm: ||d||^2 = 1 + (n-1)*(1/(n-1))^2 = 1 + 1/(n-1) = n/(n-1).
        So ||d||_F = sqrt(n/(n-1)), and the normalized T2* = d * sqrt((n-1)/n).

    Verification:
        T2* e1 = T2*_{11} e1 = sqrt((n-1)/n) * e1
        ||T2* e1||^2 = (n-1)/n
        ||T2*||^2_F = 1 (by construction)
        K^2 = (n-1)/n ✓

    Args:
        n: spatial dimension (>= 2)
        dtype: JAX dtype

    Returns:
        Shape (n, n), trace-free symmetric, Frobenius norm = 1.
    """
    diag_vals = jnp.array(
        [1.0] + [-1.0 / (n - 1)] * (n - 1), dtype=dtype
    )
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

    Parameterization:
        T1: unconstrained vector in R^n; normalized to unit sphere at each step.
        T2: unconstrained n x n matrix; projected to STF and normalized to
            Frobenius sphere at each step.

    Each restart uses a different random initialization. Returns the maximum
    K^2 found across all restarts.

    Args:
        n: spatial dimension
        num_steps: gradient ascent steps per restart
        key: JAX PRNGKey
        lr: learning rate for gradient ascent
        num_restarts: number of random restarts (run in parallel via vmap)
        dtype: JAX dtype

    Returns:
        dict with keys:
            'best_K2': best K^2 found (scalar)
            'analytic_K2': (n-1)/n (ground truth)
            'all_K2': shape (num_restarts,) array of final K^2 per restart
    """
    grad_fn = jax.grad(kato_ratio_direct, argnums=(0, 1))

    def _normalize_T1(T1):
        return T1 / (jnp.linalg.norm(T1) + 1e-8)

    def _normalize_T2(T2):
        T2_stf = project_tracefree((T2 + T2.T) / 2)
        return T2_stf / (jnp.sqrt(frobenius_sq(T2_stf)) + 1e-8)

    def _one_restart(key_r):
        k1, k2 = jax.random.split(key_r)
        T1 = jax.random.normal(k1, (n,), dtype=dtype)
        T2 = jax.random.normal(k2, (n, n), dtype=dtype)
        T1 = _normalize_T1(T1)
        T2 = _normalize_T2(T2)

        def _step(carry, _):
            T1, T2 = carry
            g1, g2 = grad_fn(T1, T2)
            T1 = _normalize_T1(T1 + lr * g1)
            T2 = _normalize_T2(T2 + lr * g2)
            return (T1, T2), None

        (T1_final, T2_final), _ = jax.lax.scan(_step, (T1, T2), None, length=num_steps)
        return kato_ratio_direct(T1_final, T2_final)

    keys = jax.random.split(key, num_restarts)
    all_K2 = jax.vmap(_one_restart)(keys)

    return {
        'best_K2': jnp.max(all_K2),
        'analytic_K2': kato_analytic(n),
        'all_K2': all_K2,
    }
