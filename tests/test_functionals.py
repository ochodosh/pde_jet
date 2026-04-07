"""
Tests for gradient_of_scalar_functional and laplacian_of_scalar_functional.

Each test encodes a mathematical property verified from first principles.
"""

import jax
import jax.numpy as jnp
import pytest

from pde_jet import random_harmonic_jet
from pde_jet._functionals import (
    gradient_of_scalar_functional,
    laplacian_of_scalar_functional,
)


# ---------------------------------------------------------------------------
# gradient_of_scalar_functional — 3-arg and 2-arg W_fn
# ---------------------------------------------------------------------------


def test_gradient_3arg_identity():
    """f(u, s, q) = s  →  ∇W = 2 H@g (formula: f_u=0, f_s=1, f_q=0)."""
    key = jax.random.PRNGKey(0)
    j = random_harmonic_jet(key, n=3, k=2)
    g = j.tensors[1]
    H = j.tensors[2]

    grad = gradient_of_scalar_functional(lambda u, s, q: s, j)
    expected = 2.0 * (H @ g)
    assert jnp.allclose(grad, expected, atol=1e-5)


def test_gradient_2arg_same_as_3arg_wrapper():
    """f(u, s) should give the same result as lambda u, s, q: f(u, s)."""
    key = jax.random.PRNGKey(1)
    j = random_harmonic_jet(key, n=3, k=2)

    f2 = lambda u, s: s / (u ** 2 + 1.0)
    f3 = lambda u, s, q: s / (u ** 2 + 1.0)

    grad2 = gradient_of_scalar_functional(f2, j)
    grad3 = gradient_of_scalar_functional(f3, j)
    assert jnp.allclose(grad2, grad3, atol=1e-6)


def test_gradient_2arg_constant_f():
    """f(u, s) = 1 (constant)  →  ∇W = 0 everywhere."""
    key = jax.random.PRNGKey(2)
    j = random_harmonic_jet(key, n=3, k=2)

    grad = gradient_of_scalar_functional(lambda u, s: jnp.ones(()), j)
    assert jnp.allclose(grad, jnp.zeros(3), atol=1e-6)


def test_gradient_2arg_linear_u():
    """f(u, s) = u  →  ∇W = g  (formula: f_u=1, f_s=0)."""
    key = jax.random.PRNGKey(3)
    j = random_harmonic_jet(key, n=3, k=2)
    g = j.tensors[1]

    grad = gradient_of_scalar_functional(lambda u, s: u, j)
    assert jnp.allclose(grad, g, atol=1e-5)


# ---------------------------------------------------------------------------
# laplacian_of_scalar_functional — 3-arg and 2-arg W_fn
# ---------------------------------------------------------------------------


def test_laplacian_2arg_same_as_3arg_wrapper():
    """f(u, s) should give the same result as lambda u, s, q: f(u, s)."""
    key = jax.random.PRNGKey(4)
    j = random_harmonic_jet(key, n=3, k=2)

    f2 = lambda u, s: s * u
    f3 = lambda u, s, q: s * u

    lap2 = laplacian_of_scalar_functional(f2, j)
    lap3 = laplacian_of_scalar_functional(f3, j)
    assert jnp.allclose(lap2, lap3, atol=1e-6)


def test_laplacian_2arg_constant():
    """f(u, s) = 1  →  ΔW = 0."""
    key = jax.random.PRNGKey(5)
    j = random_harmonic_jet(key, n=3, k=2)

    lap = laplacian_of_scalar_functional(lambda u, s: jnp.ones(()), j)
    assert jnp.allclose(lap, 0.0, atol=1e-6)


def test_laplacian_2arg_linear_s():
    """f(u, s) = s  →  ΔW = 2 ||H||²_F  (formula: f_s=1, f_uu=f_us=f_ss=0)."""
    key = jax.random.PRNGKey(6)
    j = random_harmonic_jet(key, n=3, k=2)
    H = j.tensors[2]

    from pde_jet._tensor import frobenius_sq
    lap = laplacian_of_scalar_functional(lambda u, s: s, j)
    expected = 2.0 * frobenius_sq(H)
    assert jnp.allclose(lap, expected, atol=1e-5)


def test_laplacian_2arg_kato_ratio():
    """f(u, s) = s / u^2 (Kato-like).

    Verify that calling with 2 args gives the same scalar as the 3-arg form.
    """
    key = jax.random.PRNGKey(7)
    # Need u > 0; fix T0 to a positive value
    from pde_jet import fix_u, replace_tensor
    j = random_harmonic_jet(key, n=3, k=2)
    j = replace_tensor(j, 0, jnp.array(1.0))

    f2 = lambda u, s: s / u ** 2
    f3 = lambda u, s, q: s / u ** 2

    lap2 = laplacian_of_scalar_functional(f2, j)
    lap3 = laplacian_of_scalar_functional(f3, j)
    assert jnp.allclose(lap2, lap3, atol=1e-5)
