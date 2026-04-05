"""
HarmonicJet: the k-jet of a harmonic function at the origin in R^n.

A k-jet is represented as a tuple of tensors (T^(0), T^(1), ..., T^(k)) where
    T^(m) has shape (n,)*m
    T^(m)_{i1...im} = d_{i1}...d_{im} u(0)  (m-th order partial derivative tensor)

For m = 0: scalar (function value), no harmonicity constraint.
For m = 1: vector (gradient), no harmonicity constraint (any gradient is realizable
           by Cauchy-Kowalewski).
For m >= 2: T^(m) is fully symmetric and trace-free (STF), which is the
            infinitely-iterated consequence of Delta u = 0.

JAX pytree registration:
    The tensors tuple is the "children" (dynamic/batchable data).
    n and k are "auxiliary data" (static shape metadata, NOT leaves).
    This allows jit/vmap/grad to work on batches of jets while n and k
    remain fixed compile-time constants. Attempting to vmap over jets with
    different (n, k) will raise an error, which is the correct behavior.
"""

import jax
import jax.numpy as jnp

from ._harmonics import project_tracefree
from ._tensor import symmetrize


class HarmonicJet:
    """A k-jet of a harmonic function u at the origin in R^n.

    Attributes:
        tensors: tuple of length k+1 where tensors[m] has shape (n,)*m.
                 For m >= 2, tensors[m] is fully symmetric and trace-free.
        n: spatial dimension (Python int, static)
        k: jet order (Python int, static)
    """

    def __init__(self, tensors: tuple, n: int, k: int):
        self.tensors = tensors
        self.n = n
        self.k = k

    def __repr__(self) -> str:
        return f"HarmonicJet(n={self.n}, k={self.k})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, HarmonicJet):
            return False
        if self.n != other.n or self.k != other.k:
            return False
        return all(
            jnp.allclose(a, b) for a, b in zip(self.tensors, other.tensors)
        )


def _jet_flatten(jet: HarmonicJet):
    """Flatten HarmonicJet for JAX pytree: children are the tensor arrays."""
    children = jet.tensors  # tuple of arrays — JAX traverses tuples automatically
    aux = (jet.n, jet.k)
    return children, aux


def _jet_unflatten(aux, children):
    """Reconstruct HarmonicJet from pytree children and aux data."""
    n, k = aux
    return HarmonicJet(tuple(children), n, k)


jax.tree_util.register_pytree_node(HarmonicJet, _jet_flatten, _jet_unflatten)


def make_harmonic_jet(tensors: tuple, n: int, k: int) -> HarmonicJet:
    """Construct a HarmonicJet, enforcing the harmonic (STF) constraint.

    For m = 0, 1: tensors are stored as-is (no harmonicity constraint).
    For m >= 2: applies symmetrize then project_tracefree.

    Args:
        tensors: tuple of arrays where tensors[m] has shape (n,)*m.
                 tensors[0] may be a scalar (shape ()).
        n: spatial dimension
        k: jet order; must equal len(tensors) - 1

    Returns:
        HarmonicJet with STF tensors for m >= 2.
    """
    assert len(tensors) == k + 1, (
        f"Expected {k+1} tensors for a {k}-jet, got {len(tensors)}"
    )
    projected = []
    for m, T in enumerate(tensors):
        expected_shape = (n,) * m
        assert T.shape == expected_shape, (
            f"tensors[{m}] has shape {T.shape}, expected {expected_shape}"
        )
        if m >= 2:
            T = project_tracefree(symmetrize(T))
        projected.append(T)
    return HarmonicJet(tuple(projected), n, k)


def zero_jet(n: int, k: int, dtype=jnp.float32) -> HarmonicJet:
    """The zero k-jet in R^n: all derivative tensors are zero.

    Args:
        n: spatial dimension
        k: jet order
        dtype: JAX dtype

    Returns:
        HarmonicJet with all-zero tensors.
    """
    tensors = tuple(jnp.zeros((n,) * m, dtype=dtype) for m in range(k + 1))
    return HarmonicJet(tensors, n, k)


def random_harmonic_jet(
    key: jax.Array, n: int, k: int, dtype=jnp.float32
) -> HarmonicJet:
    """Sample a random harmonic k-jet in R^n.

    For each degree m:
    - Sample a random (n,)*m tensor from a standard normal distribution.
    - Symmetrize and project to the STF subspace.

    The resulting distribution is the pushforward of the standard normal on
    R^{n^m} through the STF projection (not uniform on the STF sphere, but
    suitable for random initialization and testing).

    Args:
        key: JAX PRNGKey
        n: spatial dimension
        k: jet order
        dtype: JAX dtype

    Returns:
        HarmonicJet with random STF tensors for m >= 2.
    """
    tensors = []
    for m in range(k + 1):
        key, subkey = jax.random.split(key)
        T = jax.random.normal(subkey, shape=(n,) * m, dtype=dtype)
        if m >= 2:
            T = project_tracefree(symmetrize(T))
        tensors.append(T)
    return HarmonicJet(tuple(tensors), n, k)
