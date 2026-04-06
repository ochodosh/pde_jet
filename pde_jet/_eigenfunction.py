"""
EigenfunctionJet: k-jet of a solution to Δu + λu = 0 on R^n.

Mathematical background:
    For Δu + λu = 0, differentiating the PDE m times and evaluating at x=0:

        tr(T^(m+2)) = −λ T^(m)    for all m ≥ 0

    where tr contracts the first two indices.  So the trace of each derivative
    tensor is determined by lower-order tensors — unlike the harmonic case where
    all traces vanish.

    The "free data" for the jet is the sequence of STF parts P_TF(T^(m)).
    Given those, the full T^(m) is recovered by the inverse Fischer decomposition:

        T^(m)_new = P_TF(T^(m)) − Σ_{s=1}^{⌊m/2⌋} b(m,s,n) sym_outer(δ^⊗s, (−λ)^s T^(m−2s)_new)

    where b(m,s,n) are the STF projection coefficients (_stf_coeff from _harmonics.py)
    and tensors are processed in order m=0,1,...,k so lower-order T^(m−2s)_new
    are available at each step.

    For λ=0 all correction terms vanish → T^(m)_new = P_TF(T^(m)) → identical to
    HarmonicJet.

    Verification of the formula for small m (derived from the trace condition):
        m=2: T^2_new = P_TF(T^2) − (λT^0/n) δ,  tr(T^2_new) = −λ T^0 ✓
        m=3: T^3_new = P_TF(T^3) + (3/(n+2)) sym(δ ⊗ (−λT^1)),  tr = −λ T^1 ✓
        m=4: (two-term sum)  tr(T^4_new) = −λ T^2_new ✓  (the s=2 term cancels the
             spurious δ tr(T^2_new) contribution from the s=1 term)

Storage convention:
    EigenfunctionJet.tensors[m] is the FULL symmetric tensor T^(m) satisfying
    tr(T^(m)) = −λ T^(m−2), NOT the trace-free part.  This is consistent with
    how _operators.py consumes jet tensors (evaluate_polynomial, gradient_at, etc.).
"""

import jax
import jax.numpy as jnp

from ._harmonics import _stf_coeff, project_tracefree
from ._tensor import delta_sym, sym_outer, symmetrize


class EigenfunctionJet:
    """A k-jet of a solution to Δu + λu = 0 at the origin in R^n.

    Attributes:
        tensors: tuple of length k+1 where tensors[m] has shape (n,)*m.
                 For m=0,1: unconstrained.
                 For m≥2: fully symmetric with tr(T^m) = −lam * T^{m−2}.
        n:   spatial dimension (Python int, static aux data)
        k:   jet order (Python int, static aux data)
        lam: eigenvalue λ in Δu + λu = 0 (Python float, static aux data)

    When lam=0 the tensors coincide with those of a HarmonicJet (trace-free).
    """

    def __init__(self, tensors: tuple, n: int, k: int, lam: float):
        self.tensors = tensors
        self.n = n
        self.k = k
        self.lam = lam

    def __repr__(self) -> str:
        return f"EigenfunctionJet(n={self.n}, k={self.k}, lam={self.lam})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EigenfunctionJet):
            return False
        if self.n != other.n or self.k != other.k or self.lam != other.lam:
            return False
        return all(
            jnp.allclose(a, b) for a, b in zip(self.tensors, other.tensors)
        )


def _jet_flatten(jet: EigenfunctionJet):
    """Flatten EigenfunctionJet for JAX pytree: children are the tensor arrays."""
    children = jet.tensors
    aux = (jet.n, jet.k, jet.lam)
    return children, aux


def _jet_unflatten(aux, children):
    """Reconstruct EigenfunctionJet from pytree children and aux data."""
    n, k, lam = aux
    return EigenfunctionJet(tuple(children), n, k, lam)


jax.tree_util.register_pytree_node(EigenfunctionJet, _jet_flatten, _jet_unflatten)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _project_eigenfunction_tensors(
    tensors: tuple, n: int, k: int, lam: float
) -> tuple:
    """Project arbitrary symmetric tensors to satisfy tr(T^m) = -lam * T^{m-2}.

    Processes tensors in order m=0,1,...,k.  For each m≥2:
        1. Extract the STF part: H = P_TF(symmetrize(T^m)).
        2. Add back the trace-carrying part determined by lower-order tensors:
           T^m_new = H − Σ_{s=1}^{⌊m/2⌋} b(m,s,n) sym_outer(δ^⊗s, (−λ)^s T^{m−2s}_new)

    Tensors of degree 0 and 1 are left unchanged.

    This is a Python-level loop over m and s (all static integers); every array
    operation inside is a JAX primitive and is jit/vmap compatible.

    Args:
        tensors: tuple of length k+1, tensors[m] has shape (n,)*m
        n: spatial dimension
        k: jet order
        lam: eigenvalue

    Returns:
        tuple of length k+1 with the eigenfunction constraint enforced.
    """
    new: list = [tensors[0], tensors[1]]  # T^0, T^1 unconstrained
    for m in range(2, k + 1):
        H = project_tracefree(symmetrize(tensors[m]))
        for s in range(1, m // 2 + 1):
            b = _stf_coeff(m, s, n)
            lower = new[m - 2 * s]           # T^{m-2s}_new, already projected
            # (-lam)^s * lower is a JAX scalar/array; sym_outer handles ranks correctly
            H = H - b * sym_outer(delta_sym(n, s), ((-lam) ** s) * lower)
        new.append(H)
    return tuple(new)


def _reproject_eigenfunction(j: EigenfunctionJet) -> EigenfunctionJet:
    """Re-project all tensors of degree m≥2 to satisfy the eigenfunction constraint.

    Called after each gradient step in optimize_ratio to restore the PDE
    constraint after the unconstrained gradient update.
    """
    new_tensors = _project_eigenfunction_tensors(j.tensors, j.n, j.k, j.lam)
    return EigenfunctionJet(new_tensors, j.n, j.k, j.lam)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def make_eigenfunction_jet(
    tensors: tuple, n: int, k: int, lam: float
) -> EigenfunctionJet:
    """Construct an EigenfunctionJet, enforcing tr(T^m) = -lam * T^{m-2}.

    For m=0,1: tensors are stored as-is.
    For m≥2: symmetrize then apply eigenfunction projection.

    Args:
        tensors: tuple of arrays where tensors[m] has shape (n,)*m.
        n: spatial dimension.
        k: jet order; must equal len(tensors) − 1.
        lam: eigenvalue in Δu + lam*u = 0.

    Returns:
        EigenfunctionJet with the trace constraint enforced.
    """
    assert len(tensors) == k + 1, (
        f"Expected {k+1} tensors for a {k}-jet, got {len(tensors)}"
    )
    for m, T in enumerate(tensors):
        expected = (n,) * m
        assert T.shape == expected, (
            f"tensors[{m}] has shape {T.shape}, expected {expected}"
        )
    projected = _project_eigenfunction_tensors(tensors, n, k, lam)
    return EigenfunctionJet(projected, n, k, lam)


def zero_eigenfunction_jet(
    n: int, k: int, lam: float, dtype=jnp.float32
) -> EigenfunctionJet:
    """The zero k-jet for Δu + λu = 0 in R^n: all tensors are zero.

    The zero jet trivially satisfies tr(0) = 0 = −λ·0 for any λ.

    Args:
        n: spatial dimension
        k: jet order
        lam: eigenvalue
        dtype: JAX dtype

    Returns:
        EigenfunctionJet with all-zero tensors.
    """
    tensors = tuple(jnp.zeros((n,) * m, dtype=dtype) for m in range(k + 1))
    return EigenfunctionJet(tensors, n, k, lam)


def random_eigenfunction_jet(
    key: jax.Array, n: int, k: int, lam: float, dtype=jnp.float32
) -> EigenfunctionJet:
    """Sample a random jet satisfying the eigenfunction constraint Δu + λu = 0.

    For each degree m:
    - m=0,1: sample from a standard normal (unconstrained).
    - m≥2: sample a random symmetric tensor and project to enforce
      tr(T^m) = −λ T^{m−2}.

    The distribution is the pushforward of the standard normal through the
    eigenfunction projection — suitable for random initialization.

    Args:
        key: JAX PRNGKey
        n: spatial dimension
        k: jet order
        lam: eigenvalue
        dtype: JAX dtype

    Returns:
        EigenfunctionJet with the trace constraint enforced.
    """
    raw = []
    for m in range(k + 1):
        key, subkey = jax.random.split(key)
        T = jax.random.normal(subkey, shape=(n,) * m, dtype=dtype)
        raw.append(T)
    projected = _project_eigenfunction_tensors(tuple(raw), n, k, lam)
    return EigenfunctionJet(projected, n, k, lam)
