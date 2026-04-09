# PDE Jet Library — Development Guidelines

## What This Is

A JAX library for storing and manipulating **jets** of PDE solutions — that is, the collection of partial derivatives of a solution up to order k at a point. Primary use case: machine learning mathematical experiments, e.g., testing sharp constants in inequalities for harmonic functions (Kato inequality, etc.).

## Core Principles

### 1. Correctness Is Paramount

This is a mathematical library. **99% correct is failure.** Every implementation must be derived from first principles:

- Before writing any function, write down the mathematical definition it encodes.
- Think through edge cases mathematically, not just computationally (e.g., what happens at order 0, in dimension 1, with a flat jet?).
- When in doubt, derive it. Do not guess at formulas or copy them without verification.
- Tests are useful but **logical code review is essential**. A passing test on wrong mathematics is worse than a failing test — it provides false confidence. Always verify that the test itself is mathematically sound before trusting it.
- Numerical precision matters. Be explicit about float types and understand when operations may lose precision.

### 2. Simplicity and Generalizability

- Prefer clean mathematical abstractions over clever code.
- Data structures should reflect the underlying math directly — someone who knows the math should be able to read the code.
- Avoid premature specialization. A function for harmonic functions should, where possible, be a special case of a function for general PDE solutions.
- No unnecessary abstraction layers. If it can be a function, it should be a function.

### 3. GPU-First, JAX Throughout

- **Everything must be vectorizable.** Design all operations to work on batches from the start. There are no scalar-only implementations.
- Prefer **mathematical simplifications over runtime computation**. If an index sum can be evaluated in closed form, derive the closed form. Do not rely on JAX to optimize away avoidable work.
- Use `jax.numpy` exclusively (no `numpy` in hot paths). Keep operations `jit`-compatible by default — no Python control flow over traced values.
- Avoid in-place mutation. All data structures are immutable (JAX pytrees or named tuples).
- When writing a new operation, ask: can this be expressed as a sequence of `jnp` primitives that `vmap`/`jit` can handle without retracing?

## Jupyter Notebooks

Notebooks in `notebooks/` are Colab-ready experiment scripts. When creating or editing them:

- **Verify execution order before finishing.** Mentally step through every cell 0→N and confirm each cell only references names defined in earlier cells. Never leave a cell that uses a variable defined in a later cell.
- **Natural narrative order:** imports → definitions → training/computation → plots → inspection. Do not put inspect/plot cells before the training loop that produces the data they display.
- **After any NotebookEdit that reorders or splits cells**, re-read the full cell list (e.g. via Bash + `python3 -c "import json; ..."`) and verify the order is correct before reporting done.
- Notebooks are gitignored; keep them Colab-compatible (pip install cell at top, no local paths).

## Conventions

- **Dimension** `n`: the ambient space dimension (e.g., `n=3` for 3D harmonic functions).
- **Order** `k`: the jet order (derivatives up to and including order `k`).
- Multi-index notation follows standard PDE convention: a multi-index `α` with `|α| = m` denotes an `m`-th order partial derivative.
- All mathematical symbols used in code should match the notation in the docstring or a referenced source.
