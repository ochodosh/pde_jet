# Fischer Decomposition and Symmetric Tensor Algebra

This document explains the mathematical foundations used in `pde_jet`: the Fischer decomposition for symmetric tensors, its use in projecting to trace-free (harmonic) tensors, and the extension to eigenfunction jets satisfying Δu + λu = 0.

---

## 1. Setup

Let R^n be Euclidean space with standard inner product δᵢⱼ (Kronecker delta). A **rank-m symmetric tensor** T on R^n is an array T_{i₁…iₘ} (with indices running over 1,…,n) that is invariant under all permutations of its indices:

$$T_{i_{\sigma(1)} \cdots i_{\sigma(m)}} = T_{i_1 \cdots i_m} \quad \text{for all } \sigma \in S_m.$$

The space of such tensors is denoted Sym^m(R^n). Its dimension is C(n+m-1, m).

The **trace** of T contracts the first two indices:

$$(\operatorname{tr} T)_{i_3 \cdots i_m} = \sum_{i=1}^n T_{i\, i\, i_3 \cdots i_m} = \delta^{ab} T_{ab\, i_3 \cdots i_m}.$$

The trace of a rank-m tensor is a rank-(m-2) tensor.

---

## 2. Jets of PDE Solutions

The **k-jet** of a smooth function u at x=0 is the tuple of derivative tensors

$$\mathbf{T} = (T^{(0)}, T^{(1)}, \ldots, T^{(k)}),$$

where

$$T^{(m)}_{i_1 \cdots i_m} = \partial_{i_1} \cdots \partial_{i_m} u(0).$$

Since partial derivatives commute, each T^(m) is fully symmetric. The Taylor polynomial of u up to order k is

$$p_k(x) = \sum_{m=0}^k \frac{1}{m!}\, T^{(m)}_{i_1 \cdots i_m}\, x^{i_1} \cdots x^{i_m}.$$

The PDE satisfied by u imposes algebraic constraints on the jet:

- **Harmonic** (Δu = 0): tr(T^(m)) = 0 for all m ≥ 2.
- **Helmholtz eigenfunction** (Δu + λu = 0): tr(T^(m)) = −λ T^(m-2) for all m ≥ 2.

In both cases the free data — the "degrees of freedom" in the jet — turns out to be a sequence of **symmetric trace-free (STF) tensors**, one per degree.

---

## 3. Symmetric Trace-Free (STF) Tensors

A symmetric rank-m tensor H is **symmetric trace-free (STF)** if it is fully symmetric and all its traces vanish:

$$\operatorname{tr}(H) = 0 \quad \Longleftrightarrow \quad \sum_{i=1}^n H_{i\, i\, i_3 \cdots i_m} = 0 \text{ for all fixed } i_3, \ldots, i_m.$$

(By full symmetry, if one contracted pair vanishes, so do all contracted pairs.)

The space STF^m(R^n) has dimension

$$\dim \operatorname{STF}^m = \binom{n+m-1}{m} - \binom{n+m-3}{m-2}, \quad m \geq 2,$$

with dim STF^0 = 1, dim STF^1 = n (no trace constraint on scalars or vectors).

**Geometric meaning**: STF^m(R^n) is exactly the space of coefficient tensors of degree-m homogeneous harmonic polynomials on R^n. The Laplacian kills exactly the trace components.

---

## 4. The Fischer Decomposition

**Theorem** (Fischer decomposition for symmetric tensors). Every T ∈ Sym^m(R^n) decomposes *uniquely* as

$$T_{i_1 \cdots i_m} = \sum_{s=0}^{\lfloor m/2 \rfloor} c_s \cdot \operatorname{sym}(\underbrace{\delta \otimes \cdots \otimes \delta}_{s} \otimes H^{(m-2s)})_{i_1 \cdots i_m},$$

where H^(m-2s) ∈ STF^{m-2s}(R^n) and cₛ are normalization constants. Equivalently,

$$\operatorname{Sym}^m(\mathbb{R}^n) = \bigoplus_{s=0}^{\lfloor m/2 \rfloor} \operatorname{sym}(\delta^{\otimes s} \otimes \operatorname{STF}^{m-2s}(\mathbb{R}^n)).$$

The analogy in polynomial language: every homogeneous polynomial of degree m decomposes uniquely as a harmonic polynomial plus |x|² times a harmonic polynomial of degree m−2, plus |x|⁴ times degree m−4, and so on. The Fischer decomposition is the tensor version.

**Consequence**: the STF part H^(m) = P_TF(T) is well-defined (no ambiguity in the decomposition), and the higher-trace parts H^(m-2), H^(m-4), … are proportional to the iterated traces tr(T), tr²(T), … of T.

---

## 5. The STF Projection

The STF projection P_TF: Sym^m → STF^m extracts the top component in the Fischer decomposition. It is given by the formula

$$P_\mathrm{TF}(T)_{i_1 \cdots i_m} = \sum_{s=0}^{\lfloor m/2 \rfloor} b(m,s,n)\, \operatorname{sym}\!\left(\underbrace{\delta \otimes \cdots \otimes \delta}_{s} \otimes \operatorname{tr}^s(T)\right)_{i_1 \cdots i_m},$$

where the coefficients b(m,s,n) are determined by the trace-cancellation condition: we need tr(P_TF(T)) = 0 for all T.

### Deriving the coefficients

Setting tr(P_TF(T)) = 0 gives a triangular recurrence. The solution is

$$b(m, s, n) = \frac{(-1)^s \, \binom{m}{2s} \, (2s-1)!!}{\prod_{j=0}^{s-1} (n + 2m - 4 - 2j)}, \qquad b(m, 0, n) = 1,$$

where (2s−1)!! = 1·3·5·⋯·(2s−1) is the double factorial (with (−1)!! := 1).

**Verified special cases** (derived independently from the trace condition):

| m | s | b(m,s,n) |
|---|---|-----------|
| 2 | 1 | −1/n |
| 3 | 1 | −3/(n+2) |
| 4 | 1 | −6/(n+4) |
| 4 | 2 | 3/((n+4)(n+2)) |
| 5 | 1 | −10/(n+6) |
| 5 | 2 | 15/((n+6)(n+4)) |

**Example (m=2):**

$$P_\mathrm{TF}(T)_{ij} = T_{ij} - \frac{1}{n}\,(\operatorname{tr} T)\,\delta_{ij}.$$

This is the standard traceless part of a matrix: subtract (1/n) times the trace times the identity.

**Example (m=3):**

$$P_\mathrm{TF}(T)_{ijk} = T_{ijk} - \frac{3}{n+2}\, \operatorname{sym}(\delta \otimes \operatorname{tr}(T))_{ijk}.$$

Here tr(T) is a vector and sym(δ ⊗ v)_{ijk} = (δ_{ij}v_k + δ_{ik}v_j + δ_{jk}v_i)/3.

### Key properties of P_TF

1. **Idempotent**: P_TF(P_TF(T)) = P_TF(T).
2. **Trace-free**: tr(P_TF(T)) = 0.
3. **Fixes STF inputs**: if H is already STF, then P_TF(H) = H.
4. **Kills pure-trace tensors**: P_TF(sym(δ^⊗s ⊗ τ)) = 0 for any τ and s ≥ 1.
   (Any tensor in the "lower" Fischer components projects to zero.)
5. **Linearity**: P_TF is a linear projection.

Property 4 is the key fact used in the eigenfunction reconstruction below.

---

## 6. The Inverse Fischer Decomposition

The Fischer decomposition is a direct sum, so it has an explicit inverse. Given T ∈ Sym^m with STF component H = P_TF(T) and traces τₛ = tr^s(T) (s = 1, …, ⌊m/2⌋), we can recover T from H and the traces as:

$$T_{i_1 \cdots i_m} = H_{i_1 \cdots i_m} - \sum_{s=1}^{\lfloor m/2 \rfloor} b(m,s,n)\, \operatorname{sym}(\delta^{\otimes s} \otimes \tau_s)_{i_1 \cdots i_m}.$$

**Proof sketch**: Apply P_TF to both sides. The left side gives P_TF(T) = H. The right side gives P_TF(H) − Σ b(m,s,n) P_TF(sym(δ^s ⊗ τₛ)) = H − 0 = H, using property 4. ✓

Take the trace of both sides. tr(H) = 0, and

$$\operatorname{tr}\!\left(\operatorname{sym}(\delta^{\otimes s} \otimes \tau_s)\right) = c(m,s,n)\,\tau_s + \text{lower-order terms}$$

for an explicitly computable constant c(m,s,n). Setting tr(T) = τ₁ and solving gives back the same coefficients b(m,s,n). ✓

This inverse formula says: **given the STF part and the target traces, T is uniquely determined.**

---

## 7. Harmonic Jets (Δu = 0)

For harmonic functions Δu = 0, applying the Laplacian to the Taylor polynomial and collecting terms of each degree gives

$$\operatorname{tr}(T^{(m)}) = 0 \quad \text{for all } m \geq 2.$$

So T^(m) ∈ STF^m(R^n) for m ≥ 2, and **T^(0), T^(1) are unconstrained**.

The projection in `_harmonics.py` is exactly P_TF applied to each T^(m):

```
make_harmonic_jet: T^(m) ← P_TF(symmetrize(T^(m)))  for m ≥ 2
```

The free data (independent components) of a harmonic k-jet is:

$$\underbrace{1}_{\text{T}^{(0)}} + \underbrace{n}_{\text{T}^{(1)}} + \sum_{m=2}^k \dim\operatorname{STF}^m(\mathbb{R}^n).$$

---

## 8. Eigenfunction Jets (Δu + λu = 0)

For Δu + λu = 0, differentiate the PDE m times at x=0:

$$\partial_{i_1} \cdots \partial_{i_m}(\Delta u + \lambda u)\big|_{x=0} = 0.$$

Since Δ and ∂ᵢ commute, and Δ∂_{i_1}⋯∂_{iₘ}u = ∂_{i_1}⋯∂_{iₘ}Δu, this gives

$$\operatorname{tr}(T^{(m+2)}) + \lambda\, T^{(m)} = 0 \quad \Longrightarrow \quad \operatorname{tr}(T^{(m)}) = -\lambda\, T^{(m-2)}, \quad m \geq 2.$$

The traces are no longer zero; they are determined recursively by the eigenvalue λ and lower-order tensors. Explicitly:

$$\operatorname{tr}^s(T^{(m)}) = (-\lambda)^s\, T^{(m-2s)}, \quad 1 \leq s \leq \lfloor m/2 \rfloor.$$

### Reconstruction from STF parts

The **free data** is still the sequence of STF parts H^(m) = P_TF(T^(m)), since every tensor is uniquely determined by its STF part plus its traces.

Using the inverse Fischer decomposition with τₛ = (-λ)^s T^(m-2s) substituted:

$$\boxed{T^{(m)} = P_\mathrm{TF}(T^{(m)}) - \sum_{s=1}^{\lfloor m/2 \rfloor} b(m,s,n)\, \operatorname{sym}\!\left(\delta^{\otimes s} \otimes (-\lambda)^s T^{(m-2s)}\right)}$$

This is the formula implemented in `_project_eigenfunction_tensors` in `_eigenfunction.py`. Tensors are processed in order m = 0, 1, 2, …, k so that T^(m-2s) on the right-hand side is already available (it was computed at a previous step).

### Verification for small m

**m=2**: Using b(2,1,n) = −1/n and T^(0) = u(0):

$$T^{(2)}_{ij} = P_\mathrm{TF}(T^{(2)})_{ij} - (-\tfrac{1}{n}) \cdot (-\lambda T^{(0)}) \cdot \delta_{ij} = P_\mathrm{TF}(T^{(2)})_{ij} - \frac{\lambda T^{(0)}}{n}\,\delta_{ij}.$$

Check: tr(T^(2)) = 0 + (−λT^(0)/n)·n = −λT^(0). ✓

**m=3**: Using b(3,1,n) = −3/(n+2):

$$T^{(3)}_{ijk} = P_\mathrm{TF}(T^{(3)})_{ijk} + \frac{3}{n+2}\,\operatorname{sym}(\delta \otimes (-\lambda T^{(1)}))_{ijk}.$$

Check: tr(T^(3))_k = 0 + (3/(n+2)) · tr(sym(δ ⊗ (−λT^(1))))_k.
Now tr(sym(δ ⊗ v))_k = (n+2)/3 · v_k (computed by contracting sym(δ⊗v)_{iik} over i).
So tr(T^(3))_k = (3/(n+2)) · (n+2)/3 · (−λT^(1)_k) = −λT^(1)_k. ✓

**m=4** (two-term sum): Using b(4,1,n) = −6/(n+4) and b(4,2,n) = 3/((n+4)(n+2)):

$$T^{(4)} = P_\mathrm{TF}(T^{(4)}) + \frac{6}{n+4}\operatorname{sym}(\delta \otimes (-\lambda T^{(2)})) - \frac{3}{(n+4)(n+2)}\operatorname{sym}(\delta^{\otimes 2} \otimes \lambda^2 T^{(0)}).$$

The trace gives two contributions. The s=1 term produces a main part −λT^(2) plus a spurious isotropic term proportional to δ·tr(T^(2)) = δ·(−λT^(0)). The s=2 term exactly cancels that spurious contribution:

$$\operatorname{tr}(T^{(4)})_{kl} = (-\lambda) T^{(2)}_{kl} + \frac{\lambda^2 T^{(0)}}{n+4}\delta_{kl} - \frac{3}{(n+4)(n+2)} \cdot \frac{\lambda^2 T^{(0)}(n+2)}{3}\,\delta_{kl} = -\lambda T^{(2)}_{kl}. \checkmark$$

This cancellation pattern generalizes to all m: the s-th term in the sum corrects the spurious trace contribution from all s' < s terms.

### λ = 0 recovery

When λ = 0, every correction term vanishes:

$$T^{(m)} = P_\mathrm{TF}(T^{(m)}) + 0 = \text{STF part of } T^{(m)}.$$

The eigenfunction jet reduces to a harmonic jet. ✓

---

## 9. Symmetrized Outer Products and the `sym_outer` Operation

The construction sym(δ^⊗s ⊗ τ) appearing throughout is:

$$\operatorname{sym}(\underbrace{\delta \otimes \cdots \otimes \delta}_{s} \otimes \tau)_{i_1 \cdots i_{m}} = \operatorname{Sym}(\delta_{i_1 i_2} \cdots \delta_{i_{2s-1} i_{2s}} \cdot \tau_{i_{2s+1} \cdots i_m}),$$

where Sym averages over all m! permutations of the index tuple. In code this is `sym_outer(delta_sym(n, s), tau)`.

For s=1: sym(δ ⊗ τ)_{ijk} = (δ_{ij}τ_k + δ_{ik}τ_j + δ_{jk}τ_i)/3 (for rank-3 result).

**Trace formula**: for any τ ∈ Sym^{m-2},

$$\operatorname{tr}\!\left(\operatorname{sym}(\delta \otimes \tau)\right)_{i_3 \cdots i_m} = \frac{n + 2(m-2)}{m(m-1)/2 \cdot \binom{m}{2}^{-1}}\, \tau_{i_3 \cdots i_m} + \text{lower-order trace terms}.$$

The precise constant depends on m and n and is the key quantity used in deriving b(m,1,n) from the trace-cancellation condition.

---

## 10. Summary of the Library Implementation

| Component | File | Mathematical object |
|-----------|------|-------------------|
| `symmetrize(T)` | `_tensor.py` | Sym: Sym^m averaging |
| `trace(T)` | `_tensor.py` | tr: δ^{ab} T_{ab…} |
| `full_trace_k(T, k)` | `_tensor.py` | tr^k: iterate trace k times |
| `delta_sym(n, s)` | `_tensor.py` | sym(δ^⊗s): symmetrized metric |
| `sym_outer(A, B)` | `_tensor.py` | sym(A ⊗ B) |
| `frobenius_sq(T)` | `_tensor.py` | ‖T‖²_F = Σ T²_{i…} (full array) |
| `_stf_coeff(m, s, n)` | `_harmonics.py` | b(m,s,n): Fischer coefficient |
| `project_tracefree(T)` | `_harmonics.py` | P_TF(T): STF projection |
| `HarmonicJet` | `_jet.py` | k-jet with T^(m) ∈ STF^m, m≥2 |
| `EigenfunctionJet` | `_eigenfunction.py` | k-jet with tr(T^m) = −λT^{m-2} |
| `_project_eigenfunction_tensors` | `_eigenfunction.py` | Inverse Fischer with target traces |

---

## 11. References

- **Fischer decomposition** (polynomial version): E.M. Fischer, *Über algebraische Modulsysteme und lineare homogene partielle Differentialgleichungen mit konstanten Koeffizienten*, J. Reine Angew. Math. (1917). The tensor version is a standard result in harmonic analysis on R^n.

- **Harmonic polynomial basis**: Axler, Bourdon, Ramey, *Harmonic Function Theory* (2nd ed., Springer 2001), Chapter 5. The Fischer decomposition appears as the decomposition of homogeneous polynomials into harmonic components.

- **Kato inequality** (sharp constant K² = (n−1)/n for harmonic functions): Kato, T., *Schrödinger operators with singular potentials*, Israel J. Math. (1972); see also Braverman, Milatovic, Shubin (2002) for a clean proof.

- **Higher Kato inequality** (K'² = n/(n+2)): Derived in this library from the eigenvalue analysis of the contraction map L_{T²}: STF³ → R^n.
