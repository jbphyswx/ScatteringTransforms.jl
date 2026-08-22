# Theory of Scattering Transforms

The wavelet scattering transform builds translation-invariant, deformation-stable descriptors of
signals, images, and volumes by cascading wavelet convolutions with a pointwise modulus. This
page describes exactly what `ScatteringTransforms.jl` computes.

## Two outputs: averaged coefficients and the localized field

For a signal `x`, define the propagated field along a path `p = (λ₁, λ₂, …)` of wavelet indices:

```math
U_0 x = x,\qquad U_{p}\,x = \big|\,U_{p'} x \star \psi_{\lambda_k}\,\big|
```

where `p'` drops the last index `λ_k`. The package exposes two reductions of `U_p x`:

- **Averaged coefficients** (`st(x)`, the default) — the spatial mean of each propagated field,
  ```math
  \bar S_p x = \big\langle U_p x \big\rangle .
  ```
  These are the scalar "scattering coefficients" used as texture/field statistics
  (Cheng & Ménard 2021). Concretely `S0 = ⟨x⟩`, `S1[λ] = ⟨|x ⋆ ψ_λ|⟩`, and
  `S2[λ₁,λ₂] = ⟨||x ⋆ ψ_{λ₁}| ⋆ ψ_{λ₂}|⟩`.

- **Localized field** (`scattering_field(st, x)`) — Mallat's translation-*covariant* field,
  low-pass filtered by the scaling function `φ_J` and subsampled,
  ```math
  S_p x = \big(\,U_p x \star \phi_J\,\big)\!\downarrow s .
  ```
  Because `\hat\phi_J(0)=1`, the spatial mean of `S_p x` equals `\bar S_p x` — the two outputs
  are consistent, and the package tests enforce this exactly.

The averaging (or `φ_J` low-pass) is what makes the descriptors invariant to translations; the
localized field additionally inherits Mallat's stability to small diffeomorphisms.

## Admissible paths

Coefficients are only computed along paths whose **effective scale is strictly increasing**,
`j_eff(λ_{k+1}) > j_eff(λ_k)` — each successive wavelet is strictly coarser (lower frequency).
For 2D/3D this means the *scale* strictly increases while **all orientation pairs are allowed**;
same-scale pairs are excluded. The admissible paths are enumerated once into a `ScatteringTree`.

## Wavelets

### 1D Morlet

In the Fourier domain, normalized frequency `ω ∈ [0, ½]`,

```math
\hat\psi_j(\omega) = e^{-(\omega-\xi_j)^2/2\sigma_j^2} - \kappa_j\, e^{-\omega^2/2\sigma_j^2},
\qquad \xi_j = \tfrac12 \, 2^{-j/Q},
```

with the constant-`Q` bandwidth `σ_j = ξ_j (1-2^{-1/Q})/(1+2^{-1/Q})/\sqrt{2\ln(1/r)}`
(Lostanlen/Kymatio), `Q` wavelets per octave. The second term enforces the zero-mean
admissibility condition `\hat\psi_j(0)=0`. The filter is analytic (zero for `ω<0`).

### 2D oriented Morlet

Oriented Morlet wavelets at scale `j` and orientation `θ = πℓ/L` (`ℓ = 0,…,L-1`), elliptical in
the Fourier plane and analytic on the half-plane `k·\hat θ ≥ 0`. The low-pass `φ_J` is a Gaussian
matched to the coarsest scale.

## Reduced descriptors

For analysis it is common to reduce the raw coefficients (`Reductions` module, and
`compute_shape_sparsity` for 2D):

- **normalized** `s1 = S1/S0`, `s2 = S2/S1` — remove dependence on overall amplitude;
- **log** `log S1`, `log S2` — gaussianize heavy-tailed coefficients of intermittent fields;
- **sparsity** `s₂₁ = ⟨S₂/S₁⟩` over orientations — energy cascade from `j₁` to coarser `j₂`;
- **shape / anisotropy** `s₂₂ = ⟨S₂\cos 2Δθ⟩/⟨S₂⟩` — the second angular harmonic, `≈ 0` for
  isotropic fields and nonzero for oriented structure.

## Reconstruction

There is **no exact analytic inverse** of the scattering transform — the modulus discards the
local phase of each wavelet coefficient. Three reconstruction levels are available:

1. **Exact linear wavelet-frame inverse** (`wavelet_transform` / `iwavelet`). The *complex*,
   pre-modulus layer `x ⋆ ψ_λ` plus the low-pass `x ⋆ φ` is exactly invertible: because the bank
   is a tight frame (`Σ_λ|\hatψ_λ|² + |\hatφ|² ≡ 1`), the dual frame is itself and
   ```math
   x = \sum_\lambda (x\star\psi_\lambda)\star\psi_\lambda^\ast + (x\star\phi)\star\phi^\ast ,
   ```
   recovered to machine precision (1D/2D/3D).
2. **Phase retrieval** (`reconstruct_phase`) from the first-order moduli `|x ⋆ ψ_λ|` alone, via
   Gerchberg–Saxton alternating projections (reconstruct with the exact inverse, re-impose the
   target magnitudes, repeat); determined up to a global sign (Waldspurger & Mallat 2015).
3. **Gradient-descent synthesis** (`synthesize`, in the DifferentiationInterface extension) from
   the scattering coefficients themselves: from noise, minimize `‖S(\hat x) − S(x)‖²`
   (Bruna & Mallat microcanonical models). This yields a new *sample* with matching multiscale
   statistics — not the original field — and is differentiated through the mutation-free
   `scattering(st, x)` by any `ADTypes` backend (with Enzyme:
   `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`).

## Monogenic (Riesz) scattering

`MonogenicScattering` replaces the oriented analytic modulus with the rotation-covariant
**monogenic amplitude**. From an *isotropic* band-pass `ψ_j` (radial in frequency, real,
zero-mean) and the Riesz multipliers `R_d(k) = -i\,k_d/|k|` (`Σ_d|R_d|²=1` off-DC):

```math
A_j = \sqrt{\,(x\star\psi_j)^2 + \textstyle\sum_d (x\star R_d\psi_j)^2\,},
```

which also yields a local *phase* and continuous *orientation* (`monogenic_components`), recovered
without quantizing into discrete orientation bins. On the sphere (`spherical_monogenic_scattering`,
NUFSHT extension) the Riesz operator `R = ð∘(-Δ_S)^{-1/2}` is harmonic-diagonal; the Riesz energy
`|U^R_j|² = |∇_S g_j|²` (with `g_j=(-Δ_S)^{-1/2}` of the band) is evaluated with **spin-0**
transforms only, via the identity `|∇_S g|² = ½Δ_S(g²) − g\,Δ_S g`.

## Computation

Convolutions are done in the spectral domain. The core ships a dependency-free **direct-sum
DFT** default; loading `FFTW` selects an `O(N\log N)` fast path automatically
(`spectral = AutoSpectralBackend()`). Batches reuse one plan (`scattering_batch`), and `using OhMyThreads`
enables a multithreaded batched transform (`ThreadedBackend`). The hot path is written with
broadcasts/reductions so it also runs on GPU arrays. The mutation-free `scattering(st, x)` is the
autodiff-friendly counterpart used by synthesis.

## Applications

Texture and field classification, audio timbre, turbulence intermittency, and submesoscale
oceanography (sea-surface-height variability) — settings where higher-order, non-Gaussian
structure beyond the power spectrum is informative.

## References

- Mallat, S. (2012). Group invariant scattering. *Comm. Pure Appl. Math.*, 65(10), 1331–1398.
- Bruna, J., & Mallat, S. (2013). Invariant scattering convolution networks. *IEEE PAMI*,
  35(8), 1872–1886.
- Andén, J., & Mallat, S. (2014). Deep scattering spectrum. *IEEE Trans. Signal Process.*
- Allys, E. et al. (2019). The RWST, a comprehensive statistical description of the non-Gaussian
  structures in the ISM. *A&A*.
- Cheng, T. Y., & Ménard, B. (2021). How to quantify fields or textures? A guide to the
  scattering transform. [arXiv:2112.01288](https://arxiv.org/pdf/2112.01288).
- Waldspurger, I., & Mallat, S. (2015). Phase retrieval for the Cauchy wavelet transform / wavelet
  transform modulus.
- Bruna, J., & Mallat, S. (2018). Multiscale sparse microcanonical models.
  [arXiv:1801.02013](https://arxiv.org/abs/1801.02013).
- Felsberg, M., & Sommer, G. (2001). The monogenic signal. *IEEE Trans. Signal Process.*, 49(12).
- Unser, M., Sage, D., & Van De Ville, D. (2009). Multiresolution monogenic signal analysis using
  the Riesz–Laplace wavelet transform. *IEEE Trans. Image Process.*, 18(11).
