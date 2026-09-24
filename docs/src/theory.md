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
  The spatial mean of `S_p x` equals `\bar S_p x`, so the two outputs are consistent and the tests
  enforce that exactly. Under `↓ s` this needs more of `φ_J` than `\hat\phi_J(0)=1` — see
  [The low-pass and decimation](@ref).

Averaging over the domain, or low-pass filtering by `φ_J`, makes the descriptors invariant to
translations; the localized field is also stable to small diffeomorphisms (Mallat 2012).

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

with the constant-`Q` bandwidth `σ_j = ξ_j (1-2^{-1/Q})/(1+2^{-1/Q})/\sqrt{2\ln(1/r)}`, which
puts the crossing of adjacent wavelets at `r` of their peak, `Q` wavelets per octave. The second
term enforces the zero-mean admissibility condition `\hat\psi_j(0)=0`. The filter is analytic (zero
for `ω<0`).

### 2D oriented Morlet

At scale `j` and orientation `θ = πℓ/L` (`ℓ = 0,…,L-1`), with angular wavenumber `k`,
`k_∥ = k·\hat θ` and `k_⊥` the component across it:

```math
\hat\psi(k) = e^{-((k_\parallel-k_0)^2\sigma_\parallel^2 + k_\perp^2\sigma_\perp^2)/2}
            - \beta\, e^{-(k_\parallel^2\sigma_\parallel^2 + k_\perp^2\sigma_\perp^2)/2},
\qquad \beta = e^{-(k_0\sigma_\parallel)^2/2},
```

for `k_∥ ≥ 0` and zero otherwise, with real-space widths `σ_∥ = 0.8·2^j`, `σ_⊥ = σ_∥ L/4` and
`k_0 = 3π/(4·2^j)`. The envelope's angular standard deviation at `|k| = k_0` is `4/(L k_0σ_∥)`, a
fixed fraction `0.68` of the orientation spacing `π/L`.

### 3D oriented Morlet

The same envelope about a unit direction `\hat n`, with `k_∥ = k·\hat n`, over `n_{orient}`
directions spread on the sphere. With their spacing `Δθ = \sqrt{4π/n_{orient}}` in place of `π/L`,
the 2D rule `σ_⊥ = σ_∥ π/(4Δθ)` gives `σ_⊥ = σ_∥\sqrt{π n_{orient}}/8`.

### Littlewood–Paley normalization

For a real field the wavelet layer `Wx = (x ⋆ φ_J,\ x ⋆ ψ_λ)_λ` satisfies

```math
\|Wx\|^2 = \frac1N \sum_k A(k)\,|\hat x(k)|^2,\qquad
A(k) = |\hatφ_J(k)|^2 + \tfrac12 \sum_λ \big(|\hatψ_λ(k)|^2 + |\hatψ_λ(-k)|^2\big),
```

so `(1-α)\|x\|^2 ≤ \|Wx\|^2 ≤ \|x\|^2` when `1-α ≤ A ≤ 1` (Andén & Mallat 2014). Every wavelet is
scaled by one constant `c` that puts the maximum of the wavelet part of `A` at 1 for the bank
continued over every scale. That bank is self-similar, so its `A` is periodic in log-frequency (and
in angle) and `c` depends on the wavelet design alone: `lp_scale_1d(Q)`, `lp_scale_2d(L)`,
`lp_scale_3d(n_orient)`. A wavelet at scale `j` is then the same function in a bank of any depth
`J`, and so is its coefficient.

`littlewood_paley(fb)` returns `A` on the bank's grid. `\hatφ_J(0) = 1` and `\hatψ_λ(0) = 0` give
`A(0) = 1`. `A` is smallest in the band between `φ_J` and the coarsest wavelet, and in 2D/3D at the
corners of the frequency grid, past the finest wavelet.

On an even axis the Nyquist bin holds `±½` at once. A wavelet's value there is the root mean square
of its values at the two aliases, so `A` at that bin equals the continuous sum at `½`.

### The low-pass and decimation

The **coefficients** `\bar S_p x = ⟨U_p x⟩` contain no low-pass: the average is over the whole
domain. The **localized field** `S_p x = U_p x \star φ_J` is defined by its low-pass, the Gaussian
`\hatφ_J(k) = e^{-|k|^2σ^2/2}` with `σ = σ_0 2^J`, `σ_0 = 0.8`
([`ScatteringTransforms.Filters.gaussian_lowpass!`](@ref)), which is also every bank's `averaging`.

Subsampling that field by `s` forces one condition. Since

```math
\langle S_p x\rangle = \tfrac1N \sum_m \big(\hat U_p\,\hat φ_J\big)[m N/s],
```

`⟨S_p x⟩ = \bar S_p x` **iff `\hatφ_J` vanishes on the subsampling lattice** `\{mN/s : m ≠ 0\}`. The
lattice starts at `|k| = 2π/s`, so at the default `s = 2^{J-1}` the largest term is
`e^{-(4πσ_0)^2/2}`, `10^{-22}` at `σ_0 = 0.8`.

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
   pre-modulus layer `W_λ = x ⋆ ψ_λ` with the low-pass `Y = x ⋆ φ_J` is invertible because
   `A > 0`, and the canonical dual frame gives, for a real field,
   ```math
   x = \mathrm{Re}\,\mathcal F^{-1}\Big[\big(\hatφ_J\,\hat Y + \textstyle\sum_λ \hatψ_λ\,\hat W_λ\big)\big/A\Big],
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
zero-mean) and the Riesz multipliers `R_d(k) = -i\,k_d/|k|` (`Σ_d|R_d|²=1` off the DC bin;
`R_d = 0` on axis `d`'s Nyquist bin, the mean of its two aliases):

```math
A_j = \sqrt{\,(x\star\psi_j)^2 + \textstyle\sum_d (x\star R_d\psi_j)^2\,},
```

which also yields a local *phase* and continuous *orientation* (`monogenic_components`). The
band-pass and its Riesz components carry `|\hatψ_j|^2(1 + Σ_d|R_d|^2)` per frequency, so
`A = |\hatφ_J|^2 + Σ_j |\hatψ_j|^2(1 + Σ_d|R_d|^2)` and the wavelets are scaled by
`lp_scale_1d(Q)/2`. On the sphere (`spherical_monogenic_scattering`) the Riesz operator
`R = ð∘(-Δ_S)^{-1/2}` is harmonic-diagonal, and the Riesz energy is `|U^R_j|² = |∇_S g_j|²` with
`g_j = (-Δ_S)^{-1/2}` of the band. With the angular momentum `L = -i\,r×∇`, a real `g` has
`|∇_S g|^2 = |L_+ g|^2 + (∂_φ g)^2`, and `L_+ Y_{ℓm} = \sqrt{(ℓ-m)(ℓ+m+1)}\,Y_{ℓ,m+1}` keeps the
degree, so the energy is three scalar syntheses at the band limit, exact for band-limited `g`. The
in-core direct plan takes the gradient from `∂_θ P̄_ℓ^m` and `∂_φ` evaluated at each point.

## Computation

Convolutions are done in the spectral domain. The core ships a dependency-free **direct-sum
DFT** default; loading `FFTW` selects an `O(N\log N)` fast path automatically
(`spectral = AutoSpectralBackend()`). Batches reuse one plan (`scattering_batch`), and `using OhMyThreads`
enables a multithreaded batched transform (`ThreadedBackend`). The hot path is written with
broadcasts/reductions so it also runs on GPU arrays. The mutation-free `scattering(st, x)` is the
autodiff-friendly counterpart used by synthesis.

## Applications

Texture and field classification, audio timbre, turbulence intermittency, and geophysical
fields — settings where higher-order, non-Gaussian structure beyond the power spectrum is
informative.

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
