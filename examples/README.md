# Examples

Working examples for `ScatteringTransforms.jl`.

```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # local package wired via [sources]
julia --project=. basic_usage.jl        # use `-t4` to exercise threading
```

## `basic_usage.jl` — guided tour
1D averaged coefficients; the localized (Mallat) field and its mean↔averaged consistency;
reduced descriptors (`normalized_coefficients`); batched and threaded transforms; 2D
sparsity/anisotropy; 3D volumetric scattering; Float32 end-to-end.

## `zero_allocation_streaming.jl` — streaming many signals
Naive per-call allocation vs a pre-allocated buffer reused via `scattering_transform!`
(steady-state **zero** allocation with a mutable S0 container), plus `scattering_batch` over a
stack. The pattern for TB-scale datasets.

## `backends.jl` — spectral & compute backends
In-core direct-sum vs FFTW fast path (identical results, ~100–1000× faster); serial vs
`ThreadedBackend` batched transforms; GPU is exercised if a CUDA device is available and skipped
gracefully otherwise.

## `synthesis_and_inverse.jl` — reconstruction
The three reconstruction levels: the exact linear wavelet-frame inverse (`iwavelet ∘
wavelet_transform`, machine precision); phase retrieval from first-order moduli
(`reconstruct_phase`, Gerchberg–Saxton); and gradient-descent synthesis from the scattering
coefficients (`synthesize`, via DifferentiationInterface + Mooncake), which produces a new
sample with matching multiscale statistics.

## `monogenic.jl` — monogenic (Riesz) scattering
`MonogenicScattering` in 1D/2D/3D, the Riesz-multiplier partition of unity, and
`monogenic_components` — the smooth amplitude envelope and the continuously-recovered local
orientation/phase (vs the oriented-modulus transform).

All five examples double as smoke tests (they `@test` their own invariants).
