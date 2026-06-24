# Examples

Working examples for `ScatteringTransforms.jl`.

```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
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

All three examples double as smoke tests (they `@test` their own invariants).
