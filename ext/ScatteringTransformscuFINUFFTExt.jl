module ScatteringTransformscuFINUFFTExt

"""
    ScatteringTransformscuFINUFFTExt — cuFINUFFT binding for the scattered-planar cascade

The only CUDA-specific piece of the scattered-planar device path. Everything around it is
vendor-agnostic: the cascade is broadcasts and reductions over whatever array type the points are, and
`ScatteredPlanar.build` derives the filter bank and every buffer `similar` to those points. Only the
NUFFT has no device-generic form, so it is bound per vendor here — a ROCm or Metal NUFFT would be an
analogous three-method extension. The KernelAbstractions-native alternative that needs no per-vendor
binding at all is `NonuniformFFTs`.

A `CuArray` point set selects cuFINUFFT through the creation seam (`Plans.nufft_guru_make`); point
assignment and execution dispatch on the returned plan. Loaded by `using CUDA`.

`FINUFFT` is a trigger alongside `CUDA` because both are needed: `cufinufft_plan` is defined in
FINUFFT proper (so it can be named here), while the `cufinufft_*` methods come from FINUFFT's own
CUDA extension, which `using CUDA` is what loads.
"""

using CUDA: CUDA
using FINUFFT: FINUFFT
using ScatteringTransforms: ScatteringTransforms as ST

# `cufinufft_plan` is mutable and owns the device plan, so the finalizer that frees it goes there and
# the scattered-planar wrapper stays immutable — the same arrangement as the host guru plans.
# `nthreads` is accepted and ignored: it configures a CPU library's thread pool, and this plan runs on
# the device, where the launch configuration is cuFINUFFT's own (`gpu_method`, block sizes).
function ST.Plans.nufft_guru_make(::CUDA.CuArray, type::Integer, ms::NTuple{2, Int},
                                  iflag::Integer, ntrans::Integer, eps::Real, ::Type{T};
                                  nthreads::Integer = 0) where {T}
    g = FINUFFT.cufinufft_makeplan(type, collect(ms), iflag, ntrans, eps; dtype = T, modeord = 1)
    finalizer(FINUFFT.cufinufft_destroy!, g)
    return g
end

ST.Plans.nufft_guru_setpts!(g::FINUFFT.cufinufft_plan, x, y) =
    (FINUFFT.cufinufft_setpts!(g, x, y); g)
ST.Plans.nufft_guru_destroy!(g::FINUFFT.cufinufft_plan) = (FINUFFT.cufinufft_destroy!(g); nothing)
ST.Plans.nufft_guru_exec!(g::FINUFFT.cufinufft_plan, input, output) =
    (FINUFFT.cufinufft_exec!(g, input, output); output)

end # module ScatteringTransformscuFINUFFTExt
