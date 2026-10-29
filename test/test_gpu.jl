using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArrays
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms as ST

# JLArrays has no FFT; these plans copy whole arrays through FFTW.
struct JLFFT{T,P<:AbstractFFTs.Plan{T}} <: AbstractFFTs.Plan{T}
    inner::P
    inplace::Bool
end
JLFFT(p::AbstractFFTs.Plan{T}, inplace) where {T} = JLFFT{T,typeof(p)}(p, inplace)

AbstractFFTs.plan_fft(x::JLArrays.JLArray, region = 1:ndims(x); kwargs...) =
    JLFFT(FFTW.plan_fft(Array(x), region; kwargs...), false)
AbstractFFTs.plan_fft!(x::JLArrays.JLArray, region = 1:ndims(x); kwargs...) =
    JLFFT(FFTW.plan_fft!(Array(x), region; kwargs...), true)
AbstractFFTs.plan_ifft(x::JLArrays.JLArray, region = 1:ndims(x); kwargs...) =
    JLFFT(FFTW.plan_ifft(Array(x), region; kwargs...), false)
AbstractFFTs.plan_ifft!(x::JLArrays.JLArray, region = 1:ndims(x); kwargs...) =
    JLFFT(FFTW.plan_ifft!(Array(x), region; kwargs...), true)
AbstractFFTs.plan_bfft!(x::JLArrays.JLArray, region = 1:ndims(x); kwargs...) =
    JLFFT(FFTW.plan_bfft!(Array(x), region; kwargs...), true)

LinearAlgebra.mul!(out::JLArrays.JLArray, p::JLFFT, x::JLArrays.JLArray) =
    copyto!(out, p.inner * Array(x))
Base.:*(p::JLFFT, x::JLArrays.JLArray) =
    p.inplace ? LinearAlgebra.mul!(x, p, x) : JLArrays.JLArray(p.inner * Array(x))
Base.size(p::JLFFT) = size(p.inner)
KA.synchronize(::JLArrays.JLBackend) = nothing

_gpu_flat(c::ST.Coefficients.ScatteringCoefficients1D) = ST.Coefficients.flatten1d(c)
_gpu_flat(c::ST.Coefficients.ScatteringCoefficients2D) = ST.Coefficients.flatten2d(c)

Test.@testset "Device transforms with scalar indexing disabled (JLArray)" begin
    task_local_storage(:ScalarIndexing, GPUArraysCore.ScalarDisallowed) do
        gpu = ComputationalBackends.GPUBackend(JLArrays.JLBackend())
        FB = SpectralBackends.FFTSpectralBackend()
        stored, computed = ST.FilterBanks.StoredFilters(), ST.FilterBanks.ComputedFilters()

        Test.@testset "gridded and monogenic coefficients, batches and localized fields" begin
            cases = (
                (ST.ScatteringCore.ScatteringTransform{1}, 64, (64,), 3, (Q = 1,), computed),
                (ST.ScatteringCore.ScatteringTransform{2}, (16, 16), (16, 16), 2, (L = 4,), stored),
                (ST.ScatteringCore.ScatteringTransform{2}, (16, 16), (16, 16), 2, (L = 4,), computed),
                (ST.ScatteringCore.ScatteringTransform{3}, (8, 8, 8), (8, 8, 8), 2, (n_orient = 4,), computed),
                (ST.Monogenic.MonogenicScatteringTransform, (64,), (64,), 3, (Q = 1,), computed),
                (ST.Monogenic.MonogenicScatteringTransform, (16, 16), (16, 16), 2, (Q = 1,), stored),
                (ST.Monogenic.MonogenicScatteringTransform, (8, 8, 8), (8, 8, 8), 2, (Q = 1,), computed))
            for (build, arg, dims, J, options, filters) in cases
                host = build(Float32, arg, J; filters, options..., transform = FB)
                dev = build(Float32, arg, J, gpu; filters, options...)
                x = randn(Float32, dims...)
                xj = JLArrays.JLArray(x)
                Test.@test _gpu_flat(dev(xj)) ≈ _gpu_flat(host(x)) rtol = 1e-4
                X = randn(Float32, dims..., 3)
                Test.@test Array(ST.scattering_batch(dev, JLArrays.JLArray(X); execution = gpu)) ≈
                           ST.scattering_batch(host, X) rtol = 1e-4
                f = ST.ScatteringFields.scattering_field(dev, xj; subsample = 2)
                ref = ST.ScatteringFields.scattering_field(host, x; subsample = 2)
                Test.@test Array(f.data) ≈ ref.data rtol = 1e-4
            end
        end

        Test.@testset "scattered planar adjoints and fits" begin
            M, ms, J = 180, (8, 8), 2
            x, y = rand(M), rand(M)
            X = [1 + 0.4cos(2π * x[k]) + 0.2b * sin(2π * y[k]) for k in 1:M, b in 1:3]
            transform = SpectralBackends.DirectSumSpectralBackend()
            for analysis in (ST.Plans.WeightedAdjoint(), ST.Plans.LeastSquares())
                options = (; period = 1.0, L = 4, transform, analysis, ntrans = 2,
                            filters = computed, maxiter = 200, rtol = 1e-8)
                host = ST.ScatteredPlanar.build(Float64, x, y, ms, J; options...)
                dev = ST.ScatteredPlanar.build(Float64, x, y, ms, J, gpu; options...)
                Test.@test Array(ST.scattering_batch(dev, JLArrays.JLArray(X); execution = gpu)) ≈
                           ST.scattering_batch(host, X) rtol = 1e-6
                foreach(ST.close_transform!, (host, dev))
            end
        end

        Test.@testset "spherical batches and monogenic components" begin
            M, lmax, J = 100, 4, 2
            θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
            φ = [2π * mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:M]
            X = [cos(θ[k]) + 0.2b * sin(θ[k]) * cos(φ[k]) for k in 1:M, b in 1:3]
            for build in (ST.spherical_scattering, ST.spherical_monogenic_scattering)
                nufft = SpectralBackends.DirectSumSpectralBackend()
                host = build(θ, φ, lmax, J; nufft)
                dev = build(JLArrays.JLArray(θ), JLArrays.JLArray(φ), lmax, J;
                            nufft)
                Test.@test _gpu_flat(dev(JLArrays.JLArray(X[:, 1]))) ≈ _gpu_flat(host(X[:, 1])) rtol = 1e-6
                Test.@test Array(ST.scattering_batch(dev, JLArrays.JLArray(X); execution = gpu)) ≈
                           ST.scattering_batch(host, X) rtol = 1e-6
                if build === ST.spherical_monogenic_scattering
                    got = ST.spherical_monogenic_components(dev, JLArrays.JLArray(X[:, 1]), 1)
                    ref = ST.spherical_monogenic_components(host, X[:, 1], 1)
                    Test.@test Array(got.amplitude) ≈ ref.amplitude rtol = 1e-6
                end
                foreach(ST.close_transform!, (host, dev))
            end
        end

        Test.@testset "complex batches retain the field mean" begin
            dims, B = (16, 16), 3
            host = ST.ScatteringCore.ScatteringTransform(Float32, dims, 2; L = 4, transform = FB)
            dev = ST.ScatteringCore.ScatteringTransform(Float32, dims, 2, gpu; L = 4)
            X = randn(ComplexF32, dims..., B)
            Test.@test Array(ST.scattering_batch(dev, JLArrays.JLArray(X); execution = gpu)) ≈
                       ST.scattering_batch(host, X) rtol = 1e-4
        end

        Test.@testset "phase retrieval and the frame inverse" begin
            Inv = ST.Inverse
            N, J = 128, 5
            x = randn(Float32, N)
            xj = JLArrays.JLArray(x)
            host = ST.ScatteringCore.ScatteringTransform(Float32, N, J; max_order = 1, transform = FB)
            dev = ST.ScatteringCore.ScatteringTransform(Float32, N, J, gpu; max_order = 1)
            Test.@test Array(Inv.iwavelet(dev, Inv.wavelet_transform(dev, xj))) ≈ x rtol = 1e-4
            wt = Inv.wavelet_transform(host, x)
            moduli = map(w -> JLArrays.JLArray(abs.(w)), wt.wavelet)
            Test.@test Array(Inv.reconstruct_phase(dev, moduli; iters = 5, init = xj,
                                                   seed_lowpass = JLArrays.JLArray(wt.lowpass))) ≈ x rtol = 1e-4
            xr = Array(Inv.reconstruct_phase(dev, moduli; iters = 2))
            Test.@test size(xr) == (N,)
            Test.@test all(isfinite, xr)
        end
    end
end
