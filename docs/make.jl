using Documenter: Documenter
using ScatteringTransforms: ScatteringTransforms   # @docs blocks use fully-qualified `ScatteringTransforms.…` paths

Documenter.makedocs(
    sitename = "ScatteringTransforms.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jbphyswx.github.io/ScatteringTransforms.jl",
        # The API reference is one page covering every surface, so it runs past the default warning
        # size. Splitting it would scatter the reference rather than improve it.
        size_threshold_warn = 300 * 1024,
        size_threshold = 400 * 1024,
    ),
    modules = [
        ScatteringTransforms,
        ScatteringTransforms.Execution,
        ScatteringTransforms.Plans,
        ScatteringTransforms.Filters,
        ScatteringTransforms.FilterBanks,
        ScatteringTransforms.PathGraph,
        ScatteringTransforms.ScatteringCore,
        ScatteringTransforms.Batched,
        ScatteringTransforms.Coefficients,
        ScatteringTransforms.ScatteringFields,
        ScatteringTransforms.Scattering1D,
        ScatteringTransforms.Scattering2D,
        ScatteringTransforms.Scattering3D,
        ScatteringTransforms.ScatteredPlanar,
        ScatteringTransforms.SubsampledScattering,
        ScatteringTransforms.Monogenic,
        ScatteringTransforms.SphericalCore,
        ScatteringTransforms.Inverse,
        ScatteringTransforms.Reductions,
    ],
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API Reference" => "api.md",
    ],
    authors = "Jordan Benjamin",
)

Documenter.deploydocs(
    repo = "github.com/jbphyswx/ScatteringTransforms.jl",
    devbranch = "main",
)
