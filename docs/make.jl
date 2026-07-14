using Documenter: Documenter
using ScatteringTransforms: ScatteringTransforms   # @docs blocks use fully-qualified `ScatteringTransforms.…` paths

Documenter.makedocs(
    sitename = "ScatteringTransforms.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jbphyswx.github.io/ScatteringTransforms.jl",
    ),
    modules = [
        ScatteringTransforms,
        ScatteringTransforms.Backends,
        ScatteringTransforms.Domains,
        ScatteringTransforms.Plans,
        ScatteringTransforms.Filters,
        ScatteringTransforms.FilterBanks,
        ScatteringTransforms.PathGraph,
        ScatteringTransforms.ScatteringCore,
        ScatteringTransforms.Coefficients,
        ScatteringTransforms.ScatteringFields,
        ScatteringTransforms.Scattering1D,
        ScatteringTransforms.Scattering2D,
        ScatteringTransforms.Scattering3D,
        ScatteringTransforms.SubsampledScattering,
        ScatteringTransforms.Reductions,
    ],
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API Reference" => "api.md",
    ],
    authors = "Jordan Benjamin",
    warnonly = true,
)

Documenter.deploydocs(
    repo = "github.com/jbphyswx/ScatteringTransforms.jl",
    devbranch = "main",
)
