using Documenter
using ScatteringTransforms

makedocs(
    sitename = "ScatteringTransforms.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jbphyswx.github.io/ScatteringTransforms.jl",
        assets = String["assets/style.css"],
    ),
    modules = [
        ScatteringTransforms,
        ScatteringTransforms.Filters,
        ScatteringTransforms.FilterBanks,
        ScatteringTransforms.ScatteringCore,
        ScatteringTransforms.Coefficients,
        ScatteringTransforms.Scattering1D,
        ScatteringTransforms.Scattering2D,
    ],
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API Reference" => "api.md",
    ],
    repo = "https://github.com/jbphyswx/ScatteringTransforms.jl/blob/{commit}{path}#L{line}",
    authors = "Jordan Benjamin",
    warnonly = true,
)

deploydocs(
    repo = "github.com/jbphyswx/ScatteringTransforms.jl",
    devbranch = "main",
)
