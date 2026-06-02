using Documenter
using ScatteringTransforms

makedocs(
    sitename = "ScatteringTransforms.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jbphyswx.github.io/ScatteringTransforms.jl",
        assets = String["assets/style.css"],
    ),
    modules = [ScatteringTransforms],
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API Reference" => "api.md",
    ],
    repo = "https://github.com/jbphyswx/ScatteringTransforms.jl/blob/{commit}{path}#L{line}",
    sitename = "ScatteringTransforms.jl",
    authors = "Jordan Benjamin",
)

deploydocs(
    repo = "github.com/jbphyswx/ScatteringTransforms.jl",
    devbranch = "main",
)
