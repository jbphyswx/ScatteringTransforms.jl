# API Reference

## 1D Scattering

### Core Types

```@docs
ScatteringTransforms.Scattering1D.ScatteringTransform1D
```

### Functions

```@docs
ScatteringTransforms.Scattering1D.scattering_transform!
ScatteringTransforms.Scattering1D.compute_S1!
ScatteringTransforms.Scattering1D.compute_S2!
```

## 2D Scattering

### Core Types

```@docs
ScatteringTransforms.Scattering2D.ScatteringTransform2D
```

### Functions

```@docs
ScatteringTransforms.Scattering2D.compute_shape_sparsity
```

## Coefficient Storage

```@docs
ScatteringTransforms.Coefficients.ScatteringCoefficients1D
ScatteringTransforms.Coefficients.ScatteringCoefficients2D
ScatteringTransforms.Coefficients.zeroth_order
ScatteringTransforms.Coefficients.first_order
ScatteringTransforms.Coefficients.second_order
ScatteringTransforms.Coefficients.update_S0
ScatteringTransforms.Coefficients.flatten1d
ScatteringTransforms.Coefficients.flatten2d
```

## Core Operations

```@docs
ScatteringTransforms.ScatteringCore.wavelet_convolve
ScatteringTransforms.ScatteringCore.wavelet_convolve!
ScatteringTransforms.ScatteringCore.apply_modulus
ScatteringTransforms.ScatteringCore.apply_modulus!
ScatteringTransforms.ScatteringCore.spatial_average
ScatteringTransforms.ScatteringCore.ScatteringLayer
```

## Filter Banks

```@docs
ScatteringTransforms.FilterBanks.FilterBank1D
ScatteringTransforms.FilterBanks.FilterBank2D
ScatteringTransforms.FilterBanks.build_filter_bank1d
ScatteringTransforms.FilterBanks.build_filter_bank2d
```

## Filters

```@docs
ScatteringTransforms.Filters.Morlet1D
ScatteringTransforms.Filters.Morlet2D
ScatteringTransforms.Filters.frequency_response
```
