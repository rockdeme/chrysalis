```{eval-rst}
.. automodule:: chrysalis
```

# API overview

Import chrysalis as:

```
import chrysalis as ch
```

## Core functions

Identifying spatially variable genes, dimensionality reduction, archetypal analysis.

Main functions required to identify tissue compartments.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   detect_svgs
   pca
   aa
```

## Plotting

Visualization module.

### Tissue compartments

Visualizations to examine the identified compartments in the tissue space.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   plot
   plot_compartment
   plot_compartments
```

### Quality control

Plot quality control metrics to determine the correct number of spatially variable genes or PCs (Principal Components).

```{eval-rst}
.. autosummary::
   :toctree: generated/

   plot_explained_variance
   plot_svgs
```

### Compartment-associated genes

Generate a visualization of the top-contributing genes for each tissue compartment.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   plot_heatmap
   plot_weights
```

## Utility functions

Sample interation, spatially variable gene contributions.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   integrate_adatas
   get_compartment_df
```
