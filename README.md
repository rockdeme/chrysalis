# chrysalis

<p align="center">
   <img src="misc/logo.svg" width="500">
</p>

**chrysalis** is a visualization tool for generating neat and meaningful visual representation of spatial transcriptomics datasets based on spatially variable genes, leveraging PCA and Archetype Analysis. 

It is designed to seamlessly integrate into `scanpy` pipelines based on `anndata` objects.

By combining PCA with AA, distinct tissue compartments and cellular niches can be defined and highlighted with specific colors.

On the `V1_Human_Lymph_Node` dataset, we can identify different regions, such as germinal centers (yellow), B cell follicles (dark orange) and T cell compartments (lime) using **chrysalis**. There are more examples in the gallery section.
<p align="center">
   <img src="plots/V1_Human_Lymph_Node_aa.svg" width="500">
</p>

## Package
**chrysalis** can be used with any preexisting snapshot of 10X Visium dataset and on new samples without the need of preprocessing. It is designed to be as lightweight as possible, however it relies on `libpysal` for its fast implementation of Moran's I.

**chrysalis** requires the following packages before installation:
- asd
- asd

To install **chrysalis**:
```terminal
pip install chrysalis
```
## Usage

```python
import chrysalis as ch

adata = sc.datasets.visium_sge(sample_id=sample)
ch.calculate(adata)
ch.plot_aa(adata, pcs=8)
plt.show()
```
`ch.calculate(adata)` stores some data under `adata.uns` allowing `ch.plot(adata)` to be called without the need of recalculating the embeddings every time.

## Gallery


