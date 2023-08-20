<p align="center">
   <img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/misc/banner.png" width="850">
</p>

**chrysalis** is a powerful and lightweight method designed to identify and visualise tissue compartments in spatial 
transcriptomics datasets, all without the need for external references.
**chrysalis** achieves this by identifying spatially variable genes (SVGs) through spatial autocorrelation. 
It then employs dimensionality reduction and archetypal analysis to locate extremal points in the low-dimensional 
feature space, which represent pure tissue compartments.
Each observation (i.e. capture spot) in the gene expression matrix is subsequently represented as a proportion of these 
distinct compartments.
**chrysalis** features a unique approach based on maximum intensity projection, allowing the simultaneous visualization 
of diverse tissue compartments.
Moreover, it seamlessly integrates into `scanpy` based pipelines.

If you like **chrysalis**, consider citing our [preprint](https://github.com/rockdeme/chrysalis/#reference).

<p align="center">
   <img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/misc/panel_1.png" width=800">
</p>

**chrysalis** can define distinct tissue compartments and cellular niches with specific gene expression signatures, 
highlighted with specific colors. For instance, in the `V1_Human_Lymph_Node` dataset, **chrysalis** identifies
various regions, such as germinal centers (yellow), B cell follicles (dark orange), and T cell compartments 
(lime). You can find more examples in the [gallery](https://github.com/rockdeme/chrysalis/tree/master/gallery#readme) section.

<p align="center">
   <img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/misc/panel_2.png" width="850">
</p>

## Package
**chrysalis** can be used with any pre-existing `anndata` snapshot of 10X Visium, Slide-seqV2 and Stereo-seq datasets 
generated with `scanpy`, and on new samples without the need of preprocessing. It is designed to be as lightweight as 
possible, however currently it relies on `libpysal` for its fast implementation of Moran's I.

**chrysalis** requires the following packages:
- numpy
- pandas
- matplotlib
- scanpy
- pysal
- archetypes
- scikit_learn
- scipy
- tqdm
- seaborn

To install **chrysalis**:
```terminal
pip install chrysalis-st
```

## Documentation, Tutorials and API details

User documentation is available on: https://chrysalis.readthedocs.io/

Basic tutorials covering the main functionality of **chrysalis** are available on the documentation site.
- first step-by-step tutorial: https://chrysalis.readthedocs.io/en/latest/tutorials/lymph_node_tutorial.html


## Usage

```python
import chrysalis as ch
import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

ch.detect_svgs(adata)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

ch.pca(adata)

ch.aa(adata, n_pcs=20, n_archetypes=8)

ch.plot(adata)
plt.show()
```

## Reference

**Chrysalis: decoding tissue compartments in spatial transcriptomics with archetypal analysis**  
*Demeter Turos, Jelica Vasiljevic, Kerstin Hahn, Sven Rottenberg, Alberto Valdeolivas*  
doi: https://doi.org/10.1101/2023.08.17.553606
