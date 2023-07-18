# Getting started

Quickstart guide to install and run **chrysalis**.

## Install

```shell
pip install chrysalis-st
```

## Run

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
