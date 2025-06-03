import numpy as np
import scanpy as sc
import chrysalis as ch
from pysal.lib import weights
from libpysal.weights import WSP
from scipy.sparse import csr_matrix
from dev.fast_morans import moran_sparse_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#%%
# download data
adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
adata.write_h5ad('data/toy_data.h5ad')

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

ch.compute_svg_scores(adata)
ch.select_svgs(adata, top_svg=1000)

ch.pca(adata)
ch.aa(adata, n_pcs=20, n_archetypes=8)

ch.plot(adata, dim=8)
plt.show()

genes = adata.var.sort_values(by="Moran's I", ascending=False).index[:10]
vals = adata.var.sort_values(by="Moran's I", ascending=False)["Moran's I"][:10]

#%%
# new function for moran - test it against pysal

min_spots = 0.05
neighbors = 6

sc.settings.verbosity = 0
adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

adx = sc.pp.filter_genes(adata, min_cells=int(len(adata) * min_spots), copy=True)
adx.var_names_make_unique()  # moran dies so need some check later
if "log1p" not in adx.uns_keys():
    sc.pp.normalize_total(adx, inplace=True)
    sc.pp.log1p(adx)

ad = sc.pp.filter_genes(adata, min_cells=int(len(adata) * min_spots), copy=True)
ad.var_names_make_unique()  # moran dies so need some check later
if "log1p" not in ad.uns_keys():
    sc.pp.normalize_total(ad, inplace=True)
    sc.pp.log1p(ad)
ch.detect_svgs(ad, min_morans=0.08, min_spots=0.05)

points = adata.obsm['spatial']  # removed deep copy here, we don't need it probably
w = weights.KNN.from_array(points, k=neighbors)
w.transform = 'R'
w_sparse = w.sparse
w_sparse = w_sparse.tocsr()

X = adx[:, :].X.todense()

results = moran_sparse_matrix(X, w_sparse.data, w_sparse.indices, w_sparse.indptr)

plt.scatter(ad.var["Moran's I"], results)
plt.show()

#%%

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

ch.compute_svg_scores(adata)

plt.scatter(adata.var["n_cells_by_counts"], adata.var["morans"])
plt.show()

ch.select_svgs(adata, top_svg=1000)

ch.plot_svgs(adata)
plt.show()

ch.pca(adata)


