import STAGATE
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from glob import glob
import seaborn as sns
import matplotlib as mpl
import sklearn.neighbors
import matplotlib.pyplot as plt
from benchmarks.bm_functions import get_correlation_df, collect_correlation_results, collect_metadata
import numpy as np


filepath = 'data/mouse_brain_anterior/ffpe_cranial/'
adata = sc.read_h5ad(filepath + 'chr.h5ad')

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# include 8 neighbours with the cutoff similarly to 6 for visium
STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
STAGATE.Stats_Spatial_Net(adata)

coor = pd.DataFrame(adata.obsm['spatial'])
coor.index = adata.obs.index
coor.columns = ['imagerow', 'imagecol']

nbrs = sklearn.neighbors.NearestNeighbors(radius=150).fit(coor)
distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
KNN_list = []
for it in range(indices.shape[0]):
    KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))
KNN_df = pd.concat(KNN_list)
KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

adata = STAGATE.train_STAGATE(adata, alpha=0)

for i in range(30):
    adata.obs[f'stagate_{i}'] = adata.obsm['STAGATE'][:, i]

with mpl.rc_context({'figure.figsize': [4.5, 5]}):
    sc.pl.spatial(adata, color=[f'stagate_{i}' for i in range(30)], size=2, ncols=4, show=False)
    plt.savefig(filepath + 'stagate_comps.png')
    plt.close()

# correlation with tissue zones
stagate_df = adata.obs[[f'stagate_{i}' for i in range(30)]]
stagate_df.to_csv(filepath + 'stagate_comps.csv')
