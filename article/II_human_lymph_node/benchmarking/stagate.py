import pandas as pd
import scanpy as sc
import matplotlib as mpl
import STAGATE
import time


start_time = time.time()

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
sc.pp.calculate_qc_metrics(adata, inplace=True)

sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
STAGATE.Stats_Spatial_Net(adata)

coor = pd.DataFrame(adata.obsm['spatial'])
coor.index = adata.obs.index
coor.columns = ['imagerow', 'imagecol']
import sklearn.neighbors

nbrs = sklearn.neighbors.NearestNeighbors(radius=150).fit(coor)
distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
KNN_list = []
for it in range(indices.shape[0]):
    KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
KNN_df = pd.concat(KNN_list)
KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

adata = STAGATE.train_STAGATE(adata, alpha=0)

for i in range(30):
    adata.obs[f'S{i}'] = adata.obsm['STAGATE'][:, i]

with mpl.rc_context({'figure.figsize': [4.5, 5]}):
        sc.pl.spatial(adata, color=[f'S{i}' for i in range(30)], size=2, ncols=4)

stagate_df = adata.obs[[f'S{i}' for i in range(30)]]
stagate_df.to_csv('stagate_lymph_node.csv')

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)