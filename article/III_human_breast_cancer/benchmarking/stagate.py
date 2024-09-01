import STAGATE
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import sklearn.neighbors


data_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/xenium_human_breast_cancer/'

adata = sc.read_h5ad(data_path + 'visium_sample.h5ad')

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STAGATE.Cal_Spatial_Net(adata, rad_cutoff=300)
STAGATE.Stats_Spatial_Net(adata)

coor = pd.DataFrame(adata.obsm['spatial'])
coor.index = adata.obs.index
coor.columns = ['imagerow', 'imagecol']


nbrs = sklearn.neighbors.NearestNeighbors(radius=1000).fit(coor)
distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
KNN_list = []
for it in range(indices.shape[0]):
    KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))

KNN_df = pd.concat(KNN_list)
KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

sc.pl.spatial(adata, color='CD74')

adata = STAGATE.train_STAGATE(adata, alpha=0)

for i in range(30):
    adata.obs[f'S{i}'] = adata.obsm['STAGATE'][:, i]


with mpl.rc_context({'figure.figsize': [4.5, 5]}):
        sc.pl.spatial(adata, color=[f'S{i}' for i in range(30)], size=2, ncols=4)

stagate_df = adata.obs[[f'S{i}' for i in range(30)]]
stagate_df.to_csv('stagate_breast_cancer.csv')
