import STAGATE
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from glob import glob
import seaborn as sns
import sklearn.neighbors
import matplotlib.pyplot as plt
from article.I_synthetic_data.bm_functions import get_correlation_df


filepath = 'data/tabula_sapiens_immune_size'
adatas = glob(filepath + '/*/*.h5ad')

results_df = pd.DataFrame()

for idx, adp in tqdm(enumerate(adatas), total=len(adatas)):
    print(adp)
    sample_folder = '/'.join(adp.split('/')[:-1]) + '/'
    adata = sc.read_h5ad(adp)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # include 8 neighbours with the cutoff similarly to 6 for visium
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=3.3)
    STAGATE.Stats_Spatial_Net(adata)

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(radius=3.3).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    adata = STAGATE.train_STAGATE(adata, alpha=0)

    for i in range(30):
        adata.obs[f'stagate_{i}'] = adata.obsm['STAGATE'][:, i]

    # doesn't work without the proper anndata structure - missing some spatial info
    # with mpl.rc_context({'figure.figsize': [4.5, 5]}):
    #     sc.pl.spatial(adata, color=[f'S{i}' for i in range(30)], size=2, ncols=4, show=False)
    #     plt.savefig(sample_folder + 'stagate_comps.png')
    #     plt.close()

    # correlation with tissue zones
    stagate_df = adata.obs[[f'stagate_{i}' for i in range(30)]]
    stagate_df.to_csv(sample_folder + 'stagate_comps.csv')

    tissue_zone_df = adata.obsm['tissue_zones']
    # tissue_zone_df = tissue_zone_df[[c for c in tissue_zone_df.columns if 'uniform' not in c]]

    corr_df = get_correlation_df(tissue_zone_df, stagate_df)
    corr_df.to_csv(sample_folder + 'stagate_pearson.csv')

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    sns.heatmap(corr_df, square=True, center=0, ax=ax)
    plt.tight_layout()
    plt.savefig(sample_folder + 'stagate_corr_heatmap.png')
    plt.close()
