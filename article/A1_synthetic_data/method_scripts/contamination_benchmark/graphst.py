import os
import torch
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from glob import glob
import seaborn as sns
from GraphST import GraphST
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from article.A1_synthetic_data.bm_functions import get_correlation_df


filepath = '/storage/homefs/pt22a065/chr_data/tabula_sapiens_immune_contamination'
adatas = glob(filepath + '/*/*.h5ad')

results_df = pd.DataFrame()

for idx, adp in tqdm(enumerate(adatas), total=len(adatas)):
    print(adp)
    sample_folder = '/'.join(adp.split('/')[:-1]) + '/'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Check if all necessary output files already exist in the sample_folder
    if (os.path.exists(sample_folder + 'graphst_comps.csv') and
            os.path.exists(sample_folder + 'graphst_pearson.csv') and
            os.path.exists(sample_folder + 'graphst_corr_heatmap.png')):
        print(f"Skipping {sample_folder} as output files already exist.")
        continue

    adata = sc.read_h5ad(adp)
    adata.var_names_make_unique()

    try:
        # first attempt without any filtering
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    except:
        try:
            # second attempt with min_counts=10
            sc.pp.filter_genes(adata, min_counts=10, inplace=True)
            # sc.pp.filter_cells(adata, min_counts=100, inplace=True)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        except:
            # final attempt with min_counts=100
            sc.pp.filter_genes(adata, min_counts=100, inplace=True)
            # sc.pp.filter_cells(adata, min_counts=100, inplace=True)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    # define model
    model = GraphST.GraphST(adata, device=torch.device('cpu'))

    # train model
    adata = model.train()

    graphst_df = pd.DataFrame(adata.obsm['emb'])
    graphst_df.index = adata.obs.index

    pca = PCA(n_components=20, svd_solver='arpack', random_state=42)
    graphst_pcs = pca.fit_transform(graphst_df)
    graphst_pcs_df = pd.DataFrame(data=graphst_pcs, index=graphst_df.index)

    graphst_pcs_df.to_csv(sample_folder + 'graphst_comps.csv')

    tissue_zone_df = adata.obsm['tissue_zones']
    # tissue_zone_df = tissue_zone_df[[c for c in tissue_zone_df.columns if 'uniform' not in c]]

    corr_df = get_correlation_df(tissue_zone_df, graphst_pcs_df)
    corr_df.to_csv(sample_folder + 'graphst_pearson.csv')

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    sns.heatmap(corr_df, square=True, center=0, ax=ax)
    plt.tight_layout()
    plt.savefig(sample_folder + 'graphst_corr_heatmap.png')
    plt.close()
