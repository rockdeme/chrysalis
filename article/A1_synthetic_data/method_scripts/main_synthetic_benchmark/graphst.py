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


filepath = 'data/tabula_sapiens_immune'
adatas = glob(filepath + '/*/*.h5ad')

results_df = pd.DataFrame()

for idx, adp in tqdm(enumerate(adatas), total=len(adatas)):
    print(adp)
    sample_folder = '/'.join(adp.split('/')[:-1]) + '/'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    adata = sc.read_h5ad(adp)
    adata.var_names_make_unique()

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