import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import seaborn as sns
from glob import glob
import chrysalis as ch
import matplotlib.pyplot as plt
from article.A1_synthetic_data.bm_functions import get_correlation_df


filepath = 'data/tabula_sapiens_immune_size'
adatas = glob(filepath + '/*/*.h5ad')

results_df = pd.DataFrame()

for idx, adp in tqdm(enumerate(adatas), total=len(adatas)):
    print(adp)
    sample_folder = '/'.join(adp.split('/')[:-1]) + '/'
    adata = sc.read_h5ad(adp)

    # number of non-uniform compartments
    uniform = int(adata.uns['parameters']['n_tissue_zones'] * adata.uns['parameters']['uniform'])
    tissue_zones = adata.uns['parameters']['n_tissue_zones'] - uniform
    tissue_zones = int(tissue_zones)

    # chrysalis pipeline
    ch.detect_svgs(adata, neighbors=8, top_svg=1000, min_morans=0.01)
    ch.plot_svgs(adata)
    plt.savefig(sample_folder + 'ch_svgs.png')
    plt.close()

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    ch.pca(adata, n_pcs=40)

    ch.plot_explained_variance(adata)
    plt.savefig(sample_folder + 'ch_expl_variance.png')
    plt.close()

    ch.aa(adata, n_pcs=20, n_archetypes=tissue_zones)

    ch.plot(adata, dim=tissue_zones, marker='s')
    plt.savefig(sample_folder + 'ch_plot.png')
    plt.close()

    col_num = int(np.sqrt(tissue_zones))
    ch.plot_compartments(adata, marker='s', ncols=col_num)
    plt.savefig(sample_folder + 'ch_comps.png')
    plt.close()

    # correlation with tissue zones
    compartments = adata.obsm['chr_aa']
    compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)
    tissue_zone_df = adata.obsm['tissue_zones']
    # tissue_zone_df = tissue_zone_df[[c for c in tissue_zone_df.columns if 'uniform' not in c]]

    corr_df = get_correlation_df(tissue_zone_df, compartment_df)
    corr_df.to_csv(sample_folder + 'pearson.csv')

    sns.heatmap(corr_df, square=True, center=0)
    plt.tight_layout()
    plt.savefig(sample_folder + 'corr_heatmap.png')
    plt.close()
