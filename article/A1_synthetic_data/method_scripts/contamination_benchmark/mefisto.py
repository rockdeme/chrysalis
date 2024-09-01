import os
import mofax
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from mofapy2.run.entry_point import entry_point
from article.A1_synthetic_data.bm_functions import get_correlation_df


filepath = '/storage/homefs/pt22a065/chr_data/tabula_sapiens_immune_contamination'
adatas = glob(filepath + '/*/*.h5ad')

results_df = pd.DataFrame()

for idx, adp in tqdm(enumerate(adatas), total=len(adatas)):
    print(adp)
    sample_folder = '/'.join(adp.split('/')[:-1]) + '/'

    # Check if all necessary output files already exist in the sample_folder
    if (os.path.exists(sample_folder + 'mefisto_comps.csv') and
            os.path.exists(sample_folder + 'mefisto_pearson.csv') and
            os.path.exists(sample_folder + 'mefisto_corr_heatmap.png')):
        print(f"Skipping {sample_folder} as output files already exist.")
        continue

    adata = sc.read_h5ad(adp)

    # number of non-uniform compartments
    uniform = int(adata.uns['parameters']['n_tissue_zones'] * adata.uns['parameters']['uniform'])
    tissue_zones = adata.uns['parameters']['n_tissue_zones'] - uniform
    tissue_zones = int(tissue_zones)

    sc.pp.filter_genes(adata, min_cells=len(adata) * 0.05)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

    adata.obs = pd.concat([adata.obs,
                           pd.DataFrame(adata.obsm["spatial"], columns=["imagerow", "imagecol"],
                                        index=adata.obs_names),
                           ], axis=1)

    ent = entry_point()
    ent.set_data_options(use_float32=True)
    ent.set_data_from_anndata(adata, features_subset="highly_variable")

    ent.set_model_options(factors=tissue_zones)
    ent.set_train_options(save_interrupted=True)
    ent.set_train_options(seed=2021)

    # We use 1000 inducing points to learn spatial covariance patterns
    n_inducing = 1000  # 500 for size tests

    ent.set_covariates([adata.obsm["spatial"]], covariates_names=["imagerow", "imagecol"])
    ent.set_smooth_options(sparseGP=True, frac_inducing=n_inducing / adata.n_obs,
                           start_opt=10, opt_freq=10)

    ent.build()
    ent.run()
    ent.save(sample_folder + "mefisto_temp.hdf5")
    m = mofax.mofa_model(sample_folder + "mefisto_temp.hdf5")
    factor_df = m.get_factors(df=True)

    # factor_df = ent.model.getFactors(df=True)

    factor_df.to_csv(sample_folder + 'mefisto_comps.csv')

    # correlation with tissue zones
    tissue_zone_df = adata.obsm['tissue_zones']
    # tissue_zone_df = tissue_zone_df[[c for c in tissue_zone_df.columns if 'uniform' not in c]]

    corr_df = get_correlation_df(tissue_zone_df, factor_df)
    corr_df.to_csv(sample_folder + 'mefisto_pearson.csv')

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    sns.heatmap(corr_df, square=True, center=0, ax=ax)
    plt.tight_layout()
    plt.savefig(sample_folder + 'mefisto_corr_heatmap.png')
    plt.close()
