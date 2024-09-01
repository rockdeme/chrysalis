import time
import mofax
import pandas as pd
import scanpy as sc
from mofapy2.run.entry_point import entry_point


datadir = "/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/xenium_human_breast_cancer/mefisto/"

data_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/xenium_human_breast_cancer/'
adata = sc.read_h5ad(data_path + 'visium_sample.h5ad')

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

adata.obs = pd.concat([adata.obs,
                       pd.DataFrame(adata.obsm["spatial"], columns=["imagerow", "imagecol"], index=adata.obs_names),
                      ], axis=1)


ent = entry_point()
ent.set_data_options(use_float32=True)
ent.set_data_from_anndata(adata, features_subset="highly_variable")

ent.set_model_options(factors=8)
ent.set_train_options()
ent.set_train_options(seed=2021)

# We use 1000 inducing points to learn spatial covariance patterns
n_inducing = 1000

ent.set_covariates([adata.obsm["spatial"]], covariates_names=["imagerow", "imagecol"])
ent.set_smooth_options(sparseGP=True, frac_inducing=n_inducing/adata.n_obs,
                       start_opt=10, opt_freq=10)


ent.build()
ent.run()
ent.save(datadir + "ST_model.hdf5")

m = mofax.mofa_model(datadir + "ST_model.hdf5")
factor_df = m.get_factors(df=True)
factor_df.to_csv(datadir + 'factors.csv')