import os
import math
import random
import numpy as np
import pandas as pd
import scanpy as sc
from os import path
from tqdm import tqdm
from glob import glob
import seaborn as sns
from nsf.models import sf
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from tensorflow_probability import math as tm
from article.A1_synthetic_data.bm_functions import get_correlation_df
from nsf.utils import preprocess, training, visualize, postprocess, misc


filepath = '/storage/homefs/pt22a065/chr_data/tabula_sapiens_immune_contamination'
adatas = glob(filepath + '/*/*.h5ad')
tfk = tm.psd_kernels

results_df = pd.DataFrame()

for idx, adp in tqdm(enumerate(adatas), total=len(adatas)):
    try:
        print(adp)
        sample_folder = '/'.join(adp.split('/')[:-1]) + '/'

        # Check if all necessary output files already exist in the sample_folder
        if (os.path.exists(sample_folder + 'nsf_comps.csv') and
                os.path.exists(sample_folder + 'nsf_pearson.csv') and
                os.path.exists(sample_folder + 'nsf_corr_heatmap.png')):
            print(f"Skipping {sample_folder} as output files already exist.")
            continue

        adata = sc.read_h5ad(adp)

        # number of non-uniform compartments
        uniform = int(adata.uns['parameters']['n_tissue_zones'] * adata.uns['parameters']['uniform'])
        tissue_zones = adata.uns['parameters']['n_tissue_zones'] - uniform
        tissue_zones = int(tissue_zones)

        sc.pp.calculate_qc_metrics(adata, inplace=True)
        # sc.pp.filter_genes(adata, min_cells=len(adata) * 0.05)
        sc.pp.filter_genes(adata, min_counts=100, inplace=True)

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        adata.layers = {"counts": adata.X.copy()}  # store raw counts before normalization changes ad.X
        sc.pp.normalize_total(adata, inplace=True, layers=None, key_added="sizefactor")
        sc.pp.log1p(adata)

        adata.var['deviance_poisson'] = preprocess.deviancePoisson(adata.layers["counts"])
        o = np.argsort(-adata.var['deviance_poisson'])
        idx = list(range(adata.shape[0]))
        random.shuffle(idx)
        adata = adata[idx, o]
        adata = adata[:, :2000]

        Dtr, Dval = preprocess.anndata_to_train_val(adata, layer="counts",
                                                    sz="constant")  # size factor set to constant from scanpy for
                                                                    # the samples with y nan error
        Dtr_n, Dval_n = preprocess.anndata_to_train_val(adata)  # normalized data
        fmeans, Dtr_c, Dval_c = preprocess.center_data(Dtr_n, Dval_n)  # centered features
        Xtr = Dtr["X"]  # note this should be identical to Dtr_n["X"]
        Ntr = Xtr.shape[0]
        Dtf = preprocess.prepare_datasets_tf(Dtr, Dval=Dval, shuffle=False)
        Dtf_n = preprocess.prepare_datasets_tf(Dtr_n, Dval=Dval_n, shuffle=False)
        Dtf_c = preprocess.prepare_datasets_tf(Dtr_c, Dval=Dval_c, shuffle=False)
        visualize.heatmap(Xtr, Dtr["Y"][:, 0], marker="D", s=15)
        plt.close()

        # Visualize raw data
        plt.imshow(np.log1p(Dtr["Y"])[:50, :100], cmap="Blues")
        plt.close()

        # Visualize inducing points
        Z = misc.kmeans_inducing_pts(Xtr, 500)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(Xtr[:, 0], Xtr[:, 1], marker="D", s=50, )
        ax.scatter(Z[:, 0], Z[:, 1], c="red", s=30)
        plt.close()

        # initialize inducing points and tuning parameters
        Z = misc.kmeans_inducing_pts(Xtr, 1000)  # 2363
        M = Z.shape[0]
        ker = tfk.MaternThreeHalves
        S = 3  # samples for elbo approximation
        # NSF: Spatial only with non-negative factors
        L = tissue_zones  # number of latent factors, ideally divisible by 2
        J = adata.shape[1]  # 2000

        mpth = path.join("/storage/homefs/pt22a065/chr_benchmarks/nsf/models/V5/")

        fit = sf.SpatialFactorization(J, L, Z, psd_kernel=ker, nonneg=True, lik="poi")
        fit.elbo_avg(Xtr, Dtr["Y"], sz=Dtr["sz"])
        fit.init_loadings(Dtr["Y"], X=Xtr, sz=Dtr["sz"])
        fit.elbo_avg(Xtr, Dtr["Y"], sz=Dtr["sz"])
        pp = fit.generate_pickle_path("scanpy", base=mpth)
        tro = training.ModelTrainer(fit, pickle_path=pp)
        tro.train_model(*Dtf, ckpt_freq=10000)

        ttl = "NSF: spatial, non-negative factors, Poisson likelihood"
        visualize.plot_loss(tro.loss, title=ttl)  # ,ss=range(2000,4000))
        plt.savefig(sample_folder + 'nsf_loss.png')
        plt.close()

        hmkw = {"figsize": (4, 4), "s": 0.3, "marker": "D", "subplot_space": 0,
                "spinecolor": "white"}
        insf = postprocess.interpret_nsf(fit, Xtr, S=10, lda_mode=False)
        tgnames = [str(i) for i in range(1, L + 1)]

        # fig, axes = visualize.multiheatmap(Xtr, np.sqrt(insf["factors"]), (4, 3), **hmkw)
        # visualize.set_titles(fig, tgnames, x=0.05, y=.85, fontsize="medium", c="white",
        #                      ha="left", va="top")
        # plt.savefig(sample_folder + 'nsf_comps.png')
        # plt.close()

        data = {'factors': insf, 'positions': Xtr}


        def transform_coords(X):
            # code from nsf github
            X[:, 1] = -X[:, 1]
            xmin = X.min(axis=0)
            X -= xmin
            x_gmean = np.exp(np.mean(np.log(X.max(axis=0))))
            X *= 4 / x_gmean
            return X - X.mean(axis=0)


        X = adata.obsm["spatial"].copy().astype('float32')
        tcoords = transform_coords(X)

        pair_idx = []
        for xy in data['positions']:
            distances = [math.dist([xy[0], xy[1]], [idx[0], idx[1]]) for idx in tcoords]
            pair_idx.append(np.argmin(distances))

        nsf_df = pd.DataFrame(data=np.zeros([len(adata), data['factors']['factors'].shape[1]]))
        for idx, i in enumerate(pair_idx):
            nsf_df.iloc[i, :] = data['factors']['factors'][idx, :]
        nsf_df.index = adata.obs.index

        nsf_df.to_csv(sample_folder + 'nsf_comps.csv')
        nsf_df = nsf_df[~(nsf_df.eq(0).all(axis=1))]

        tissue_zone_df = adata.obsm['tissue_zones'].loc[list(nsf_df.index)]

        corr_df = get_correlation_df(tissue_zone_df, nsf_df)
        corr_df.to_csv(sample_folder + 'nsf_pearson.csv')

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        sns.heatmap(corr_df, square=True, center=0, ax=ax)
        plt.tight_layout()
        plt.savefig(sample_folder + 'nsf_corr_heatmap.png')
        plt.close()

    except (LinAlgError, ValueError) as e:
        print(f"LinAlgError encountered for matrix. Continuing...{e}")
        continue
