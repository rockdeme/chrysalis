import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import archetypes as arch
from anndata import AnnData
from pysal.lib import weights
from pysal.explore import esda
from sklearn.decomposition import PCA


def detect_svgs(adata: AnnData, min_spots: float=0.05, top_svg: int=1000, min_morans: float=0.20, neighbors: int=6,
                geary: bool=False):
    """
    Calculate spatial autocorrelation (Moran's I) to define spatially variable genes.

    By default we only calculate autocorrelation for genes expressed in at least 5% of capture spots
    defined with `min_spots`.

    :param adata:
        The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
        Spatial data needs to be stored in `.obsm['spatial']` as X and Y coordinate columns.
    :param min_spots: Run calculation only for genes expressed in equal or higher fraction of the total capture spots.
    :param top_svg: Number of spatially variable genes to keep.
    :param min_morans: Cutoff using Moran's I. Maximum SVG number defined by `top_svg` won't be affected.
    :param neighbors: Number of nearest neighbours used for calculating autocorrelation.
    :param geary: Calculate Geary's C. Selected SVGs are not affect by this, stored in `.var["Geary's C"]`.
    :return:
        Updates `.var` with the following fields:

        - **.var["Moran's I"]** – Moran's I value for all genes.
        - **.var["spatially_variable"]** – Boolean labels of the examined genes based on the defined cutoffs.

    Example usage:

    >>> import chrysalis as ch
    >>> import scanpy as sc
    >>> adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pp.filter_cells(adata, min_counts=6000)
    >>> sc.pp.filter_genes(adata, min_cells=10)
    >>> ch.detect_svgs(adata)

    """

    assert 0 < min_spots < 1

    sc.settings.verbosity = 0
    ad = sc.pp.filter_genes(adata, min_cells=int(len(adata) * min_spots), copy=True)
    ad.var_names_make_unique()  # moran dies so need some check later
    if "log1p" not in adata.uns_keys():
        sc.pp.normalize_total(ad, inplace=True)
        sc.pp.log1p(ad)

    gene_matrix = ad.to_df()

    points = adata.obsm['spatial'].copy()
    points[:, 1] = points[:, 1] * -1

    w = weights.KNN.from_array(points, k=neighbors)
    w.transform = 'R'

    moran_dict = {}
    if geary:
        geary_dict = {}

    for c in tqdm(ad.var_names, desc='Calculating SVGs'):
        moran = esda.moran.Moran(gene_matrix[c], w, permutations=0)
        moran_dict[c] = moran.I
        if geary:
            geary = esda.geary.Geary(gene_matrix[c], w, permutations=0)
            geary_dict[c] = geary.C

    moran_df = pd.DataFrame(data=moran_dict.values(), index=moran_dict.keys(), columns=["Moran's I"])
    moran_df = moran_df.sort_values(ascending=False, by="Moran's I")
    adata.var["Moran's I"] = moran_df["Moran's I"]

    if geary:
        geary_df = pd.DataFrame(data=geary_dict.values(), index=geary_dict.keys(), columns=["Geary's C"])
        geary_df = geary_df.sort_values(ascending=False, by="Geary's C")
        adata.var["Geary's C"] = geary_df["Geary's C"]

    # select thresholds
    if len(moran_df[:top_svg]) < len(moran_df[moran_df["Moran's I"] > min_morans]):
        adata.var['spatially_variable'] = [True if x in moran_df[:top_svg].index else False for x in adata.var_names]
    else:
        moran_df = moran_df[moran_df["Moran's I"] > min_morans]
        adata.var['spatially_variable'] = [True if x in moran_df.index else False for x in adata.var_names]


def pca(adata: AnnData, n_pcs: int=50):
    """
    Perform PCA (Principal Component Analysis) to calculate PCA coordinates, loadings, and variance decomposition.

    Spatially variable genes need to be defined in `.var['spatially_variable']` using `chrysalis.detect_svgs`.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param n_pcs: Number of principal components to be calculated.
    :return:
        Adds PCs to `.obsm['chr_X_pca']` and updates `.uns` with the following fields:

        - **.uns['chr_pca']['variance_ratio']** – Explained variance ratio.
        - **.uns['chr_pca']['loadings']** – Spatially variable gene loadings.
        - **.uns['chr_pca']['features']** – Spatially variable gene names.

    Example usage:

    >>> import chrysalis as ch
    >>> import scanpy as sc
    >>> adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pp.filter_cells(adata, min_counts=6000)
    >>> sc.pp.filter_genes(adata, min_cells=10)
    >>> ch.detect_svgs(adata)
    >>> sc.pp.normalize_total(adata, inplace=True)
    >>> sc.pp.log1p(adata)
    >>> ch.pca(adata)

    """

    # todo: this only works with CSL matrix, need something to check if the matrix is dense
    pcs = np.asarray(adata[:, adata.var['spatially_variable'] == True].X.todense())
    pca = PCA(n_components=n_pcs, svd_solver='arpack', random_state=42)
    adata.obsm['chr_X_pca'] = pca.fit_transform(pcs)

    if 'chr_pca' not in adata.uns.keys():
        adata.uns['chr_pca'] = {'variance_ratio': pca.explained_variance_ratio_,
                                'loadings': pca.components_,
                                'features': list(adata[:, adata.var['spatially_variable'] == True].var_names)}
    else:
        adata.uns['chr_pca']['variance_ratio'] = pca.explained_variance_ratio_
        adata.uns['chr_pca']['loadings'] = pca.components_
        adata.uns['chr_pca']['features'] = list(adata[:, adata.var['spatially_variable'] == True].var_names)


def aa(adata: AnnData, n_archetypes: int=8, n_pcs: int=None):
    """
    Run archetypal analysis on the low-dimensional embedding.

    Calculates archetypes, alphas, loadings, and RSS (Residual Sum of Squares). Requires input calculated with
    `chrysalis.pca`.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param n_archetypes: Number of archetypes (tissue compartments) to be identified.
    :param n_pcs: Number of PCs (Principal Components) to be used.
    :return:
        Updates `.uns` with the following fields:

        - **.uns['chr_aa']['archetypes']** – Archetypes.
        - **.uns['chr_aa']['alphas']** – Alphas.
        - **.uns['chr_aa']['loadings']** – Gene loadings.
        - **.uns['chr_aa']['RSS']** – RSS reconstruvtion error.

    Example usage:

    >>> import chrysalis as ch
    >>> import scanpy as sc
    >>> adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pp.filter_cells(adata, min_counts=6000)
    >>> sc.pp.filter_genes(adata, min_cells=10)
    >>> ch.detect_svgs(adata)
    >>> sc.pp.normalize_total(adata, inplace=True)
    >>> sc.pp.log1p(adata)
    >>> ch.pca(adata)
    >>> ch.aa(adata, n_pcs=20, n_archetypes=8)

    """

    if not isinstance(n_archetypes, int):
        raise TypeError
    if n_archetypes < 2:
        raise ValueError(f"n_archetypes cannot be less than 2.")

    if n_pcs is None:
        pcs = n_archetypes-1
    else:
        pcs = n_pcs

    model = arch.AA(n_archetypes=n_archetypes, n_init=3, max_iter=200, tol=0.001, random_state=42)
    model.fit(adata.obsm['chr_X_pca'][:, :pcs])
    adata.obsm['chr_aa'] = model.alphas_

    # get the mean of the original feature matrix and add it to the multiplied archetypes with the PCA loading matrix
    # aa_loadings = np.mean(pcs, axis=0) + np.dot(model.archetypes_.T, pca.components_[:n_archetypes, :])
    aa_loadings = np.dot(model.archetypes_, adata.uns['chr_pca']['loadings'][:pcs, :])

    if 'chr_aa' not in adata.uns.keys():
        adata.uns['chr_aa'] = {'archetypes': model.archetypes_,
                               'alphas': model.alphas_,
                               'loadings': aa_loadings,
                               'RSS': model.rss_}
    else:
        adata.uns['chr_aa']['archetypes'] = model.archetypes_
        adata.uns['chr_aa']['alphas'] = model.alphas_
        adata.uns['chr_aa']['loadings'] = aa_loadings
        adata.uns['chr_aa']['RSS'] = model.rss_
