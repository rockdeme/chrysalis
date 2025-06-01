import numpy as np
import scanpy as sc
import chrysalis as ch
from pysal.lib import weights
from libpysal.weights import WSP
from scipy.sparse import csr_matrix
from dev.fast_morans import moran_sparse_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# download data
adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
adata.write_h5ad('data/toy_data.h5ad')

# normalize and calculate Moran's I
adata_norm = adata.copy()
sc.pp.calculate_qc_metrics(adata_norm, inplace=True)
sc.pp.filter_cells(adata_norm, min_counts=6000)
sc.pp.filter_genes(adata_norm, min_cells=10)

ch.detect_svgs(adata_norm, min_morans=0.08, min_spots=0.05)

genes = adata_norm.var.sort_values(by="Moran's I", ascending=False).index[:10]
vals = adata_norm.var.sort_values(by="Moran's I", ascending=False)["Moran's I"][:10]

#%%
# new function for moran - test it against pysal

min_spots = 0.05
neighbors = 6

sc.settings.verbosity = 0
adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

adx = sc.pp.filter_genes(adata, min_cells=int(len(adata) * min_spots), copy=True)
adx.var_names_make_unique()  # moran dies so need some check later
if "log1p" not in adx.uns_keys():
    sc.pp.normalize_total(adx, inplace=True)
    sc.pp.log1p(adx)

ad = sc.pp.filter_genes(adata, min_cells=int(len(adata) * min_spots), copy=True)
ad.var_names_make_unique()  # moran dies so need some check later
if "log1p" not in ad.uns_keys():
    sc.pp.normalize_total(ad, inplace=True)
    sc.pp.log1p(ad)
ch.detect_svgs(ad, min_morans=0.08, min_spots=0.05)

points = adata.obsm['spatial']  # removed deep copy here, we don't need it probably
w = weights.KNN.from_array(points, k=neighbors)
w.transform = 'R'
w_sparse = w.sparse
w_sparse = w_sparse.tocsr()

X = adx[:, :].X.todense()

results = moran_sparse_matrix(X, w_sparse.data, w_sparse.indices, w_sparse.indptr)

plt.scatter(ad.var["Moran's I"], results)
plt.show()

#%%

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

ch.compute_svg_scores(adata)

plt.scatter(adata.var["n_cells_by_counts"], adata.var["morans"])
plt.show()


def select_svgs(adata, top_svg: int=1000, min_morans: float=None, min_spots: float=None,
                use_var: str='spatially_variable', svg_var: str='morans'):

    morans_vec = adata.var[svg_var]
    morans_vec = morans_vec.sort_values(ascending=False)

    # logic to exclude sparsely expressed genes
    if min_spots is not None:
        assert 0 < min_spots < 1

        if 'n_cells_by_counts' not in adata.var.columns:
            _, var_df = sc.pp.calculate_qc_metrics(adata, inplace=False)
            n_cells = var_df['n_cells_by_counts']
        else:
            n_cells = adata.var['n_cells_by_counts']

        threshold = int(len(adata) * min_spots)
        selected_genes = n_cells[n_cells > threshold].index
        morans_vec = morans_vec[selected_genes]

    # select thresholds
    if min_morans is None:
        adata.var[use_var] = [True if x in morans_vec[:top_svg].index else False for x in adata.var_names]
    elif len(morans_vec[:top_svg]) < len(morans_vec[morans_vec > min_morans]):
        adata.var[use_var] = [True if x in morans_vec[:top_svg].index else False for x in adata.var_names]
    else:
        morans_vec = morans_vec[morans_vec > min_morans]
        adata.var[use_var] = [True if x in morans_vec.index else False for x in adata.var_names]


select_svgs(adata, top_svg=1000)


def plot_svgs(adata, figsize=(3.5, 3.5), svg_var: str='morans', svg_bool: str='spatially_variable',
              text: bool=True):

    morans_df = adata.var[svg_var].sort_values(ascending=False)
    morans_df = morans_df.dropna()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(list(morans_df), linewidth=3, color='#8b33ff')

    if svg_bool in adata.var.columns:
        n_svg = len(adata.var[svg_bool][adata.var[svg_bool] == True])
        ax.axvline(x=n_svg, color='#ff9a4e', linestyle='--', linewidth=2)
        if text:
            ax.text(n_svg + len(morans_df) * 0.05,
                    morans_df.max() * 0.95,
                    f'n = {n_svg}',
                    color='black',
                    fontsize=10,
                    verticalalignment='top')

    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("Moran's I")
    ax.set_xlabel('Gene Rank')
    ax.set_title(f'SVG Rank Plot')
    plt.tight_layout()


plot_svgs(adata)
plt.show()
