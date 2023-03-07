import scanpy as sc
from glob import glob
import matplotlib.pyplot as plt
import anndata
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mtick
import numpy as np
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point
from pysal.lib import weights
from pysal.explore import esda
import matplotlib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering


def morans_plot(gdf, label, ax=None):
    w = weights.KNN.from_dataframe(gdf, k=6)
    w.transform = 'R'
    moran = esda.moran.Moran(gdf[label], w)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.regplot(x=f'{label}_std', y=f'{label}_lag_std', ci=None, data=gdf, line_kws={'color': 'r'},
                scatter_kws={'alpha':0.1, 'marker': '.', 's': 5, 'color': 'grey'}, ax=ax)
    ax.axvline(0, c='k', alpha=0.5)
    ax.axhline(0, c='k', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Moran Plot - Moran's I {round(moran.I, ndigits=3)} p value: {round(moran.p_sim, ndigits=4)}")
    plt.show()


def local_morans_plot(gdf, title=None, ax=None):
    cmap = matplotlib.colors.ListedColormap(((0.839, 0.152, 0.156),
                                             (1.000, 0.596, 0.588),
                                             (0.682, 0.780, 0.909),
                                             (0.121, 0.466, 0.705),))
    if ax:
        gdf.plot(markersize=6, column='lisa_q', cmap=cmap, ax=ax)
        color_id = {k: v for k, v in zip(gdf['lisa_q'].cat.categories, cmap.colors)}
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_id.values()]
        ax.legend(markers, color_id.keys(), numpoints=1)
        if title:
            ax.set_title(title)
    else:
        gdf.plot(markersize=6, column='lisa_q', cmap=cmap)
        color_id = {k: v for k, v in zip(gdf['lisa_q'].cat.categories, cmap.colors)}
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_id.values()]
        plt.legend(markers, color_id.keys(), numpoints=1)
        if title:
            plt.title(title)
        plt.show()


def morans(adata, label):
    gdf = gpd.GeoDataFrame(data={'spots': [Point(x, y) for x, y in zip(adata.obsm['spatial'][:, 0],
                                                                        adata.obsm['spatial'][:, 1] * -1)]},
                           index=adata.obs_names)
    gdf.geometry = gdf['spots']
    gdf[label] = adata.to_df()[label]
    w = weights.KNN.from_dataframe(gdf, k=6)
    w.transform = 'R'
    gdf[f'{label}_lag'] = weights.spatial_lag.lag_spatial(w, gdf[label])
    gdf[f'{label}_std'] = gdf[label] - gdf[label].mean()
    gdf[f'{label}_lag_std'] = (gdf[f'{label}_lag'] - gdf[f'{label}_lag'].mean())

    lisa = esda.moran.Moran_Local(list(gdf[label]), w, transformation="r", permutations=99)
    hl = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    gdf['lisa_Is'] = lisa.Is
    gdf['lisa_q'] = lisa.q
    gdf['lisa_q'] = [hl[x] for x in gdf['lisa_q']]
    gdf['lisa_q'] = gdf['lisa_q'].astype('category')
    return gdf






adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
sc.pp.filter_genes(adata, min_cells=1000)
adata.var_names_make_unique()  # moran dies so need some check later
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

gene_matrix = adata.to_df()
gene_list = list(gene_matrix.columns)
gdf = gpd.GeoDataFrame(gene_matrix)
gdf['spots'] = [Point(x, y) for x, y in zip(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1] * -1)]
gdf.geometry = gdf['spots']

w = weights.KNN.from_dataframe(gdf, k=6)
w.transform = 'R'
moran_dict = {}
for c in tqdm(gene_list):
    moran = esda.moran.Moran(gdf[c], w, permutations=0)
    moran_dict[c] = moran.I

moran_df = pd.DataFrame(data=moran_dict.values(), index=moran_dict.keys())
moran_df = moran_df.sort_values(ascending=False, by=0)


sns.histplot(moran_df)
sns.rugplot(x=[moran_df.iloc[150][0]], c='red')
plt.show()

adata.var['highly_variable'] = [True if x in moran_df[:150].index else False for x in adata.var_names]

hv = adata[:, adata.var['highly_variable'] == True].to_df().corr('pearson')

sns.clustermap(hv)
plt.show()

clustering = AgglomerativeClustering(n_clusters=10).fit(hv)
cluster_df = pd.DataFrame(data={'cluster': clustering.labels_}, index=hv.index)
cluster_df['cluster'] = cluster_df['cluster'].astype('category')
for idx, cl in enumerate(cluster_df['cluster'].cat.categories):
    cl_df = cluster_df[cluster_df['cluster'] == cl]
    gene_df = adata.to_df()[cl_df.index].mean(axis=1)
    adata.obs[f'unit_{idx}'] = gene_df

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.spatial(adata, img_key='hires', color='unit_3', size=1.5, alpha=1,
              ax=ax, show=False, cmap='rocket_r')
fig.tight_layout()
plt.show()


asd = sc.pp.pca(adata, copy=True)
asd.uns['pca']['variance_ratio']
asd.obsm['X_pca'].shape

for i in range(20):
    asd.obs[f'pca_{i}'] = asd.obsm['X_pca'][:, i]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.spatial(asd, img_key='hires', color='pca_0', size=1.5, alpha=1,
              ax=ax, show=False, cmap='rocket_r')
fig.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.spatial(asd, img_key='hires', color=None, size=1.5, alpha=0,
              ax=ax, show=False, cmap='rocket_r')
fig.tight_layout()
plt.show()
fig, ax = plt.subplots(2, 4, figsize=(20, 4 * 2))
ax = ax.flatten()
for idx, axis in enumerate(ax):
    sc.pl.spatial(asd, img_key='hires', color=f'pca_{idx}', size=1.5, alpha=1,
                  ax=axis, show=False, cmap='rocket_r', alpha_img=0)
fig.tight_layout()
plt.show()

sc.pp.highly_variable_genes()


sc.pl.spatial(adata, img_key="hires", color=list(moran_df[0].index[-5:]), alpha_img=0, alpha=1, s=13)
plt.show()




fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.spatial(asd, img_key='hires', color='CENPU', size=1.5, alpha=1,
              ax=ax, show=False, cmap='rocket_r')
fig.tight_layout()
plt.show()
