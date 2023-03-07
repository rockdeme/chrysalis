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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys



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


import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_moransI(w_orig, y):
    # w = spatial weight (topological or actual distance)
    # y = actual value
    # y_hat = mean value
    # REF: https://github.com/yatshunlee/spatial_autocorrelation/blob/main/spatial_autocorrelation/moransI.py modified
    # wth some ChatGPT magic to remove the for loops

    if not isinstance(y, np.ndarray):
        raise TypeError("Passed array (feature) should be in numpy array (ndim = 1)")
    if y.shape[0] != w_orig.shape[0]:
        raise ValueError("Feature array is not the same shape of weight")
    if w_orig.shape[0] != w_orig.shape[1]:
        raise ValueError("Weight array should be in square shape")

    w = w_orig.copy()
    y_hat = np.mean(y)
    D = y - y_hat
    D_sq = (y - y_hat) ** 2
    N = y.shape[0]
    sum_W = np.sum(w)
    w *= D.reshape(-1, 1) * D.reshape(1, -1) * (w != 0)
    moransI = (np.sum(w) / sum(D_sq)) * (N / sum_W)
    return round(moransI, 8)


sw = np.array(w.sparse.todense())

for i in tqdm(range(100)):
    get_moransI(sw, np.array(gdf[c]))


asd_dict = {}
for c in tqdm(gene_list):
    moran = esda.moran.Moran(gdf[c], w, permutations=0)
    asd = get_moransI(sw, np.array(gdf[c]))
    asd_dict[asd] = moran.I

y = gdf[c].copy()


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




sc.pp.pca(adata)
sc.pl.pca_variance_ratio(adata)


for i in range(20):
    adata.obs[f'pca_{i}'] = adata.obsm['X_pca'][:, i]
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.spatial(adata, img_key='hires', color='pca_0', size=1.5, alpha=1,
              ax=ax, show=False, cmap=cmap)
fig.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def black_to_color(color):

    # Define the colors in the colormap
    colors = ["black", color]

    # Create a colormap object using the defined colors
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

    return cmap


def hsv_to_hex(h, s, v):
    # Convert the HSV values to RGB values
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # Convert the RGB values to a hex color code
    hex_code = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
    return hex_code


def get_rgb_from_colormap(cmap, vmin, vmax, value):

    # Normalize the value within the range [0, 1]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    value_normalized = norm(value)

    # Get the RGBA value from the colormap
    rgba = plt.get_cmap(cmap)(value_normalized)

    # Convert the RGBA value to RGB
    # color = tuple(np.array(rgba[:3]) * 255)
    color = np.array(rgba[:, :3])

    return color


def blend_colors(colors_1, colors_2, weight=0.5):

    # Ensure weight is between 0 and 1
    weight = max(0, min(1, weight))

    # Blend the colors using linear interpolation
    blended_colors = []
    for i in range(len(colors_1)):
        r = (1 - weight) * colors_1[i][0] + weight * colors_2[i][0]
        g = (1 - weight) * colors_1[i][1] + weight * colors_2[i][1]
        b = (1 - weight) * colors_1[i][2] + weight * colors_2[i][2]
        blended_colors.append((r, g, b))
    return blended_colors


def mip_colors(colors_1, colors_2):

    # Blend the colors using linear interpolation
    mip_color = []
    for i in range(len(colors_1)):
        r = max(colors_1[i][0], colors_2[i][0])
        g = max(colors_1[i][1], colors_2[i][1])
        b = max(colors_1[i][2], colors_2[i][2])
        mip_color.append((r, g, b))
    return mip_color


def chrysalis_plot(adata, pcs=8, hexcodes=None):
    def norm_weight(a, b):
        # return (b - a) / b
        return 0.5

    if hexcodes == None:
        hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
    else:
        assert len(hexcodes) >= pcs

    cmaps = []
    var_r = np.cumsum(adata.uns['pca']['variance_ratio'][:pcs])

    for pc in range(pcs):
        pc_cmap = black_to_color(hexcodes[pc])
        pc_rgb = get_rgb_from_colormap(pc_cmap,
                                       vmin=min(adata.obs[f'pca_{pc}']),
                                       vmax=max(adata.obs[f'pca_{pc}']),
                                       value=adata.obs[f'pca_{pc}'])
        cmaps.append(pc_rgb)

    cblend = mip_colors(cmaps[0], cmaps[1],
                        # weight=norm_weight(var_r[0], var_r[1])
                        )
    if len(cmaps) > 2:
        i = 2
        for cmap in cmaps[2:]:
            cblend = mip_colors(cblend, cmap,
                                # weight=norm_weight(var_r[i - 1], var_r[i])
                                )
            i += 1

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.axis('off')
    # ax[idx].set_xlim((0, 8500))
    # ax[idx].set_ylim((-8500, 0))
    row = adata.obsm['spatial'][:, 0]
    col = adata.obsm['spatial'][:, 1] * -1
    plt.scatter(row, col, s=25, marker="h", c=cblend)
    ax.set_aspect('equal')
    plt.show()


chrysalis_plot(adata)



for i in range(20):
    asd.obs[f'pca_{i}'] = asd.obsm['X_pca'][:, i]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.spatial(asd, img_key='hires', color='pca_0', size=1.5, alpha=1,
              ax=ax, show=False, cmap='rocket_r')
fig.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sc.pl.spatial(adata, img_key='hires', color=None, size=1.5, alpha=0,
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