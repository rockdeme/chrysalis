import colorsys
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import geopandas as gpd
import archetypes as arch
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


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


def black_to_color(color):
    # define the colors in the colormap
    colors = ["black", color]

    # create a colormap object using the defined colors
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

    return cmap


def hls_to_hex(h, l, s):
    # convert the HSV values to RGB values
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    # convert the RGB values to a hex color code
    hex_code = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), 76)
    return hex_code


def generate_random_colors(num_colors, hue_range=(0, 1), saturation=0.5, lightness=0.5, min_distance=0.2):
    colors = []
    hue_list = []

    while len(colors) < num_colors:
        # Generate a random hue value within the specified range
        hue = np.random.uniform(hue_range[0], hue_range[1])

        # Check if the hue is far enough away from the previous hue
        if len(hue_list) == 0 or all(abs(hue - h) > min_distance for h in hue_list):
            hue_list.append(hue)
            saturation = saturation
            lightness = lightness
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_code = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            colors.append(hex_code)

    return colors

def get_rgb_from_colormap(cmap, vmin, vmax, value):
    # normalize the value within the range [0, 1]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    value_normalized = norm(value)

    # get the RGBA value from the colormap
    rgba = plt.get_cmap(cmap)(value_normalized)

    # convert the RGBA value to RGB
    # color = tuple(np.array(rgba[:3]) * 255)
    color = np.array(rgba[:, :3])

    return color


def blend_colors(colors_1, colors_2, weight=0.5):
    # ensure weight is between 0 and 1
    weight = max(0, min(1, weight))

    # blend the colors using linear interpolation
    blended_colors = []
    for i in range(len(colors_1)):
        r = (1 - weight) * colors_1[i][0] + weight * colors_2[i][0]
        g = (1 - weight) * colors_1[i][1] + weight * colors_2[i][1]
        b = (1 - weight) * colors_1[i][2] + weight * colors_2[i][2]
        blended_colors.append((r, g, b))
    return blended_colors


def mip_colors(colors_1, colors_2):
    # blend the colors using linear interpolation
    mip_color = []
    for i in range(len(colors_1)):
        r = max(colors_1[i][0], colors_2[i][0])
        g = max(colors_1[i][1], colors_2[i][1])
        b = max(colors_1[i][2], colors_2[i][2])
        mip_color.append((r, g, b))
    return mip_color

def chrysalis_plot_old(adata, pcs=8, hexcodes=None, seed=None, vis='mip_colors'):

    def norm_weight(a, b):
        # for weighting PCs if we want to use blend_colors
        return (b - a) / b

    # define PC colors
    if hexcodes is None:
        hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
        if seed is None:
            np.random.seed(len(adata))
        else:
            np.random.seed(seed)
        np.random.shuffle(hexcodes)
    else:
        assert len(hexcodes) >= pcs


    # define colormaps
    cmaps = []
    for pc in range(pcs):
        pc_cmap = black_to_color(hexcodes[pc])
        pc_rgb = get_rgb_from_colormap(pc_cmap,
                                       vmin=min(adata.obs[f'pca_{pc}']),
                                       vmax=max(adata.obs[f'pca_{pc}']),
                                       value=adata.obs[f'pca_{pc}'])
        cmaps.append(pc_rgb)

    # blend colormaps
    if vis == 'mip_colors':
        cblend = mip_colors(cmaps[0], cmaps[1],)
        if len(cmaps) > 2:
            i = 2
            for cmap in cmaps[2:]:
                cblend = mip_colors(cblend, cmap,)
                i += 1
    elif vis == 'blend_colors':
        var_r = np.cumsum(adata.uns['pca']['variance_ratio'][:pcs])  # get variance ratios to normalize
        cblend = blend_colors(cmaps[0], cmaps[1], weight=norm_weight(var_r[0], var_r[1]))
        if len(cmaps) > 2:
            i = 2
            for cmap in cmaps[2:]:
                cblend = blend_colors(cblend, cmap, weight=norm_weight(var_r[i - 1], var_r[i]))
                i += 1
    else:
        raise Exception('vis should be either mip_colors or blend colors')

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.axis('off')
    # ax[idx].set_xlim((0, 8500))
    # ax[idx].set_ylim((-8500, 0))
    row = adata.obsm['spatial'][:, 0]
    col = adata.obsm['spatial'][:, 1] * -1
    plt.scatter(row, col, s=25, marker="h", c=cblend)
    ax.set_aspect('equal')


def chrysalis_calculate_old(adata):
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
    #  moran.by_col(gdf,gene_list, w=w, permutations=0) this doesn't seem to be faster
    for c in tqdm(gene_list):
        moran = esda.moran.Moran(gdf[c], w, permutations=0)
        moran_dict[c] = moran.I

    moran_df = pd.DataFrame(data=moran_dict.values(), index=moran_dict.keys(), columns=["Moran's I"])
    moran_df = moran_df.sort_values(ascending=False, by="Moran's I")
    adata.var['highly_variable'] = [True if x in moran_df[:1000].index else False for x in adata.var_names]
    adata.var["Moran's I"] = moran_df["Moran's I"]

    sc.pp.pca(adata)

    for i in range(20):
        adata.obs[f'pca_{i}'] = adata.obsm['X_pca'][:, i]

    # archetype analysis
    model = arch.AA(n_archetypes=8, n_init=3, max_iter=200, tol=0.001, random_state=42)
    model.fit(adata.obsm['X_pca'][:, :7])

    for i in range(model.alphas_.shape[1]):
        adata.obs[f'aa_{i}'] = model.alphas_[:, i]


def plot_loadings(adata):
    hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']

    np.random.seed(len(adata))
    np.random.shuffle(hexcodes)

    loadings = pd.DataFrame(adata.varm['PCs'][:, :20], index=adata.var_names)
    sl = loadings[[0]].sort_values(ascending=False, by=0)[:10]

    fig, ax = plt.subplots(2, 4, figsize=(3 * 4, 4 * 2))
    ax = ax.flatten()
    for i in range(8):
        sl = loadings[[i]].sort_values(ascending=False, by=i)[:10]
        ax[i].axis('on')
        ax[i].grid(axis='x')
        ax[i].set_axisbelow(True)
        ax[i].barh(list(sl.index)[::-1], list(sl[i].values)[::-1], color=hexcodes[i])
        ax[i].set_xlabel('Loading')
        ax[i].set_title(f'PC {i}')
    plt.tight_layout()
    plt.show()


def chrysalis_plot_aa(adata, pcs=8, hexcodes=None, seed=None, vis='mip_colors'):

    def norm_weight(a, b):
        # for weighting PCs if we want to use blend_colors
        return (b - a) / b

    # define PC colors
    if hexcodes is None:
        hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']

        if pcs > 8:
            if seed is None:
                np.random.seed(len(adata))
            else:
                np.random.seed(seed)
            hexcodes = generate_random_colors(pcs, hue_range=(0.0, 1.0), min_distance=0.05)

        if seed is None:
            np.random.seed(len(adata))
        else:
            np.random.seed(seed)
        np.random.shuffle(hexcodes)
    else:
        assert len(hexcodes) >= pcs


    # define colormaps
    cmaps = []
    for pc in range(pcs):
        pc_cmap = black_to_color(hexcodes[pc])
        pc_rgb = get_rgb_from_colormap(pc_cmap,
                                       vmin=min(adata.obs[f'aa_{pc}']),
                                       vmax=max(adata.obs[f'aa_{pc}']),
                                       value=adata.obs[f'aa_{pc}'])
        cmaps.append(pc_rgb)

    # blend colormaps
    if vis == 'mip_colors':
        cblend = mip_colors(cmaps[0], cmaps[1],)
        if len(cmaps) > 2:
            i = 2
            for cmap in cmaps[2:]:
                cblend = mip_colors(cblend, cmap,)
                i += 1
    elif vis == 'blend_colors':
        var_r = np.cumsum(adata.uns['pca']['variance_ratio'][:pcs])  # get variance ratios to normalize
        cblend = blend_colors(cmaps[0], cmaps[1], weight=norm_weight(var_r[0], var_r[1]))
        if len(cmaps) > 2:
            i = 2
            for cmap in cmaps[2:]:
                cblend = blend_colors(cblend, cmap, weight=norm_weight(var_r[i - 1], var_r[i]))
                i += 1
    else:
        raise Exception("Vis should be either 'mip_colors' or 'blend colors'")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.axis('off')
    row = adata.obsm['spatial'][:, 0]
    col = adata.obsm['spatial'][:, 1] * -1
    ax.set_xlim((np.min(row) * 0.9, np.max(row) * 1.1))
    ax.set_ylim((np.min(col) * 1.1, np.max(col) * 0.9))
    ax.set_aspect('equal')

    # get the physical length of the x and y axes
    x_length = np.diff(ax.get_xlim())[0] * fig.dpi * fig.get_size_inches()[0]
    y_length = np.diff(ax.get_ylim())[0] * fig.dpi * fig.get_size_inches()[1]

    size = np.sqrt(x_length * y_length) * 0.000005

    plt.scatter(row, col, s=size, marker="h", c=cblend)