import colorsys
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import archetypes as arch
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


def get_moransI(w_orig, y):
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


def chrysalis_calculate(adata, min_spots=1000, top_svg=1000, n_archetypes=8):
    """
    Calculates spatially variable genes and embeddings for visualization.

    :param adata: 10X Visium anndata matrix created with scanpy.
    :param min_spots: Discard genes expressed in less capture spots than this threshold. Speeds up spatially variable
    gene computation but can be set lower if sample area is small.
    :param top_svg: Number of spatially variable genes to be considered for PCA.
    :param n_archetypes: Number of inferred archetypes, best leave it at 8, no significant gain by trying to visualize
    more.
    :return: Directly annotates the data matrix: adata.obsm['chr_X_pca'] and adata.obsm['chr_aa'].
    """
    sc.settings.verbosity = 0
    ad = sc.pp.filter_genes(adata, min_cells=min_spots, copy=True)
    ad.var_names_make_unique()  # moran dies so need some check later
    if "log1p" not in adata.uns_keys():
        sc.pp.normalize_total(ad, inplace=True)
        sc.pp.log1p(ad)

    gene_matrix = ad.to_df()

    points = adata.obsm['spatial'].copy()
    points[:, 1] = points[:, 1] * -1

    w = weights.KNN.from_array(points, k=6)
    w.transform = 'R'
    moran_dict = {}

    for c in tqdm(ad.var_names):
        moran = esda.moran.Moran(gene_matrix[c], w, permutations=0)
        moran_dict[c] = moran.I

    moran_df = pd.DataFrame(data=moran_dict.values(), index=moran_dict.keys(), columns=["Moran's I"])
    moran_df = moran_df.sort_values(ascending=False, by="Moran's I")
    adata.var['spatially_variable'] = [True if x in moran_df[:top_svg].index else False for x in adata.var_names]
    ad.var['spatially_variable'] = [True if x in moran_df[:top_svg].index else False for x in ad.var_names]
    adata.var["Moran's I"] = moran_df["Moran's I"]

    pcs = np.asarray(ad[:, ad.var['spatially_variable'] == True].X.todense())
    pca = PCA(n_components=50, svd_solver='arpack', random_state=0)
    adata.obsm['chr_X_pca'] = pca.fit_transform(pcs)
    if 'chr_pca' not in adata.uns.keys():
        adata.uns['chr_pca'] = {'variance_ratio': pca.explained_variance_ratio_}
    else:
        adata.uns['chr_pca']['variance_ratio'] = pca.explained_variance_ratio_

    model = arch.AA(n_archetypes=n_archetypes, n_init=3, max_iter=200, tol=0.001, random_state=42)
    model.fit(adata.obsm['chr_X_pca'][:, :n_archetypes-1])
    adata.obsm[f'chr_aa'] = model.alphas_


def chrysalis_plot(adata, dim=8, hexcodes=None, seed=None, mode='aa'):
    """
    Visualizes embeddings calculated with chrysalis.calculate.
    :param adata: 10X Visium anndata matrix created with scanpy.
    :param dim: Number of components to visualize.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param mode: Components to visualize: 'aa' - archetype analysis, 'pca' - PCA
    :return:
    """

    # define PC colors
    if hexcodes is None:
        hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
        if seed is None:
            np.random.seed(len(adata))
        else:
            np.random.seed(seed)
        np.random.shuffle(hexcodes)
    else:
        assert len(hexcodes) >= dim
    # define colormaps
    cmaps = []

    if mode == 'aa':
        for d in range(dim):
            pc_cmap = black_to_color(hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_aa'][:, d]),
                                           vmax=max(adata.obsm['chr_aa'][:, d]),
                                           value=adata.obsm['chr_aa'][:, d])
            cmaps.append(pc_rgb)

    elif mode == 'pca':
        for d in range(dim):
            pc_cmap = black_to_color(hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_X_pca'][:, d]),
                                           vmax=max(adata.obsm['chr_X_pca'][:, d]),
                                           value=adata.obsm['chr_X_pca'][:, d])
            cmaps.append(pc_rgb)
    else:
        raise Exception

    # mip colormaps
    cblend = mip_colors(cmaps[0], cmaps[1],)
    if len(cmaps) > 2:
        i = 2
        for cmap in cmaps[2:]:
            cblend = mip_colors(cblend, cmap,)
            i += 1

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.axis('off')
    row = adata.obsm['spatial'][:, 0]
    col = adata.obsm['spatial'][:, 1] * -1
    ax.set_xlim((np.min(row) * 0.9, np.max(row) * 1.1))
    ax.set_ylim((np.min(col) * 1.1, np.max(col) * 0.9))
    ax.set_aspect('equal')

    distances = cdist(np.column_stack((row, col)), np.column_stack((row, col)))
    np.fill_diagonal(distances, np.inf)
    min_distance = np.min(distances)

    # get the physical length of the x and y axes
    ax_len = np.diff(np.array(ax.get_position())[:, 0]) * fig.get_size_inches()[0]
    size_const = ax_len / np.diff(ax.get_xlim())[0] * min_distance * 72
    size = size_const ** 2 * 0.95
    plt.scatter(row, col, s=size, marker="h", c=cblend)
