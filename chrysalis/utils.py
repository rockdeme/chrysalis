import anndata
import colorsys
import numpy as np
import pandas as pd
from tqdm import tqdm
import archetypes as arch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .core import detect_svgs
from typing import List
from anndata import AnnData


def black_to_color(color):
    # define the colors in the colormap
    colors = ["black", color]
    # create a colormap object using the defined colors
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    return cmap


def color_to_color(first, last):
    # define the colors in the colormap
    colors = [first, last]
    # create a colormap object using the defined colors
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    return cmap


def hls_to_hex(h, l, s):
    # convert the HLS values to RGB values
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    # convert the RGB values to a hex color code
    hex_code = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), 76)
    return hex_code


def generate_random_colors(num_colors, hue_range=(0, 1), saturation=0.5, lightness=0.5, min_distance=0.05, seed=None):
    colors = []
    hue_list = []
    if seed:
        np.random.seed(seed)
    else:
        np.random.seed(42)
    while len(colors) < num_colors:
        # generate a random hue value within the specified range
        hue = np.random.uniform(hue_range[0], hue_range[1])

        # check if the hue is far enough away from the previous hue
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


def get_colors(adata, dim=8, seed=42):
    if dim > 8:
        hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5)
    else:
        hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
        if seed is None:
            np.random.seed(len(adata))
        else:
            np.random.seed(seed)
        np.random.shuffle(hexcodes)
    return hexcodes


def estimate_compartments(adata, n_pcs=20, range_archetypes=(3, 50), max_iter=10):

    if 'chr_X_pca' not in adata.obsm.keys():
        raise ValueError(".obsm['chr_X_pca'] cannot be found, run chrysalis_pca first.")

    entropy_arr = np.zeros((len(range(range_archetypes[0], range_archetypes[1])), len(adata)))
    rss_dict = {}
    i = 0
    for a in tqdm(range(range_archetypes[0], range_archetypes[1]), desc='Fitting models'):
        model = arch.AA(n_archetypes=a, n_init=3, max_iter=max_iter, tol=0.001, random_state=42)
        model.fit(adata.obsm['chr_X_pca'][:, :n_pcs])
        rss_dict[a] = model.rss_

        entropy_arr[i, :] = entropy(model.alphas_.T)
        i += 1

    adata.obsm['entropy'] = entropy_arr.T

    if 'chr_aa' not in adata.uns.keys():
        adata.uns['chr_aa'] = {'RSSs': rss_dict}
    else:
        adata.uns['chr_aa']['RSSs'] = rss_dict


def get_compartment_df(adata: AnnData, weights: bool=True):
    """
    Get spatially variable gene weights/expression values as a pandas DataFrame.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.:
    :param weights: If False, return expression values instead of weights.
    :return: Pandas DataFrame.

    """

    # SVG expression for each compartment
    exp_array = np.asarray(adata[:, adata.var['spatially_variable'] == True].X.todense())
    exp_array = np.mean(exp_array, axis=0)
    exp_aa = adata.uns['chr_aa']['loadings']
    if not weights:
        exp_aa += exp_array

    df = pd.DataFrame(data=exp_aa, columns=adata.uns['chr_pca']['features'],
                      index=[f'compartment_{x}' for x in (range(len(exp_aa)))]).T
    return df


def integrate_adatas(adatas: List[AnnData], sample_names: List[str]=None, **kwargs):
    """
    Integrate multiple samples stored in AnnData objects.

    `.var['spatially_variable']` will be outer joined.

    :param adatas: List of AnnData objects.
    :param sample_names: List of sample names. If not defined, a list of integers [0, 1, ...] will be used instead.
    :param kwargs: Keyword arguments for `chrysalis.detect_svgs`.
    :return: Integrated AnnData object. Sample IDs are stored in `.obs['sample'].

    """

    if sample_names is None:
        sample_names = np.arange(len(adatas))
    assert len(adatas) == len(sample_names)

    adatas_dict = {}
    for ad, name in zip(adatas, sample_names):
        ad.obs['sample'] = name
        if 'gene_ids' in ad.var.columns:
            ad.var['gene_symbols'] = ad.var_names
            ad.var_names = ad.var['gene_ids']

        detect_svgs(ad, **kwargs)

        ad.var[f'spatially_variable_{name}'] = ad.var['spatially_variable']
        ad.var[f"Moran's I_{name}"] = ad.var["Moran's I"]

        adatas_dict[name] = ad

    # concat samples
    adata = anndata.concat(adatas_dict, index_unique='-', uns_merge='unique', merge='unique')
    adata.obs['sample'] = adata.obs['sample'].astype('category')
    # get SVGs for all samples
    svg_columns = [c for c in adata.var.columns if 'spatially_variable' in c]
    svg_list = [list(adata.var[c][adata.var[c] == True].index) for c in svg_columns]
    # union of SVGs
    spatially_variable = list(set().union(*svg_list))
    adata.var['spatially_variable'] = [True if x in spatially_variable else False for x in adata.var_names]

    return adata


def morans_i(w_orig, y):
    """
    Not in use.

    REF: https://github.com/yatshunlee/spatial_autocorrelation/blob/main/spatial_autocorrelation/moransI.py modified.

    :param w_orig:
    :param y:
    :return:
    """

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
