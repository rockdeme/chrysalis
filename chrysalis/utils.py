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
        hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5, seed=seed,
                                          saturation=0.65, lightness=0.60)
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


def integrate_adatas(adatas: List[AnnData], sample_names: List[str]=None, calculate_svgs: bool=False,
                     sample_col: str='sample', **kwargs):
    """
    Integrate multiple samples stored in AnnData objects.

    If ENSEMBL IDs are present in the`.var['gene_ids']` column, that will be used instead of gene symbols.
    `.var['spatially_variable']` will be outer joined.

    :param adatas: List of AnnData objects.
    :param sample_names: List of sample names. If not defined, a list of integers [0, 1, ...] will be used instead.
    :param calculate_svgs: If True, the function also runs `chrysalis.detect_svgs` for every sample.
    :param sample_col: `.obs` column name to store the sample labels.
    :param kwargs: Keyword arguments for `chrysalis.detect_svgs`.
    :return:
        Integrated AnnData object. Sample IDs are stored in `.obs[sample_col]. `.var['spatially_variable']` contains
        the union of`.var['spatially_variable']` from the input AnnData objects. Sample-wise SVG data is stored in
        `.varm['spatially_variable']` and Moran's I is stored in `.varm["Moran's I"]`.

    """

    if sample_names is None:
        sample_names = np.arange(len(adatas))
    assert len(adatas) == len(sample_names)

    adatas_dict = {}
    gene_symbol_dict = {}
    for ad, name in zip(adatas, sample_names):

        # replace .uns['spatial'] with the specified sample name
        if 'spatial' in ad.uns.keys():
            assert len(ad.uns['spatial'].keys()) == 1
            curr_key = list(ad.uns['spatial'].keys())[0]
            ad.uns['spatial'][name] = ad.uns['spatial'][curr_key]
            if name != curr_key:
                del ad.uns['spatial'][curr_key]

        # check if column is already used
        if sample_col not in ad.obs.columns:
            ad.obs[sample_col] = name
        else:
            raise Exception('sample_id_col is already present in adata.obs, specify another column.')

        if 'gene_symbols' not in ad.var.columns:
            ad.var['gene_symbols'] = ad.var_names

        if 'gene_ids' in ad.var.columns:
            ad.var_names = ad.var['gene_ids']

        # check if SVGs are already present
        if 'spatially_variable' not in ad.var.columns:
            if calculate_svgs:
                detect_svgs(ad, **kwargs)
            else:
                raise Exception('spatially_variable column is not found in adata.var. Run `chrysalis.detect_svgs` '
                                'first or set the calculate_svgs argument to True.')

        ad.var[f'spatially_variable_{name}'] = ad.var['spatially_variable']
        ad.var[f"Moran's I_{name}"] = ad.var["Moran's I"]

        adatas_dict[name] = ad

    # concat samples
    adata = anndata.concat(adatas_dict, index_unique='-', uns_merge='unique', merge='first')
    adata.obs[sample_col] = adata.obs[sample_col].astype('category')
    # get SVGs for all samples
    svg_columns = [c for c in adata.var.columns if 'spatially_variable' in c]
    svg_list = [list(adata.var[c][adata.var[c] == True].index) for c in svg_columns]

    # union of SVGs
    spatially_variable = list(set().union(*svg_list))
    adata.var['spatially_variable'] = [True if x in spatially_variable else False for x in adata.var_names]

    # save sample-wise spatially_variable and Morans's I columns
    sv_cols = [x for x in adata.var.columns if 'spatially_variable_' in x]
    adata.varm['spatially_variable'] = adata.var[sv_cols].copy()
    if 'gene_ids' in adata.var.columns:
        adata.varm['spatially_variable']['gene_ids'] = adata.var['gene_ids']
    if 'gene_symbols' in adata.var.columns:
        adata.varm['spatially_variable']['gene_symbols'] = adata.var['gene_symbols']
    adata.var = adata.var.drop(columns=sv_cols)

    mi_cols = [x for x in adata.var.columns if "Moran's I_" in x]
    adata.varm["Moran's I"] = adata.var[mi_cols].copy()
    if 'gene_ids' in adata.var.columns:
        adata.varm["Moran's I"]['gene_ids'] = adata.var['gene_ids']
    if 'gene_symbols' in adata.var.columns:
        adata.varm["Moran's I"]['gene_symbols'] = adata.var['gene_symbols']
    adata.var = adata.var.drop(columns=mi_cols)

    return adata


def harmony_integration(adata, covariates, input_matrix='chr_X_pca', corrected_matrix=None,
                        random_state=42, **harmony_kw):
    """
    Integrate data using `harmonypy`, the Python implementation of the R package Harmony.

    Harmony integration is done on the PCA matrix, therefore `chrysalis.pca` must be run before this function.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param covariates: String or list of strings containing the covariate columns to integrate over.
    :param input_matrix: Input PCA matrix, by default 'chr_X_pca' is used in `.obsm`.
    :param corrected_matrix: If `corrected_matrix` is defined, a new `.obsm` matrix will be created for the integrated
        results instead of overwriting the `input_matrix`.
    :param harmony_kw: `harmonypy.run_harmony()` keyword arguments.
    :return:
        Replaces `.obsm[input_matrix]` with the corrected one, or saves the new matrix as a new .`.obsm` matrix
        specified with `corrected_matrix`.

    """

    try:
        import harmonypy as hm
    except ImportError:
        raise ImportError("Please install harmonypy: `pip install harmonypy`.")

    data_matrix = adata.obsm[input_matrix]
    metadata = adata.obs

    ho = hm.run_harmony(data_matrix, metadata, covariates, random_state, **harmony_kw)

    adjusted_matrix = np.transpose(ho.Z_corr)

    if corrected_matrix is None:
        adata.obsm[input_matrix] = adjusted_matrix
    else:
        adata.obsm[corrected_matrix] = adjusted_matrix


def get_color_vector(adata: AnnData, dim: int=8, hexcodes: List[str]=None, seed: int=None,
                     selected_comp='all'):
    # define compartment colors
    # default colormap with 8 colors
    hexcodes = get_hexcodes(hexcodes, dim, seed, len(adata))

    if selected_comp == 'all':
        # define colormaps
        cmaps = []
        for d in range(dim):
            pc_cmap = black_to_color(hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_aa'][:, d]),
                                           vmax=max(adata.obsm['chr_aa'][:, d]),
                                           value=adata.obsm['chr_aa'][:, d])
            cmaps.append(pc_rgb)

        # mip colormaps
        cblend = mip_colors(cmaps[0], cmaps[1],)
        if len(cmaps) > 2:
            i = 2
            for cmap in cmaps[2:]:
                cblend = mip_colors(cblend, cmap,)
                i += 1
    # specific compartment
    else:
        color_first = '#2e2e2e'
        pc_cmap = color_to_color(color_first, hexcodes[selected_comp])
        pc_rgb = get_rgb_from_colormap(pc_cmap,
                                       vmin=min(adata.obsm['chr_aa'][:, selected_comp]),
                                       vmax=max(adata.obsm['chr_aa'][:, selected_comp]),
                                       value=adata.obsm['chr_aa'][:, selected_comp])
        cblend = pc_rgb
    return cblend


def get_hexcodes(hexcodes, dim, seed, adata_len):
    if hexcodes is None:
        if dim > 8:
            hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5, seed=seed,
                                              saturation=0.65, lightness=0.60)
        else:
            hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
            if seed is None:
                np.random.seed(adata_len)
            else:
                np.random.seed(seed)
            np.random.shuffle(hexcodes)
    else:
        assert len(hexcodes) >= dim
    return hexcodes
