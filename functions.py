import math
import anndata
import colorsys
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm import tqdm
import archetypes as arch
from pysal.lib import weights
from pysal.explore import esda
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from skimage.color import rgb2lab, deltaE_ciede2000


def morans_i(w_orig, y):
    # REF: https://github.com/yatshunlee/spatial_autocorrelation/blob/main/spatial_autocorrelation/moransI.py modified

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


def chrysalis_calculate(adata, min_spots=0.1, top_svg=1000, min_morans=0.05, n_archetypes=8):
    """
    Calculates spatially variable genes and embeddings for visualization.

    :param adata: 10X Visium anndata matrix created with scanpy.
    :param min_spots: Discard genes expressed in less capture spots than this fraction (0<v<1). Speeds up spatially variable
    gene computation but can be set lower if sample area is small.
    :param top_svg: Number of spatially variable genes to be considered for PCA.
    :param n_archetypes: Number of inferred archetypes, best leave it at 8, no significant gain by trying to visualize
    more.
    :return: Directly annotates the data matrix: adata.obsm['chr_X_pca'] and adata.obsm['chr_aa'].
    """
    assert 0 < min_spots < 1
    assert -1 < min_morans < 1

    sc.settings.verbosity = 0
    ad = sc.pp.filter_genes(adata, min_cells=int(len(adata) * min_spots), copy=True)
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

    for c in tqdm(ad.var_names, desc='Calculating SVGs'):
        moran = esda.moran.Moran(gene_matrix[c], w, permutations=0)
        moran_dict[c] = moran.I

    moran_df = pd.DataFrame(data=moran_dict.values(), index=moran_dict.keys(), columns=["Moran's I"])
    moran_df = moran_df.sort_values(ascending=False, by="Moran's I")
    adata.var["Moran's I"] = moran_df["Moran's I"]
    # select threshold to choose from
    if len(moran_df[:top_svg]) < len(moran_df[moran_df["Moran's I"] > min_morans]):
        adata.var['spatially_variable'] = [True if x in moran_df[:top_svg].index else False for x in adata.var_names]
        ad.var['spatially_variable'] = [True if x in moran_df[:top_svg].index else False for x in ad.var_names]
    else:
        moran_df = moran_df[moran_df["Moran's I"] > min_morans]
        adata.var['spatially_variable'] = [True if x in moran_df.index else False for x in adata.var_names]
        ad.var['spatially_variable'] = [True if x in moran_df.index else False for x in ad.var_names]

    pcs = np.asarray(ad[:, ad.var['spatially_variable'] == True].X.todense())
    pca = PCA(n_components=50, svd_solver='arpack', random_state=0)
    adata.obsm['chr_X_pca'] = pca.fit_transform(pcs)

    if 'chr_pca' not in adata.uns.keys():
        adata.uns['chr_pca'] = {'variance_ratio': pca.explained_variance_ratio_,
                                'loadings': pca.components_,
                                'features': list(ad[:, ad.var['spatially_variable'] == True].var_names)}
    else:
        adata.uns['chr_pca']['variance_ratio'] = pca.explained_variance_ratio_
        adata.uns['chr_pca']['loadings'] = pca.components_
        adata.uns['chr_pca']['features'] = list(ad[:, ad.var['spatially_variable'] == True].var_names)

    model = arch.AA(n_archetypes=n_archetypes, n_init=3, max_iter=200, tol=0.001, random_state=42)
    model.fit(adata.obsm['chr_X_pca'][:, :n_archetypes-1])
    adata.obsm[f'chr_aa'] = model.alphas_

    # get the mean of the original feature matrix and add it to the multiplied archetypes with the PCA loading matrix
    # aa_loadings = np.mean(pcs, axis=0) + np.dot(model.archetypes_.T, pca.components_[:n_archetypes, :])
    aa_loadings = np.dot(model.archetypes_, adata.uns['chr_pca']['loadings'][:n_archetypes-1, :])

    if 'chr_aa' not in adata.uns.keys():
        adata.uns['chr_aa'] = {'archetypes': model.archetypes_,
                               'alphas': model.alphas_,
                               'loadings': aa_loadings,}
    else:
        adata.uns['chr_aa']['archetypes'] = model.archetypes_
        adata.uns['chr_aa']['alphas'] = model.alphas_
        adata.uns['chr_aa']['loadings'] = aa_loadings


def chrysalis_plot(adata, dim=8, hexcodes=None, seed=None, mode='aa', sample_id=None, spot_size=1.05, marker='h',
                   figsize=(5, 5), **scr_kw):
    """
    Visualizes embeddings calculated with chrysalis.calculate.
    :param adata: 10X Visium anndata matrix created with scanpy.
    :param dim: Number of components to visualize.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param mode: Components to visualize: 'aa' - archetype analysis, 'pca' - PCA
    :return:
    """

    # define compartment colors
    # default colormap with 8 colors
    if hexcodes is None:
        if dim > 8:
            hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5, seed=seed)
        else:
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

    if 'sample' in adata.obs.columns and sample_id is None:
        if sample_id not in adata.obs['sample'].cat.categories:
            raise ValueError("Invalid sample_id. Check categories in .obs['sample']")
        raise ValueError("Integrated dataset. Cannot proceed without a specified sample_id")

    if sample_id is not None:
        cblend = [x for x, b in zip(cblend, list(adata.obs['sample'] == sample_id)) if b == True]
        adata = adata[adata.obs['sample'] == sample_id]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('off')
    row = adata.obsm['spatial'][:, 0]
    col = adata.obsm['spatial'][:, 1] * -1
    row_range = np.ptp(row)
    col_range = np.ptp(col)
    ax.set_xlim((np.min(row) - 0.1 * row_range, np.max(row) + 0.1 * row_range))
    ax.set_ylim((np.min(col) - 0.1 * col_range, np.max(col) + 0.1 * col_range))
    ax.set_aspect('equal')

    # takes long time to compute the pairwise distance matrix for stereo-seq or slide-seq samples, so by looking at
    # only 5000 spots is a good enough approximation
    if len(row) < 5000:
        distances = cdist(np.column_stack((row, col)), np.column_stack((row, col)))
    else:
        distances = cdist(np.column_stack((row[:5000], col[:5000])), np.column_stack((row[:5000], col[:5000])))

    np.fill_diagonal(distances, np.inf)
    min_distance = np.min(distances)

    # get the physical length of the x and y axes
    ax_len = np.diff(np.array(ax.get_position())[:, 0]) * fig.get_size_inches()[0]
    size_const = ax_len / np.diff(ax.get_xlim())[0] * min_distance * 72
    size = size_const ** 2 * spot_size
    plt.scatter(row, col, s=size, marker=marker, c=cblend, **scr_kw)


def plot_component(adata, fig, ax, selected_dim, dim=8, hexcodes=None, seed=None, mode='aa', color_first='black',
                   sample_id=None, spot_size=1.05, marker="h", **scr_kw):

    # todo: have a look at spot_size, still not exactly proportional to the physical size of the plot

    # define PC colors
    if hexcodes is None:
        if dim > 8:
            hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5, seed=seed)
        else:
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
            pc_cmap = color_to_color(color_first, hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_aa'][:, d]),
                                           vmax=max(adata.obsm['chr_aa'][:, d]),
                                           value=adata.obsm['chr_aa'][:, d])
            cmaps.append(pc_rgb)

    elif mode == 'pca':
        for d in range(dim):
            pc_cmap = color_to_color(color_first, hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_X_pca'][:, d]),
                                           vmax=max(adata.obsm['chr_X_pca'][:, d]),
                                           value=adata.obsm['chr_X_pca'][:, d])
            cmaps.append(pc_rgb)
    else:
        raise Exception

    adata.obsm['cmap'] = cmaps[selected_dim]
    if sample_id is not None:
        adata = adata[adata.obs['sample'] == sample_id]

    # plot
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.axis('off')
    row = adata.obsm['spatial'][:, 0]
    col = adata.obsm['spatial'][:, 1] * -1
    row_range = np.ptp(row)
    col_range = np.ptp(col)
    ax.set_xlim((np.min(row) - 0.1 * row_range, np.max(row) + 0.1 * row_range))
    ax.set_ylim((np.min(col) - 0.1 * col_range, np.max(col) + 0.1 * col_range))
    ax.set_aspect('equal')

    # takes long time to compute the pairwise distance matrix for stereo-seq or slide-seq samples, so by looking at
    # only 5000 spots is a good enough approximation
    if len(row) < 5000:
        distances = cdist(np.column_stack((row, col)), np.column_stack((row, col)))
    else:
        distances = cdist(np.column_stack((row[:5000], col[:5000])), np.column_stack((row[:5000], col[:5000])))

    np.fill_diagonal(distances, np.inf)
    min_distance = np.min(distances)

    # get the physical length of the x and y axes
    ax_len = np.diff(np.array(ax.get_position())[:, 0]) * fig.get_size_inches()[0]
    size_const = ax_len / np.diff(ax.get_xlim())[0] * min_distance * 72
    size = size_const ** 2 * spot_size
    ax.scatter(row, col, s=size, marker=marker, c=adata.obsm['cmap'], **scr_kw)


def show_compartments(adata, ncols=2, size=3, sample_id=None, spot_size=0.85, hexcodes=None, title_size=10, seed=None,
                      marker="h", **scr_kw):

    ndims = adata.obsm['chr_aa'].shape[1]
    assert ndims / ncols >= 1
    nrows = math.ceil(ndims / ncols)

    if 'sample' in adata.obs.columns and sample_id is None:
        if sample_id not in adata.obs['sample'].cat.categories:
            raise ValueError("Invalid sample_id. Check categories in .obs['sample']")
        raise ValueError("Integrated dataset. Cannot proceed without a specified sample_id")

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))
    axs = axs.flatten()
    for a in axs:
        a.axis('off')
    plt.subplots_adjust(hspace=0.05, wspace=0.01, left=0.05, right=0.95, top=0.95, bottom=0.05)
    for i in range(ndims):
        plot_component(adata, fig, axs[i], dim=ndims, selected_dim=i, color_first='#2e2e2e', spot_size=spot_size,
                       sample_id=sample_id, hexcodes=hexcodes, seed=seed, marker=marker, **scr_kw)
        axs[i].set_title(f'Compartment {i}', size=title_size)


def plot_explained_variance(adata):
    sns.set_style("ticks")
    pca_df = pd.DataFrame(data=adata.uns['chr_pca']['variance_ratio'], columns=['Explained variance'])
    pca_df = pca_df.cumsum()

    n_svg = adata.var['spatially_variable'].value_counts()[True]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.lineplot(pca_df, markers=True, legend=True, ax=ax, palette=['#8b33ff'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
    ax.set_ylabel('Explained variance')
    ax.set_xlabel('PCs')
    ax.set_title(f'SVGs: {n_svg}')
    # ax.grid(axis='both')
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    plt.tight_layout()


def plot_svgs(adata):

    morans_df = adata.var["Moran's I"].sort_values(ascending=False)
    morans_df = morans_df.dropna()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.lineplot(list(morans_df), linewidth=2, color='#8b33ff')
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("Moran's I")
    ax.set_xlabel('Gene #')
    ax.set_title(f'SVGs')
    plt.tight_layout()


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


def chrysalis_svg(adata, min_spots=0.1, top_svg=1000, min_morans=0.20, neighbors=6, geary=False):
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

    # select thresholds to choose from
    if len(moran_df[:top_svg]) < len(moran_df[moran_df["Moran's I"] > min_morans]):
        adata.var['spatially_variable'] = [True if x in moran_df[:top_svg].index else False for x in adata.var_names]
    else:
        moran_df = moran_df[moran_df["Moran's I"] > min_morans]
        adata.var['spatially_variable'] = [True if x in moran_df.index else False for x in adata.var_names]


def chrysalis_pca(adata, n_pcs=50):
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


def chrysalis_aa(adata, n_archetypes=8, n_pcs=None):

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


def plot_rss(adata, title=None):

    if 'RSSs' not in adata.uns['chr_aa'].keys():
        raise ValueError(".uns['chr_aa']['RSSs'] cannot be found, run estimate_compartments first.")

    def find_elbow(x, y):
        y = np.array(list(y))
        x = np.array(list(x))
        data = np.column_stack((x, y))

        line_start = data[0]
        line_end = data[-1]

        line_vec = line_end - line_start
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))

        # for each point find the distance to the line
        vec_from_first = data - line_start
        scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (len(data), 1)), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        elbow_index = np.argmax(dist_to_line)
        return elbow_index

    ent_df = pd.DataFrame(data=adata.obsm['entropy'], columns=adata.uns['chr_aa']['RSSs'].keys())

    ent_df = pd.melt(ent_df)
    ent_df = ent_df.rename(columns={'variable': 'n_compartment', 'value': 'entropy'})
    ent_df['normalized_entropy'] = ent_df['entropy'] / ent_df['n_compartment']

    rss_df = pd.DataFrame(data=adata.uns['chr_aa']['RSSs'].values(),
                          index=adata.uns['chr_aa']['RSSs'].keys())

    elb_idx = find_elbow(adata.uns['chr_aa']['RSSs'].keys(),
                         adata.uns['chr_aa']['RSSs'].values())
    elb_aa = list(adata.uns['chr_aa']['RSSs'].keys())[elb_idx]

    if 'chr_aa' not in adata.uns.keys():
        adata.uns['chr_aa'] = {'RSS_opt': elb_aa}
    else:
        adata.uns['chr_aa']['RSS_opt'] = elb_aa

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.lineplot(rss_df, markers=True, legend=False, ax=axs[0], palette=['#8b33ff'])
    axs[0].plot(elb_aa, list(adata.uns['chr_aa']['RSSs'].values())[elb_idx], 'ro',
                markeredgecolor='white', linewidth=3)
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_ylabel('Residual sum of squares')
    axs[0].set_xlabel('Compartments')
    axs[0].set_title(f'RSS - Optimal component n: {int(elb_aa)}')
    axs[0].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    axs[0].set_axisbelow(True)

    sns.lineplot(data=ent_df, y='entropy', x='n_compartment', ax=axs[1], errorbar='se', color='#8b33ff')
    axs[1].grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    axs[1].set_axisbelow(True)
    axs[1].set_ylabel("Normalized Entropy")
    axs[1].set_xlabel('Compartments')
    axs[1].set_title(f'Entropy changes')
    axs[1].tick_params(axis='x', rotation=45)

    if title:
        plt.suptitle(title)

    plt.tight_layout()



def compartment_heatmap(adata, figsize=(5 , 7), reorder_comps=False, hexcodes=None, seed=None, **kwrgs):

    # SVG weights for each compartment
    df = pd.DataFrame(data=adata.uns['chr_aa']['loadings'], columns=adata.uns['chr_pca']['features'])
    df = df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

    dim = df.shape[0]

    # define compartment colors
    # default colormap with 8 colors
    if hexcodes is None:
        if dim > 8:
            hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5)
        else:
            hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
            if seed is None:
                np.random.seed(len(adata))
            else:
                np.random.seed(seed)
            np.random.shuffle(hexcodes)
    else:
        assert len(hexcodes) >= dim

    z = linkage(df.T, method='ward')
    order = leaves_list(z)
    df = df.iloc[:, order]
    if reorder_comps:
        z = linkage(df, method='ward')
        order = leaves_list(z)
        df = df.iloc[order,: ]
        hexcodes =  [hexcodes[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(df.T, ax=ax, cmap=sns.diverging_palette(45, 340, l=55, center="dark", as_cmap=True), **kwrgs)

    for idx, t in enumerate(ax.get_xticklabels()):
        t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))

    plt.tight_layout()


def get_compartment_df(adata, weights=True):
    # todo: these are based on a single capture spot, calculate average of multiple?
    # SVG expression for each compartment
    exp_array = np.asarray(adata[:, adata.var['spatially_variable'] == True].X.todense())
    exp_array = np.mean(exp_array, axis=0)
    exp_aa = adata.uns['chr_aa']['loadings']
    if not weights:
        exp_aa += exp_array

    df = pd.DataFrame(data=exp_aa, columns=adata.uns['chr_pca']['features'],
                      index=[f'compartment_{x}' for x in (range(len(exp_aa)))]).T
    return df


def plot_weights(adata, hexcodes=None, seed=None, compartments=None, ncols=4, w=1, h=1):

    expression_df = get_compartment_df(adata)
    dim = expression_df.shape[1]
    # define compartment colors
    # default colormap with 8 colors
    if hexcodes is None:
        if dim > 8:
            hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5)
        else:
            hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
            if seed is None:
                np.random.seed(len(adata))
            else:
                np.random.seed(seed)
            np.random.shuffle(hexcodes)
    else:
        assert len(hexcodes) >= dim

    if type(compartments) == list:
        assert all(isinstance(item, int) for item in compartments)
        compartments = [f'compartment_{x}' for x in compartments]
        assert all([True if x in expression_df.columns else False for x in compartments])
        expression_df = expression_df[compartments]

    n_comp = expression_df.shape[1]
    n_col = ncols
    n_row = math.ceil(n_comp / n_col)

    fig, ax = plt.subplots(n_row, n_col, figsize=(w * 3 * n_col, h * 4 * n_row))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, c in enumerate(expression_df.columns):
        cnum = int(c.split('_')[-1])
        sl = expression_df[[c]].sort_values(ascending=False, by=c)[:20]
        ax[idx].axis('on')
        # ax[idx].set_facecolor('#f2f2f2')
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].grid(axis='x', linestyle='-', linewidth='0.5', color='grey')
        ax[idx].set_axisbelow(True)
        ax[idx].axvline(0, color='black')
        ax[idx].barh(list(sl.index)[::-1], list(sl[c].values)[::-1], color=hexcodes[cnum])
        ax[idx].scatter(y=list(sl.index)[::-1], x=list(sl[c].values)[::-1], color='black', s=15)
        ax[idx].set_xlabel('Weight')
        ax[idx].set_title(f'Compartment {cnum}')
    plt.tight_layout()


def color_similarity_nh(adata, nh=6, mode='cos_sim'):
    scale = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    spot_diam = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']
    spot_diam = int(spot_diam * scale)

    hires = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['images']['hires']

    # get KNNs
    points = adata.obsm['spatial'].copy() * scale
    points = points.astype(int)
    w = weights.KNN.from_array(points, k=nh)

    # extract spots
    r = int(spot_diam / 2)
    patch_array = np.empty((len(points), r * 2, r * 2, 3))
    for idx, point in enumerate(points):
        p = hires[point[1] - r:point[1] + r, point[0] - r:point[0] + r]
        p = rgb2lab(p)
        # p = p.flatten()
        patch_array[idx] = p

    arr = np.empty((len(points), 1))

    for k, v in w.neighbors.items():
        patch = patch_array[k]
        nh_patch = []
        for p in v:
            npatch = hires[points[p, 1] - r:points[p, 1] + r, points[p, 0] - r:points[p, 0] + r]
            npatch = rgb2lab(npatch)
            # npatch = npatch.flatten()
            nh_patch.append(npatch)

        nh_patch = np.array(nh_patch)

        if mode == 'cos_sim':

            mean_es = []
            for i in range(nh):
                e_arr = cosine_similarity(patch.reshape(1, -1), nh_patch[i].reshape(1, -1))
                mean_es.append(e_arr)
            metric = np.mean(mean_es)

        elif mode == 'delta_e':

            mean_es = []
            for i in range(nh):
                e_arr = deltaE_ciede2000(patch, nh_patch[i])
                mean_es.append(np.mean(e_arr))
            metric = np.mean(mean_es)
        else:
            raise Exception("Mode must be 'delta_e' or 'cos_sim'.")

        arr[k] = np.mean(metric)

    # arr = arr.flatten()
    # arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr


def integrate_adatas(adatas, sample_names=None, **kwargs):

    if sample_names is None:
        sample_names = np.arange(len(adatas))
    assert len(adatas) == len(sample_names)

    adatas_dict = {}
    for ad, name in zip(adatas, sample_names):
        ad.obs['sample'] = name
        if 'gene_ids' in ad.var.columns:
            ad.var['gene_symbols'] = ad.var_names
            ad.var_names = ad.var['gene_ids']

        chrysalis_svg(ad, **kwargs)

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
