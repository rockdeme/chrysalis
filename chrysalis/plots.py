import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Union, Tuple
from scipy.spatial.distance import cdist
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from scipy.cluster.hierarchy import linkage, leaves_list
from .utils import generate_random_colors, black_to_color, get_rgb_from_colormap, mip_colors, color_to_color, \
    get_compartment_df, get_hexcodes


def hex_collection(x, y, c, s, scale_factor, ax, rotation=30, marker='h', **kwargs):
    """
    Scatter plot alternative with proper scaling.

    :param x: rows
    :param y: columns
    :param c: color
    :param s: size
    :param scale_factor: scale factor
    :param ax: axis
    :param rotation: marker rotation in radians
    :param kwargs: PatchCollection kwargs
    """

    if scale_factor != 1.0:
        x = x * scale_factor
        y = y * scale_factor
    zipped = np.broadcast(x, y, s)

    if marker == 'h':
        patches = [RegularPolygon((x, y), radius=s, numVertices=6, orientation=np.radians(rotation)) for x, y, s in zipped]
    elif marker == 's':
        patches = [RegularPolygon((x, y), radius=s, numVertices=4, orientation=np.radians(rotation)) for x, y, s in zipped]
    else:
        raise Exception("No valid marker type was defined ('h', 's')")
    collection = PatchCollection(patches, edgecolor='none', **kwargs)
    collection.set_facecolor(c)

    ax.add_collection(collection)


def plot(adata: AnnData, dim: int=8, hexcodes: List[str]=None, seed: int=None, sample_id: Union[int, str]=None,
         sample_col: str='sample', spot_size: float=1.05, marker: str='h', figsize: Tuple[int, int]=(5, 5),
         ax: Axes=None, dpi: int=100, selected_comp: Union[int, str]='all', rotation: int=0, uns_spatial_key: str=None,
         **scr_kw):
    """
    Visualize tissue compartments using MIP (Maximum Intensity Projection).

    Tissue compartments need to be calculated using `chrysalis.aa`. If no hexcodes are provided, random colors are
    generated for the individual tissue compartments. Spot size is calculated automatically, however it can be
    fine-tuned using the `spot_size` parameter.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param dim: Number of components to visualize.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param sample_id:
        ID corresponding to the sample as defined in the sample column, stored `.obs['sample']` by default.
    :param sample_col:
        The `.obs` column storing the `sample_id` information, 'sample' by default.
    :param spot_size: Adjust the final spot size.
    :param marker: Marker type.
    :param figsize: Figure size as a tuple.
    :param ax: Draw plot on a specific Matplotlib axes instead of a figure if specified.
    :param dpi: Optional DPI value used when `ax` is specified.
    :param selected_comp: Show only the selected compartment if specified.
    :param rotation: Rotate markers for alternative lattice arrangements.
    :param uns_spatial_key: Alternative key in .uns['spatial'] storing spot size and scaling factor.
    :param scr_kw: Matplotlib scatterplot keyword arguments.

    Example usage:

    >>> import chrysalis as ch
    >>> import scanpy as sc
    >>> import matplotlib.pyplot as plt
    >>> adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pp.filter_cells(adata, min_counts=6000)
    >>> sc.pp.filter_genes(adata, min_cells=10)
    >>> ch.detect_svgs(adata)
    >>> sc.pp.normalize_total(adata, inplace=True)
    >>> sc.pp.log1p(adata)
    >>> ch.pca(adata)
    >>> ch.aa(adata, n_pcs=20, n_archetypes=8)
    >>> ch.plot(adata, dim=8)
    >>> plt.show()

    """

    # todo: have a look at spot_size, still not scales linearly with the distance between spots

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


    # todo: this can raise error when we have one sample with the default sample coulmn already defined, needs fix
    if sample_col in adata.obs.columns and sample_id is None:
        if sample_id not in adata.obs['sample'].cat.categories:
            raise ValueError(f"Invalid sample_id. Check categories in .obs['{sample_col}']")
        raise ValueError("Integrated dataset. Cannot proceed without a specified sample column from .obs.")

    if sample_id is not None:
        cblend = [x for x, b in zip(cblend, list(adata.obs[sample_col] == sample_id)) if b == True]
        adata = adata[adata.obs[sample_col] == sample_id]

    if ax is None:
        # todo: update this part with hex_collection
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

    else:
        if sample_id not in adata.uns['spatial'].keys():
            size = 1
            raise Warning("Sample ID is not found in adata.uns['spatial']. Make sure that the provided sample id column"
                          "is the same as the sample ID in .uns.")
        else:
            size = adata.uns['spatial'][sample_id]['scalefactors']['spot_diameter_fullres']
            scale_factor = adata.uns['spatial'][sample_id]['scalefactors']['tissue_hires_scalef']

        if uns_spatial_key != None:
            sample_id = uns_spatial_key

        row = adata.obsm['spatial'][:, 0]
        col = adata.obsm['spatial'][:, 1] * -1
        row_range = np.ptp(row)
        col_range = np.ptp(col)
        xrange = (np.min(row) - 0.1 * row_range, np.max(row) + 0.1 * row_range)
        yrange = (np.min(col) - 0.1 * col_range, np.max(col) + 0.1 * col_range)
        xrange = tuple(np.array(xrange) * scale_factor)
        yrange = tuple(np.array(yrange) * scale_factor)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.set_aspect('equal')

        circle_radius = spot_size * scale_factor * size * 0.5
        hex_collection(row, col, cblend, circle_radius, scale_factor, ax, marker=marker, rotation=rotation, **scr_kw)


def plot_compartment(adata: AnnData, fig: plt.figure, ax: plt.axis, selected_dim: int, dim: int=8,
                     hexcodes: List[str]=None, seed: int=None, color_first: str='black', rotation: Union[int, float]=0,
                     sample_id: Union[int, str]=None, spot_size: float=1.05, marker: str='h', backend='patch_collection',
                     **scr_kw):
    """
    Visualize individual tissue compartments.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param fig: Matplotlib figure.
    :param ax: Matplotlib axis.
    :param selected_dim: Selected compartment.
    :param dim: Total number of compartments.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param color_first: Color mapped to 0 domain score values.
    :param sample_id: Sample id defined by `.obs['sample']` column.
    :param spot_size: Fine adjustments of the spot size.
    :param marker: Marker type.
    :param scr_kw: Matplotlib scatterplot keyword arguments.

    """

    # define PC colors
    hexcodes = get_hexcodes(hexcodes, dim, seed, len(adata))

    # define colormaps
    cmaps = []
    for d in range(dim):
        pc_cmap = color_to_color(color_first, hexcodes[d])
        pc_rgb = get_rgb_from_colormap(pc_cmap,
                                       vmin=min(adata.obsm['chr_aa'][:, d]),
                                       vmax=max(adata.obsm['chr_aa'][:, d]),
                                       value=adata.obsm['chr_aa'][:, d])
        cmaps.append(pc_rgb)

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

    # patch collection struggles with a large number of spots, improve this later
    if backend == 'patch_collection':
        # INTRODUCED HEX COLLECTION HERE
        size = 1
        scale_factor = 1
        if 'spatial' in adata.uns.keys():
            if sample_id not in adata.uns['spatial'].keys():
                warnings.warn("Sample ID is not found in adata.uns['spatial']. Make sure that the provided sample id column"
                              "is the same as the sample ID in .uns.")
            else:
                size = adata.uns['spatial'][sample_id]['scalefactors']['spot_diameter_fullres']
                scale_factor = adata.uns['spatial'][sample_id]['scalefactors']['tissue_hires_scalef']

        row = adata.obsm['spatial'][:, 0]
        col = adata.obsm['spatial'][:, 1] * -1
        row_range = np.ptp(row)
        col_range = np.ptp(col)
        xrange = (np.min(row) - 0.1 * row_range, np.max(row) + 0.1 * row_range)
        yrange = (np.min(col) - 0.1 * col_range, np.max(col) + 0.1 * col_range)
        xrange = tuple(np.array(xrange) * scale_factor)
        yrange = tuple(np.array(yrange) * scale_factor)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.set_aspect('equal')

        circle_radius = spot_size * scale_factor * size * 0.5
        hex_collection(row, col, adata.obsm['cmap'], circle_radius, scale_factor, ax, marker=marker,
                       rotation=rotation, **scr_kw)
    elif backend == 'scatter':
        # get the physical length of the x and y axes
        ax_len = np.diff(np.array(ax.get_position())[:, 0]) * fig.get_size_inches()[0]
        size_const = ax_len / np.diff(ax.get_xlim())[0] * min_distance * 72
        size = size_const ** 2 * spot_size
        ax.scatter(row, col, s=size, marker=marker, c=adata.obsm['cmap'], **scr_kw)
    else:
        raise Exception("Invalid backend ('patch_collection', 'scatter').")


def plot_compartments(adata: AnnData, ncols: int=2, size: int=3, sample_id: Union[int, str]=None,
                      sample_col: str='sample', rotation: Union[int, float]=0,
                      spot_size: float=0.85, hexcodes: List[str]=None, title_size: int=10, seed: int=None,
                      marker: str='h', backend='patch_collection', **scr_kw):
    """
    Visualize all compartments as individual subplots.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param ncols: Number of subplot columns.
    :param size: Subplot size.
    :param sample_id: Sample id defined by `.obs['sample']` column.
    :param spot_size: Fine adjustments of the spot size.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param title_size: Font size of subplot titles.
    :param seed: Random seed, used for mixing colors.
    :param marker: Marker type.
    :param scr_kw: Matplotlib scatterplot keyword arguments.

    """

    ndims = adata.obsm['chr_aa'].shape[1]
    assert ndims / ncols >= 1
    nrows = math.ceil(ndims / ncols)

    if sample_col in adata.obs.columns and sample_id is None:
        if sample_id not in adata.obs['sample'].cat.categories:
            raise ValueError("Invalid sample_id. Check categories in .obs['sample']")
        raise ValueError("Integrated dataset. Cannot proceed without a specified sample_id")

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))
    axs = axs.flatten()
    for a in axs:
        a.axis('off')
    plt.subplots_adjust(hspace=0.05, wspace=0.01, left=0.05, right=0.95, top=0.95, bottom=0.05)
    for i in range(ndims):
        plot_compartment(adata, fig, axs[i], dim=ndims, selected_dim=i, color_first='#2e2e2e', spot_size=spot_size,
                         sample_id=sample_id, hexcodes=hexcodes, seed=seed, marker=marker, rotation= rotation,
                         backend=backend, **scr_kw)
        axs[i].set_title(f'Compartment {i}', size=title_size)


def plot_explained_variance(adata: AnnData):
    """
    Plot the explained variance of the calculated PCs (Principal Components).

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.

    """

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


def plot_svgs(adata: AnnData, figsize=(4, 4)):
    """
    Plot a rank-order chart displaying the Moran's I values.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.

    """

    morans_df = adata.var["Moran's I"].sort_values(ascending=False)
    morans_df = morans_df.dropna()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(list(morans_df), linewidth=2, color='#8b33ff')
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("Moran's I")
    ax.set_xlabel('Gene #')
    ax.set_title(f'SVGs')
    plt.tight_layout()


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


def plot_heatmap(adata: AnnData, figsize: Tuple[int, int]=(5, 7), reorder_comps: bool=False, hexcodes: List[str]=None,
                 seed: int=None, scaling=True, **kwrgs):
    """
    Plot heatmap showing the weights of spatially variable genes for each identified tissue compartment.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param figsize: Figure size as a tuple.
    :param reorder_comps:
        Perform hierarchical clustering to reorder compartments based on the similarity of spatially variable gene
        weights.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param scaling: Column-wise scaling (x - mean / std).
    :param kwrgs: Seaborn heatmap keyword arguments.

    """

    # SVG weights for each compartment
    df = pd.DataFrame(data=adata.uns['chr_aa']['loadings'], columns=adata.uns['chr_pca']['features'])
    if scaling:
        df = df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

    dim = df.shape[0]

    # define compartment colors
    # default colormap with 8 colors
    hexcodes = get_hexcodes(hexcodes, dim, seed, len(adata))

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


def plot_weights(adata: AnnData, hexcodes: List[str]=None, seed: int=None, compartments: List[int]=None, ncols: int=4,
                 w: float=1.0, h: float=1.0):
    """
    Plot 20 top genes for each tissue compartment.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param compartments:
        If `None` show genes for all compartments, else show genes for compartments specified in a list.
    :param ncols: Number of subplot columns.
    :param w: Subplot width multiplier.
    :param h: Subplot height multiplier.

    """

    expression_df = get_compartment_df(adata)
    dim = expression_df.shape[1]
    # define compartment colors
    # default colormap with 8 colors
    if hexcodes is None:
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


def plot_svg_matrix(adatas: List[AnnData], figsize: tuple=(5, 4), obs_name: str=None, cluster: bool=False):
    """
    Plot the pairwise overlap of spatially variable genes across samples.

    :param adatas: List of AnnData objects containing the 'spatially_variable' column in `.var`.
    :param figsize: Figure size as a tuple.
    :param obs_name: `.obs` column containing categorical variables that can be used as a label for each sample.
    :param cluster: Reorder columns/rows based on hierarchical clustering.

    """

    svg_sets = []
    labels = []
    for ad in adatas:
        gene_names = ad.var[ad.var['spatially_variable'] == True].index
        svg_sets.append(set(gene_names))

        if obs_name:
            labels.append(ad.obs[obs_name][0])

    overlaps = np.zeros((len(adatas), len(adatas)))

    for i in range(len(svg_sets)):
        for j in range(len(svg_sets)):
            overlap = len(svg_sets[i].intersection(svg_sets[j]))
            overlaps[i][j] = overlap
            overlaps[j][i] = overlap

    overlaps = overlaps.astype(int)

    svg_total = list(set().union(*svg_sets))

    if cluster:
        overlaps_df = pd.DataFrame(data=overlaps)
        z = linkage(overlaps_df.T, method='ward')
        order = leaves_list(z)
        overlaps_df = overlaps_df.iloc[order, order]
        overlaps = overlaps_df.values

    xlabels = 'auto'
    ylabels = 'auto'

    if obs_name:
        if cluster:
            labels = [labels[i] for i in order]
        xlabels = labels
        ylabels = labels

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(overlaps, ax=ax, square=True, vmin=0, cmap=sns.color_palette("Spectral_r", as_cmap=True),
                annot=True, fmt=".0f", xticklabels=xlabels, yticklabels=ylabels)

    ax.set_ylabel('Sample')
    ax.set_xlabel('Sample')
    ax.set_title(f'SVG overlaps - Total overlap {len(svg_total)}')
    # ax.grid(axis='both')
    ax.set_axisbelow(True)
    plt.tight_layout()


def plot_samples(adata: AnnData, rows: int, cols: int, dim: int, selected_comp: Union[int, str]='all',
                 sample_col: str='sample', suptitle: str=None, plot_size: float=4.0, show_title: bool=True,
                 spot_size=2, rotation=0, **plot_kw):
    """
    Visualize multiple samples from an AnnData object integrated with `chrysalis.integrate_adatas` in a single figure.

    For details see `chrysalis.plot`. Individual compartments can be visualized instead of maximum intensity projection
    using 'selected_comp'.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param rows: Number of rows for subplots.
    :param cols: Number of columns for subplots.
    :param dim: Number of components to visualize.
    :param selected_comp: Show only the selected compartment if specified.
    :param sample_col:
        The `.obs` column storing the `sample_id` information, 'sample' by default.
    :param suptitle: Add suptitle to the figure.
    :param plot_size: Height and width of the individual subplots.
    :param show_title: Show title using labels from the `.obs` column defined using `sample_col`.
    :param spot_size: Adjust the final spot size.
    :param rotation: Rotate markers for alternative lattice arrangements.
    :param plot_kw: `chrysalis.plot` keyword arguments.

    """
    assert sample_col in adata.obs.columns

    fig, ax = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, i in enumerate(adata.obs[sample_col].cat.categories):
        plot(adata, dim=dim, sample_id=i, ax=ax[idx], sample_col=sample_col, selected_comp=selected_comp,
             spot_size=spot_size, rotation=rotation, **plot_kw)
        if show_title:
            obs_df = adata.obs[adata.obs[sample_col] == i]
            ax[idx].set_title(f'{obs_df[sample_col][0]}')
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=15)
    plt.tight_layout()
