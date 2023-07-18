import math
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, leaves_list
from .utils import generate_random_colors, black_to_color, get_rgb_from_colormap, mip_colors, color_to_color, \
    get_compartment_df


def plot(adata: AnnData, dim: int=8, hexcodes: List[str]=None, seed: int=None, sample_id: Union[int, str]=None,
         spot_size: float=1.05, marker: str='h', figsize: Tuple[int, int]=(5, 5), **scr_kw):
    """
    Visualize tissue compartments using MIP (Maximum Intensity Projection).

    Tissue compartments need to be calculated using `chrysalis.aa`. If no hexcodes are provided, random colors are
    generated for the individual tissue compartments. Spot size is calculated automatically, however it can be
    fine-tuned using the `spot_size` parameter.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param dim: Number of components to visualize.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param sample_id: Sample id defined by `.obs['sample']` column.
    :param spot_size: Fine adjustments of the spot size.
    :param marker: Marker type.
    :param figsize: Figure size as a tuple.
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

    # todo: have a look at spot_size, still not exactly proportional to the physical size of the plot

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


def plot_compartment(adata: AnnData, fig: plt.figure, ax: plt.axis, selected_dim: int, dim: int=8,
                     hexcodes: List[str]=None, seed: int=None, color_first: str='black',
                     sample_id: Union[int, str]=None, spot_size: float=1.05, marker: str='h', **scr_kw):
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
    for d in range(dim):
        pc_cmap = color_to_color(color_first, hexcodes[d])
        pc_rgb = get_rgb_from_colormap(pc_cmap,
                                       vmin=min(adata.obsm['chr_X_pca'][:, d]),
                                       vmax=max(adata.obsm['chr_X_pca'][:, d]),
                                       value=adata.obsm['chr_X_pca'][:, d])
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

    # get the physical length of the x and y axes
    ax_len = np.diff(np.array(ax.get_position())[:, 0]) * fig.get_size_inches()[0]
    size_const = ax_len / np.diff(ax.get_xlim())[0] * min_distance * 72
    size = size_const ** 2 * spot_size
    ax.scatter(row, col, s=size, marker=marker, c=adata.obsm['cmap'], **scr_kw)


def plot_compartments(adata: AnnData, ncols: int=2, size: int=3, sample_id: Union[int, str]=None,
                      spot_size: float=0.85, hexcodes: List[str]=None, title_size: int=10, seed: int=None,
                      marker: str='h', **scr_kw):
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
        plot_compartment(adata, fig, axs[i], dim=ndims, selected_dim=i, color_first='#2e2e2e', spot_size=spot_size,
                       sample_id=sample_id, hexcodes=hexcodes, seed=seed, marker=marker, **scr_kw)
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


def plot_svgs(adata: AnnData):
    """
    Plot a rank-order chart displaying the Moran's I values.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.

    """

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
                 seed: int=None, **kwrgs):
    """
    Plot heatmap showing the weights of the spatially variable genes for each identified tissue compartment.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param figsize: Figure size as a tuple.
    :param reorder_comps:
        Perform hierarchical clustering to reorder compartments based on the similarity of spatially variable gene
        weights.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param kwrgs: Seaborn heatmap keyword arguments.

    """

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