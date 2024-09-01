import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from functions import generate_grid, return_cell_types, sample_model


def generate_tissue_zones(pattern_type='gp', grid_size=(50, 50), seed=42, n_tissue_zones=10, unfiform_fraction=0.3,
                          gp_mean_var_ratio=1.2, gp_eta_true=1.5, gp_eta_true_uniform=0.5, gp_mean=8,
                          r_d_mean_var_ratio=0.3, r_d_mean=5):
    np.random.seed(seed)

    # generate grid
    spatial_locs, x, y = generate_grid(grid_size)

    locations = spatial_locs

    if pattern_type == 'gp':
        n_tissue_zones_uniform = int(n_tissue_zones * unfiform_fraction)
        assert n_tissue_zones_uniform <= n_tissue_zones

        n_tissue_zones = n_tissue_zones - n_tissue_zones_uniform

        # initialize parameters for GP - cell2loc paper
        # tissue zones
        mean_var_ratio = gp_mean_var_ratio
        l1_tissue_zones = np.random.gamma(gp_mean * mean_var_ratio, 1 / mean_var_ratio, size=n_tissue_zones)
        l1_tissue_zones_uniform = np.random.gamma(gp_mean * mean_var_ratio, 1 / mean_var_ratio,
                                                  size=n_tissue_zones_uniform)
        # generate the abundances for the tissue zones
        abundance_df = sample_model(spatial_locs, x, y,
                                    n_tissue_zones=n_tissue_zones,
                                    l1_tissue_zones=l1_tissue_zones,
                                    n_tissue_zones_unfirom=n_tissue_zones_uniform,
                                    l1_tissue_zones_uniform=l1_tissue_zones_uniform,
                                    eta_true=gp_eta_true, eta_true_uniform=gp_eta_true_uniform,
                                    pattern_type=pattern_type)

    elif pattern_type == 'r_d':
        mean_var_ratio = r_d_mean_var_ratio
        a = np.random.gamma(r_d_mean * mean_var_ratio, 1 / mean_var_ratio, size=n_tissue_zones)
        abundance_df = sample_model(spatial_locs, x, y, n_tissue_zones=n_tissue_zones, a=a,
                                    pattern_type=pattern_type, l1_tissue_zones=a)

    else:
        raise Exception("pattern_type parameter should be one of the following: ['gp', 'r_d']")

    return abundance_df, locations


def assign_cell_types(adata, n_tissue_zones, annot_col='louvain', cell_type_p=0.1, col_name_prefix='tissue_zone',
                      unfiform_fraction=0.3, seed=42):
    """
    Construct a DataFrame of the cell type composition of specified tissue zones by randomly drawing which tissue zones
    the cell types will belong to from a binomial distribution controlled by parameter p.

    :param adata:
    :param n_tissue_zones:
    :param annot_col:
    :param p:
    :param col_name_prefix:
    :param draw_uniform:
    :return:
    """
    np.random.seed(seed)

    cell_types, n_cell_types = return_cell_types(adata, annot_col)

    n_uniform = int(n_tissue_zones * unfiform_fraction)
    n_sparse_ct = n_cell_types - n_uniform
    n_sparse_tz = n_tissue_zones - n_uniform

    assert n_uniform < n_cell_types, 'n_uniform < n_cell_types'

    if n_uniform > 0:
        # if len(cell_types) < n_uniform:
        #     uniform_cell_types = np.random.choice(cell_types, n_uniform, replace=True)
        #     raise Warning('We have more uniform tissue zones than cell types! Cell types can be assigned to more'
        #                   'than one uniform tissue zone.')
        # else:
        #     uniform_cell_types = np.random.choice(cell_types, n_uniform, replace=False)
        # sparse_cell_types = cell_types[~np.isin(cell_types, uniform_cell_types)]
        # n_sparse = n_cell_types - n_uniform

        # we are drawing from a binomial to get the number of cell types per tissue zone
        # p = 0.02 will have 1-2, p = 0.1 will have 1-4
        n_zones_per_cell_type = np.random.binomial(n_tissue_zones, p=cell_type_p, size=n_sparse_ct) + 1
    else:
        # assert n_cell_types >= n_tissue_zones
        n_zones_per_cell_type = np.random.binomial(n_tissue_zones, p=cell_type_p, size=n_cell_types) + 1

    # generate matrix of which cell types are in which zones
    # [f'tissue_zone_{i}' for i in range(n_tissue_zones)] + [f'uniform_{i}' for i in range(n_uniform)]

    col_names = [f'{col_name_prefix}_{i}' for i in range(n_sparse_tz)] + [f'uniform_{i}' for i in range(n_uniform)]
    cell_types_df = pd.DataFrame(0, index=cell_types, columns=col_names)
    for i, n in enumerate(n_zones_per_cell_type):
        pos = np.random.randint(n_sparse_tz, size=n)
        cell_types_df.iloc[i, pos] = 1

    # and which uniform cell types belong to which uniform pattern
    if n_uniform > 0:
        for i in range(n_uniform):
            cell_types_df.iloc[n_sparse_ct + i, n_sparse_tz + i] = 1

    return cell_types_df


def assign_cell_type_abundance(cell_types_df, p_high_density=0.5, mu_low_density=1.0, mu_high_density=2.8, seed=42,
                               uniform_name_prefix='uniform'):
    """
    Add average cell type abundance to the cell_types_df by drawing from gamma distributions. We can assign high and
    low density cell types.

    # todo: implement uniform cells.

    :param cell_types_df:
    :param p_high_density:
    :param mu_low_density:
    :param mu_high_density:
    :return:

    """
    np.random.seed(seed)

    cell_types = np.unique(cell_types_df.index)

    uniform_ct_df = cell_types_df[[x for x in cell_types_df.columns if uniform_name_prefix in x]]
    uniform_cell_types = []
    for c in uniform_ct_df.columns:
        tissue_zone_ser = uniform_ct_df[c]
        ctl = tissue_zone_ser[tissue_zone_ser == 1].index
        for ct in ctl:
            uniform_cell_types.append(ct)

    # Assign cell types to either high or low density, balanced by uniform / tissue zone
    n_uniform = len(uniform_cell_types)
    if n_uniform == 0:
        high_density_cell_types = []
    else:
        high_density_cell_types = list(np.random.choice(uniform_cell_types, int(np.round(n_uniform * p_high_density)),
                                                        replace=False))
    for z, n in cell_types_df.sum().items():
        ct = list(np.random.choice(cell_types_df.index[cell_types_df[z] > 0],
                                   int(np.round(n * p_high_density)),
                                   replace=False))
        high_density_cell_types += ct

    low_density_cell_types = cell_types[~np.isin(cell_types, high_density_cell_types)]

    # generate average abundance for low and high density cell types
    mean_var_ratio = 5

    cell_types_df.loc[low_density_cell_types] = (cell_types_df.loc[low_density_cell_types]
                                                 * np.random.gamma(shape=mu_low_density * mean_var_ratio,
                                                                   scale=1 / mean_var_ratio,
                                                                   size=(len(low_density_cell_types), 1)))

    cell_types_df.loc[high_density_cell_types] = (cell_types_df.loc[high_density_cell_types]
                                                  * np.random.gamma(shape=mu_high_density * mean_var_ratio,
                                                                    scale=1 / mean_var_ratio,
                                                                    size=(len(high_density_cell_types), 1)))
    return cell_types_df


def construct_cell_abundance_matrix(abundance_df, cell_types_df, seed=42):

    np.random.seed(seed)

    cell_abundances = np.dot(abundance_df, cell_types_df.T)
    cell_abundances = cell_abundances * np.random.lognormal(0, 0.35, size=cell_abundances.shape)
    cell_abundances_df = pd.DataFrame(cell_abundances, index=abundance_df.index, columns=cell_types_df.index)

    cell_count_df = np.ceil(cell_abundances_df)
    capture_eff_df = cell_abundances_df / cell_count_df
    capture_eff_df[capture_eff_df.isna()] = 0
    return cell_count_df, capture_eff_df


def generate_synthetic_counts(adata, cell_count_df, capture_eff_df, annot_col='louvain', seed=42):
    """
    We are drawing cell indexes for each spatial location from the cell types that are mapped to that location. We do
    not remove cells that were drawn before for a specific location at the moment which could influence things
    especially if we dont have too many cells to draw from.
    :param adata:
    :param cell_count_df:
    :param capture_eff_df:
    :param annot_col:
    :return:
    """
    np.random.seed(seed)
    adata.obs['cell_idx'] = np.arange(adata.shape[0])
    cell_map_array = np.zeros((cell_count_df.shape[0], adata.shape[0]))
    for i, l in enumerate(cell_count_df.index):
        for j, ct in enumerate(cell_count_df.columns):
            cell_idx_all = adata.obs['cell_idx']
            cell_idx_all = cell_idx_all[adata.obs[annot_col] == ct]
            cell_idx = np.random.choice(cell_idx_all, int(cell_count_df.loc[l, ct]), replace=False)
            cell_map_array[i, cell_idx] = capture_eff_df.loc[l, ct]


    # counts = adata.to_df().values  # this should be better than .X but might cause isssue with large arrays
    counts = csr_matrix(adata.X)
    cell_map_array = csr_matrix(cell_map_array)
    synthetic_counts = cell_map_array.dot(counts)
    return synthetic_counts


def construct_adata(synthetic_counts, adata, abundance_df, locations, cell_count_df, capture_eff_df):
    # Compute synthetic counts
    # Create adata object
    synth_adata = anndata.AnnData(synthetic_counts)
    synth_adata.obs_names = cell_count_df.index
    synth_adata.var_names = adata.var_names

    # synth_adata.obs[[f'cell_count_{ct}' for ct in cell_count_df.columns]] = cell_count_df.astype(int)
    synth_adata.obsm['cell_abundance'] = cell_count_df.astype(int)
    # synth_adata.obs[[f'{ct}' for ct in abundance_df.columns]] = abundance_df
    synth_adata.obsm['tissue_zones'] = abundance_df
    # synth_adata.obs[[f'cell_capture_eff_{ct}' for ct in capture_eff_df.columns]] = capture_eff_df
    synth_adata.obsm['capture_efficiency'] = capture_eff_df
    synth_adata.obsm['spatial'] = locations

    return synth_adata


def add_confounders(synthetic_adata, mu_detection=5, mu_contamination=0.03, mu_depth=1, depth_loc_mean_var_ratio=25,
                    mu_depth_exp=1, depth_mean_var_ratio_exp=5, depth_override=None, contamination_override=None,
                    seed=42):
    """

    :param synthetic_adata:
    :param mu_detection: detection rate shape
    :param mu_contamination: contamination count shape
    :param mu_depth: sequencing depth shape
    :param depth_loc_mean_var_ratio: sequencing depth scale
    :param mu_depth_exp: sample-wise sequencing depth shape 
    :param depth_mean_var_ratio_exp: sample-wise sequencing depth scale 
    :return:
    """
    np.random.seed(seed)

    show_plots = False

    # sample detection rates
    gene_level = np.random.gamma(shape=mu_detection, scale=1 / 15, size=(1, synthetic_adata.shape[1]))
    # multiply counts with the detection rate coming from the gamma distribution
    synthetic_adata.var['gene_level'] = gene_level.flatten()
    synthetic_adata.X = synthetic_adata.X.multiply(gene_level)

    # sample poisson integers
    # drawing counts from a poisson distribution where the average is the detection rate multiplied count
    # we are doing this at the end after modeling contamination and sequencing depth effects

    # sth_adata.layers['expression_levels'] = sth_adata.X
    # sth_adata.X = np.random.poisson(sth_adata.X)
    # sth_adata.X = sth_adata.X

    # sample contamination
    cont_mean_var_ratio = 100
    if contamination_override is not None:
        contamination = contamination_override
        print('Overriding sample contamination with the exact value.')
    else:
        contamination = np.random.gamma(mu_contamination * cont_mean_var_ratio, 1 / cont_mean_var_ratio)
        # print(f'{contamination} input mu: {mu_contamination}')
    contamination_average = synthetic_adata.X.toarray().mean(0)
    contamination = contamination * contamination_average
    contamination = np.expand_dims(contamination, axis=0)

    locs = np.array([1 for x in range(len(synthetic_adata))])
    locs = np.expand_dims(locs, axis=1)
    per_loc_average_contamination = np.dot(locs, contamination)

    # samples per-location variability in sequencing depth
    per_location_depth = np.random.gamma(mu_depth * depth_loc_mean_var_ratio, 1 / depth_loc_mean_var_ratio,
                                         size=(synthetic_adata.shape[0], 1))
    # sample per-experiment variability in sequencing depth
    if depth_override is not None:
        per_experiment_depth = np.array([[depth_override]])
        assert per_experiment_depth.shape == (1, 1)
        print('Overriding relative sequencing depth with the exact value.')
    else:
        per_experiment_depth = np.random.gamma(mu_depth_exp * depth_mean_var_ratio_exp,
                                               1 / depth_mean_var_ratio_exp, size=(1, 1))
        # print(f'{per_experiment_depth[0][0]} input mu: {mu_depth_exp}')
    per_location_depth_total = per_location_depth * np.dot(locs, per_experiment_depth)
    if show_plots:
        plt.hist(np.array(per_location_depth_total).flatten(), bins=100)
        plt.show()

    # add sequencing depth effect
    expression_levels_depth = synthetic_adata.X.multiply(per_location_depth_total)
    expression_levels_depth = expression_levels_depth.toarray()
    synthetic_adata.X = np.random.poisson(expression_levels_depth)

    # add contamination counts
    contamination = np.random.poisson(np.array(per_loc_average_contamination) * per_location_depth_total)
    synthetic_adata.X = synthetic_adata.X + contamination
    synthetic_adata.X = csr_matrix(synthetic_adata.X)
    # synthetic_adata.X = synthetic_adata.X.astype(int)

    return synthetic_adata


def add_confounders_v2(synthetic_adata, mu_detection=5, mu_contamination=0.03, mu_depth=1, depth_loc_mean_var_ratio=25,
                       mu_depth_exp=1, depth_mean_var_ratio_exp=5, depth_override=None, contamination_override=None,
                       seed=42):
    """

    :param synthetic_adata:
    :param mu_detection: detection rate shape
    :param mu_contamination: contamination count shape
    :param mu_depth: sequencing depth shape
    :param depth_loc_mean_var_ratio: sequencing depth scale
    :param mu_depth_exp: sample-wise sequencing depth shape
    :param depth_mean_var_ratio_exp: sample-wise sequencing depth scale
    :return:
    """
    np.random.seed(seed)

    show_plots = False

    # sample detection rates
    gene_level = np.random.gamma(shape=mu_detection, scale=1 / 15, size=(1, synthetic_adata.shape[1]))
    # multiply counts with the detection rate coming from the gamma distribution
    synthetic_adata.var['gene_level'] = gene_level.flatten()
    synthetic_adata.X = synthetic_adata.X.multiply(gene_level)

    # sample poisson integers
    # drawing counts from a poisson distribution where the average is the detection rate multiplied count
    # we are doing this at the end after modeling contamination and sequencing depth effects

    # sth_adata.layers['expression_levels'] = sth_adata.X
    # sth_adata.X = np.random.poisson(sth_adata.X)
    # sth_adata.X = sth_adata.X

    # sample contamination
    cont_mean_var_ratio = 100
    if contamination_override is not None:
        contamination = contamination_override
        print(contamination_override)
        assert 1 > contamination_override > 0
        print('Overriding sample contamination with the exact value.')

    else:
        contamination = np.random.gamma(mu_contamination * cont_mean_var_ratio, 1 / cont_mean_var_ratio)
        # print(f'{contamination} input mu: {mu_contamination}')

    contamination_average = synthetic_adata.X.toarray().mean(0)
    contamination = contamination * contamination_average

    contamination = np.expand_dims(contamination, axis=0)

    locs = np.array([1 for x in range(len(synthetic_adata))])
    locs = np.expand_dims(locs, axis=1)
    per_loc_average_contamination = np.dot(locs, contamination)

    cfr = sum(sum(per_loc_average_contamination)) / sum(sum(synthetic_adata.X.toarray()))
    print(f'Cotamination count fraction {cfr} | contamination override: {contamination_override}')

    # shuffle the contamination array to replace uniform contamination
    # column shuffling - gene specific variability
    # full shuffle - across genes and spatial locations
    flattened = per_loc_average_contamination.flatten()
    np.random.shuffle(flattened)
    per_loc_average_contamination = flattened.reshape(per_loc_average_contamination.shape)

    total_counts_orig = sum(sum(synthetic_adata.X.toarray()))

    # samples per-location variability in sequencing depth
    per_location_depth = np.random.gamma(mu_depth * depth_loc_mean_var_ratio, 1 / depth_loc_mean_var_ratio,
                                         size=(synthetic_adata.shape[0], 1))
    # sample per-experiment variability in sequencing depth
    if depth_override is not None:
        per_experiment_depth = np.array([[depth_override]])
        assert per_experiment_depth.shape == (1, 1)
        print('Overriding relative sequencing depth with the exact value.')
    else:
        per_experiment_depth = np.random.gamma(mu_depth_exp * depth_mean_var_ratio_exp,
                                               1 / depth_mean_var_ratio_exp, size=(1, 1))
        # print(f'{per_experiment_depth[0][0]} input mu: {mu_depth_exp}')


    per_location_depth_total = per_location_depth * np.dot(locs, per_experiment_depth)

    # adjusting for contamination to keep the total count number the same
    exp_count_fraction = 1 - contamination_override
    per_location_depth_total = per_location_depth_total * exp_count_fraction

    if show_plots:
        plt.hist(np.array(per_location_depth_total).flatten(), bins=100)
        plt.show()

    # add sequencing depth effect

    expression_levels_depth = synthetic_adata.X.multiply(per_location_depth_total)
    expression_levels_depth = expression_levels_depth.toarray()

    print(f'Count matrix should be {per_experiment_depth[0][0]} * {exp_count_fraction} = '
          f'{per_experiment_depth[0][0] * exp_count_fraction}')
    print(f'Count matrix {sum(sum(expression_levels_depth)) / sum(sum(synthetic_adata.X.toarray()))}')
    synthetic_adata.X = np.random.poisson(expression_levels_depth)

    # add contamination counts
    contamination = np.random.poisson(np.array(per_loc_average_contamination) * np.dot(locs, per_experiment_depth))
    cfar = sum(sum(contamination)) / total_counts_orig
    print(f'Cotamination count fraction {cfar} | should be: {contamination_override * depth_override}')

    synthetic_adata.X = synthetic_adata.X + contamination

    count_number_fin = sum(sum(synthetic_adata.X))

    print(f'Sum counts: {count_number_fin} should be equal to {total_counts_orig} * {per_experiment_depth[0][0]} = '
          f'{total_counts_orig * per_experiment_depth[0][0]}')

    print(f'Total count fraction: {count_number_fin / (total_counts_orig * depth_override)}')

    synthetic_adata.X = csr_matrix(synthetic_adata.X)
    # synthetic_adata.X = synthetic_adata.X.astype(int)

    return synthetic_adata
