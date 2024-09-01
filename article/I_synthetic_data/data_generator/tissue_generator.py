import os
import math
import json
import hashlib
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from functions import plot_spatial, return_cell_types
from tools import (generate_tissue_zones, assign_cell_types, assign_cell_type_abundance,
                   construct_cell_abundance_matrix, generate_synthetic_counts, construct_adata,
                   add_confounders, add_confounders_v2)


def generate_synthetic_data(adata, annot_col, seed, pattern_type, n_tissue_zones, uniform, gp_mean, gp_eta_true,
                            gp_eta_true_uniform, r_d_mean, r_d_mean_var_ratio, cell_type_p, mu_low_density,
                            mu_high_density, p_high_density, mu_detection, mu_contamination, mu_depth,
                            depth_loc_mean_var_ratio, mu_depth_exp, depth_mean_var_ratio_exp, contamination_override,
                            depth_override, parameter_dict, save=False, root_folder=None, confounder_version='v1'):

    # some checks before running the generation
    cell_types, n_cell_types = return_cell_types(adata, annot_col)
    n_uniform = int(n_tissue_zones * uniform)
    n_sparse_ct = n_cell_types - n_uniform
    n_sparse_tz = n_tissue_zones - n_uniform
    # assert n_sparse_ct >= n_sparse_tz, (f"Sparse tissue zones = {n_sparse_tz} ({n_tissue_zones}-{n_uniform}) which is"
    #                                     f"higher than the number of cell types ({n_sparse_ct}) ({n_cell_types}-"
    #                                     f"{n_uniform}). Either decrease the "
    #                                     f"overall number of tissue zones or the fraction of uniform zones.")
    assert n_uniform < n_cell_types

    if type(parameter_dict) != dict:
        raise Warning("parameter_dict is not a dictionary!")
    sample_hash = hashlib.sha256(json.dumps(parameter_dict).encode('utf-8')).hexdigest()[:12]
    if save:
        assert root_folder is not None
        os.makedirs(root_folder, exist_ok=True)
        assert os.path.isdir(root_folder) == True

    # generate tissue zones with gaussian processes or reaction-diffusion
    abundance_df, locs = generate_tissue_zones(grid_size=(50, 50),
                                               n_tissue_zones=n_tissue_zones,
                                               unfiform_fraction=uniform,
                                               pattern_type=pattern_type, gp_mean=gp_mean, gp_eta_true=gp_eta_true,
                                               gp_eta_true_uniform=gp_eta_true_uniform,
                                               r_d_mean=r_d_mean, seed=seed, r_d_mean_var_ratio=r_d_mean_var_ratio)

    # look at the tissue zones

    nrows = math.ceil((n_tissue_zones + 1) / 5)

    plt.figure(figsize=(3 * 5, 3 * nrows))
    plot_spatial(abundance_df.values, n=(50, 50), nrows=nrows, names=abundance_df.columns, vmax=None)
    plt.tight_layout()
    if save:
        plt.savefig(root_folder + 'tissue_zones.png')
    plt.show()

    # assign cell types to tissue zones
    cell_types_df = assign_cell_types(adata, n_tissue_zones, annot_col=annot_col, unfiform_fraction=uniform,
                                      cell_type_p=cell_type_p, seed=seed)
    # add average abundances
    # mu controls cell density per spot
    cell_types_df = assign_cell_type_abundance(cell_types_df, p_high_density=p_high_density,
                                               mu_low_density=mu_low_density, mu_high_density=mu_high_density,
                                               seed=seed)

    sns.heatmap(cell_types_df, square=True)
    plt.tight_layout()
    if save:
        plt.savefig(root_folder + 'heatmap.png')
    plt.show()

    # multiply the abundances with the average and take the integer values (difference between the two is the
    # capture eff.)
    cell_count_df, capture_eff_df = construct_cell_abundance_matrix(abundance_df, cell_types_df, seed=seed)

    n_cell_types = len(cell_types_df.T.columns)
    nrows = math.ceil((n_cell_types + 1) / 5)

    plt.figure(figsize=(3 * 5, 3 * nrows))
    plot_spatial(cell_count_df.values, n=(50, 50), nrows=nrows, names=cell_types_df.T.columns, vmax=None)
    plt.tight_layout()
    if save:
        plt.savefig(root_folder + 'cell_types.png')
    plt.show()

    synthetic_counts = generate_synthetic_counts(adata, cell_count_df, capture_eff_df, annot_col=annot_col, seed=seed)

    sth_adata = construct_adata(synthetic_counts, adata, abundance_df, locs, cell_count_df, capture_eff_df)

    if confounder_version == 'v1':
        sth_adata = add_confounders(sth_adata,
                                    mu_detection=mu_detection, mu_contamination=mu_contamination, mu_depth=mu_depth,
                                    depth_loc_mean_var_ratio=depth_loc_mean_var_ratio, mu_depth_exp=mu_depth_exp,
                                    depth_mean_var_ratio_exp=depth_mean_var_ratio_exp,
                                    contamination_override=contamination_override, depth_override=depth_override,
                                    seed=seed)
    elif confounder_version == 'v2':
        sth_adata = add_confounders_v2(sth_adata,
                                       mu_detection=mu_detection, mu_contamination=mu_contamination, mu_depth=mu_depth,
                                       depth_loc_mean_var_ratio=depth_loc_mean_var_ratio, mu_depth_exp=mu_depth_exp,
                                       depth_mean_var_ratio_exp=depth_mean_var_ratio_exp,
                                       contamination_override=contamination_override, depth_override=depth_override,
                                       seed=seed)
    else:
        raise KeyError("Invalid confounder_version. It should be either 'v1' or 'v2'.")

    sth_adata.uns['parameters'] = parameter_dict

    sth_adata.var['ensembl_id'] = sth_adata.var_names
    sth_adata.var['gene_symbol'] = adata.var['feature_name']
    sth_adata.var_names = sth_adata.var['gene_symbol'].astype(str)

    sth_adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(sth_adata, qc_vars=['mt'], inplace=True)

    if save:
        sth_adata.write_h5ad(root_folder + f'{sample_hash}_sth_adata.h5ad')

    # sc.pl.highest_expr_genes(sth_adata, n_top=20)
    # sc.pl.spatial(sth_adata, color=["log1p_total_counts", 'MALAT1'], spot_size=2.7)
