import os
import json
import hashlib
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from itertools import product
from tissue_generator import generate_synthetic_data


#%%
# tabula sapiens immune
adata = sc.read_h5ad(f'data/tabula_sapiens_immune_subsampled_26k.h5ad')
adata.X = adata.raw.X

# do some QC
adata.var["mt"] = adata.var['feature_name'].str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 20, :]

# remove cell types with less than 100 cells
cell_num = pd.value_counts(adata.obs['cell_type'])
low_cell_abundance = cell_num[cell_num < 100]
adata = adata[~adata.obs['cell_type'].isin(low_cell_abundance.index)].copy()

# remove sparse genes
sc.pp.filter_genes(adata, min_cells=10)

cell_types = np.unique(adata.obs['cell_type'])
print(f'Number of cell types: {len(cell_types)}')
print(cell_types)

#%%
param_dict = {
    'annot_col': ['cell_type'],  # annotation column containing the cell type categories
    # 'seed': [*range(1)],  # generate different samples with seed
    'seed': [42],

    # tissue zone generation
    'n_tissue_zones': [6, 10, 14],  # number of tissue zones
    'uniform': [0.0, 0.5, 0.2, 0.8],  # fraction of uniform zones from the total
    'gp_mean': [8, 4, 10],  # granularity 4-8-10 looks good
    'gp_eta_true': [1.5],  # controls gradient steepness we can leave this at 1.5. 1.0-2.5 can be
                           # fine if we want to tune this
    'gp_eta_true_uniform': [0.5], # same as above just with lower values

    # cell type abundance assignment
    'cell_type_p': [0.02, 0.04], # tune number of cell types per non-uniform tissue zone
    'mu_low_density': [3],  # tune average abundance - 3 is around 6 cells/spot/cell type, 5 like 10-15.
                            # Plot the histo with
    'mu_high_density': [5],  # the distribution for more info
    'p_high_density': [0.5],  # how many cell types are high density vs low density

    # confounder parameters
    'mu_detection': [5],  # detection rate shape to multiply the counts with
    'mu_contamination': [0.03],  # contamination shape for adding random counts to each gene per location
    'mu_depth': [1],  # sequencing depth shape to multiply the counts with per location
    'depth_loc_mean_var_ratio': [25],  # sequencing depth shape
    'mu_depth_exp': [1],  # sample wise sequencing depth shape - single value drawn to multiply everything with
    'depth_mean_var_ratio_exp': [5],  # sample wise sequencing depth scale
}

#%%
filepath = 'data/tabula_sapiens_immune'

# get the total number of combinations quickly
param_combinations = product(*param_dict.values())
num_combinations = len([x for x  in param_combinations])

# reset the iterator
param_combinations = product(*param_dict.values())
print(f'Number of samples generated: {num_combinations}')

#%%
for parameters in tqdm(param_combinations, total=num_combinations):
    settings = {k:v for k, v in zip(param_dict.keys(), parameters)}
    print(settings)

    # check if folder already exists
    sample_hash = hashlib.sha256(json.dumps(settings).encode('utf-8')).hexdigest()[:12]
    folder = filepath + '/' + sample_hash + '/'

    if not os.path.isdir(folder):
        try:
            generate_synthetic_data(adata, *parameters, parameter_dict=settings, save=True, root_folder=folder)
        except AssertionError as e:
            print(f'AssertionError: {e}')
    else:
        pass

# %%
# varying contamination and sequencing depth - CONTAMINATION AND DEPTH OVERRIDES

param_dict = {
    'annot_col': ['cell_type'],  # annotation column containing the cell type categories
    'seed': [37, 42, 69],

    # spatial pattern type
    'pattern_type': ['gp'],

    # tissue zone generation
    'n_tissue_zones': [6],  # number of tissue zones
    'uniform': [0.0],  # fraction of uniform zones from the total
    'gp_mean': [8],  # granularity 4-8-10 looks good
    'gp_eta_true': [1.5],
    # controls gradient steepness we can leave this at 1.5. 1.0-2.5 can be fine if we want to tune this
    'gp_eta_true_uniform': [0.5],  # same as above just with lower values
    # r_d
    'r_d_mean': [5],
    'r_d_mean_var_ratio': [0.3],

    # cell type abundance assignment
    'cell_type_p': [0.02],  # tune number of cell types per non-uniform tissue zone
    'mu_low_density': [3],
    # tune average abundance - 3 is around 6 cells/spot/cell type, 5 like 10-15. Plot the histo with
    'mu_high_density': [5],  # the distribution for more info
    'p_high_density': [0.5],  # how many cell types are high density vs low density

    # confounder parameters
    'mu_detection': [5],  # detection rate shape to multiply the counts with
    'mu_contamination': [0.03],  # contamination shape for adding random counts to each gene per location
    'mu_depth': [1],  # sequencing depth shape to multiply the counts with per location
    'depth_loc_mean_var_ratio': [25],  # sequencing depth shape
    'mu_depth_exp': [1],  # sample wise sequencing depth shape - single value drawn to multiply everything with
    'depth_mean_var_ratio_exp': [5],  # sample wise sequencing depth scale

    # direct overrides for contamination and depth to make them more consistent
    'contamination_override': [0.03, 0.09, 0.27, 0.81],  # 0.03 * (3)^n
    'depth_override': [1, 0.2, 0.04, 0.008, 0.0016, 0.00032],  # 1.0 * (0.2)^n
}

#%%
filepath = 'data/tabula_sapiens_immune_contamination'

# get the total number of combinations quickly
param_combinations = product(*param_dict.values())
num_combinations = len([x for x  in param_combinations])

# reset the iterator
param_combinations = product(*param_dict.values())
print(f'Number of samples generated: {num_combinations}')

#%%
for parameters in tqdm(param_combinations, total=num_combinations):
# for parameters in param_combinations:
    settings = {k:v for k, v in zip(param_dict.keys(), parameters)}
    print(settings)

    # check if folder already exists
    sample_hash = hashlib.sha256(json.dumps(settings).encode('utf-8')).hexdigest()[:12]
    folder = filepath + '/' + sample_hash + '/'

    if not os.path.isdir(folder):
        try:
            generate_synthetic_data(adata, *parameters, parameter_dict=settings, save=True, root_folder=folder,
                                    confounder_version='v2')
        except AssertionError as e:
            print(f'AssertionError: {e}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
    else:
        pass
