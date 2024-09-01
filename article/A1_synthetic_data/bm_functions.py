import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr


def get_correlation_df(xdf, ydf):
    corr_matrix = np.empty((len(xdf.columns), len(ydf.columns)))
    for i, col1 in enumerate(xdf.columns):
        for j, col2 in enumerate(ydf.columns):
            corr, _ = pearsonr(xdf[col1], ydf[col2])
            corr_matrix[i, j] = corr

    corr_df = pd.DataFrame(data=corr_matrix,
                           index=xdf.columns,
                           columns=ydf.columns).T
    return corr_df


def collect_correlation_results(list_csv_path, enforce_symmetry=False):
    corr_dict = {}
    for idx, fp in enumerate(list_csv_path):
        corr_df = pd.read_csv(fp, index_col=0)
        corr_df = corr_df[[c for c in corr_df.columns if 'uniform' not in c]]
        if enforce_symmetry:
            col_num = corr_df.shape[1]
            corr_df = corr_df.iloc[:col_num, :]
        # get the max correlations and add it to the results_df
        max_corr = corr_df.max(axis=0)
        corr_dict[idx] = list(max_corr.values)
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in corr_dict.items()]))
    return df


def collect_metadata(list_adata_path):
    meta_dict = {}
    for idx, fp in enumerate(list_adata_path):
        print(fp)
        adata = sc.read_h5ad(fp)
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars=['mt'])

        # add count metrics
        total_counts = np.sum(adata.obs['total_counts'])
        med_counts = np.median(adata.obs['total_counts'])
        n_genes = np.median(adata.obs['n_genes_by_counts'])

        # add sample hash - can cause errors if we change the location of the hash in the sample name
        adata.uns['parameters']['hash'] = fp.split('/')[-1].split('_')[0]
        meta_dict[idx] = adata.uns['parameters']

        meta_dict[idx]['total_counts'] = total_counts
        meta_dict[idx]['median_counts'] = med_counts
        meta_dict[idx]['median_n_genes'] = n_genes

    meta_df = pd.DataFrame(meta_dict).T
    # meta_df = meta_df.drop(columns=['annot_col'])
    meta_df = meta_df.convert_dtypes()
    return meta_df
