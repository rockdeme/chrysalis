import scanpy as sc
import chrysalis as ch
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import pytest

def load_sample(sample_id='V1_Human_Lymph_Node'):
    adata = sc.datasets.visium_sge(sample_id=sample_id)
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.filter_cells(adata, min_counts=6000)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    return adata

def save_plot(name=None):
    save = os.getenv('SAVE_PLOTS', '0') == '1'
    if save:
        if name:
            if not os.path.isdir('temp/plots/'):
                os.makedirs('temp/plots/', exist_ok=True)
            plt.savefig(f'temp/plots/{name}.png')
        else:
            raise ValueError('No plot name specified.')
    else:
        plt.clf()

@pytest.fixture(scope='module')
def prepared_adata():
    adata = load_sample(sample_id='V1_Human_Lymph_Node')
    ch.compute_svg_scores(adata)
    ch.select_svgs(adata, top_svg=1000)
    ch.pca(adata)
    ch.aa(adata, n_pcs=20, n_archetypes=8)
    return adata

def test_svg_detection(prepared_adata):
    svg_col = prepared_adata.var.get("spatially_variable", None)
    assert svg_col is not None
    assert isinstance(svg_col.values[0], (bool, np.bool_))

def test_aa(prepared_adata):
    assert prepared_adata.obsm['chr_aa'] is not None

def test_plot_weights(prepared_adata):
    ch.plot(prepared_adata, dim=8)
    save_plot('spatial_plot')





# samples = ['V1_Mouse_Brain_Sagittal_Anterior_Section_2', 'V1_Mouse_Brain_Sagittal_Posterior_Section_2']
#
#
#
#
#
#
#
#
# def save_plot(plot_save, name=None):
#     if plot_save:
#         # plt.show()
#         if isinstance(name, str):
#             if not os.path.isdir('temp/plots/'):
#                 os.makedirs('temp/plots/', exist_ok=True)
#             plt.savefig(f'temp/plots/{name}.png')
#         else:
#             raise ValueError('No plot name specified.')
#     else:
#         plt.clf()
#
#
# def test_single_sample(save=True):
#
#     adata = preprocess_sample()
#
#     # normalization
#     sc.pp.normalize_total(adata, inplace=True)
#     sc.pp.log1p(adata)
#
#     ch.pca(adata)
#
#     ch.plot_svgs(adata)
#     print(os.getcwd())
#     save_plot(save, name='singleplot_svg')
#
#     ch.plot_explained_variance(adata)
#     save_plot(save, name='singleplot_evr')
#
#     ch.aa(adata, n_pcs=20, n_archetypes=8)
#
#     ch.plot(adata)
#     save_plot(save, name='singleplot_plot')
#
#     ch.plot_compartments(adata)
#     save_plot(save, name='singleplot_comps')
#
#     ch.plot_heatmap(adata)
#     save_plot(save, name='singleplot_heatmap')
#
#     ch.plot_weights(adata)
#     save_plot(save, name='singleplot_weights')
#
#
# def test_multi_sample_harmony(save=True):
#
#     adatas = preprocess_multisample()
#
#     # concatenate samples
#     adata = ch.integrate_adatas(adatas, sample_col='sample')
#     # replace ENSEMBL IDs with the gene symbols and make them unique
#     adata.var_names = adata.var['gene_symbols']
#     adata.var_names_make_unique()
#     # harmony
#     ch.pca(adata, n_pcs=50)
#     ch.harmony_integration(adata, 'sample', random_state=42, block_size=0.05)
#
#     ch.aa(adata, n_pcs=20, n_archetypes=10)
#
#     ch.plot_samples(adata, 1, 2, dim=10, suptitle='test')
#     save_plot(save, name='multiplot_mip_harmony')
#
#     ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', selected_comp=0)
#     save_plot(save, name='multiplot_single_harmony')
#
#
# def test_multi_sample_scanorama(save=True):
#
#     adatas = preprocess_multisample()
#
#     # scanorama
#     adatas_cor = scanorama.correct_scanpy(adatas, return_dimred=True)
#     # concatenate samples
#     adata = ch.integrate_adatas(adatas_cor, sample_col='sample')
#     # replace ENSEMBL IDs with the gene symbols and make them unique
#     adata.var_names = adata.var['gene_symbols']
#     adata.var_names_make_unique()
#
#     ch.pca(adata, n_pcs=50)
#     ch.aa(adata, n_pcs=20, n_archetypes=10)
#
#     ch.plot_samples(adata, 1, 2, dim=10, suptitle='test')
#     save_plot(save, name='multiplot_mip_scanorama')
#
#     ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', selected_comp=0)
#     save_plot(save, name='multiplot_single_scanorama')
#
# def test_multi_sample_plots(save=True):
#     adatas = preprocess_multisample()
#
#     # concatenate samples
#     adata = ch.integrate_adatas(adatas, sample_col='sample')
#     # replace ENSEMBL IDs with the gene symbols and make them unique
#     adata.var_names = adata.var['gene_symbols']
#     adata.var_names_make_unique()
#
#     ch.pca(adata, n_pcs=50)
#     ch.aa(adata, n_pcs=20, n_archetypes=10)
#
#     ch.plot_svg_matrix(adatas, figsize=(8, 7), obs_name='sample', cluster=True)
#     save_plot(save, name='multiplot_svg_matrix')
#
#     ch.plot_samples(adata, 1, 2, dim=10, suptitle='test')
#     save_plot(save, name='multiplot_mip')
#
#     ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', selected_comp=0)
#     save_plot(save, name='multiplot_single')
#
#
# if __name__ == '__main__':
#     save=True
#
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(script_directory)
#
#     print(f'Temporary files and plots are saved to the following directory: {script_directory}')
#
#     print('Running test_single_sample...')
#     test_single_sample(save=True)
#     print('Test completed!')
#     print('Running test_multi_sample_harmony...')
#     test_multi_sample_harmony(save=True)
#     print('Test completed!')
#     print('Running test_multi_sample_scanorama...')
#     test_multi_sample_scanorama(save=True)
#     print('Test completed!')
#     print('Running test_multi_sample_plots...')
#     test_multi_sample_plots(save=True)
#     print('Test completed!')
#     print('------------------------------')
#     print('All tests have been completed!')
