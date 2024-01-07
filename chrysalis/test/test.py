import scanpy as sc
import chrysalis as ch
import matplotlib.pyplot as plt
import scanorama
import os
from glob import glob


def preprocess_sample():
    # preprocessing samples
    print('Preprocessing...')

    if not os.path.isdir('temp/'):
        os.makedirs('temp/', exist_ok=True)

    try:
        adata = sc.read_h5ad('temp/V1_Human_Lymph_Node_ss.h5ad')
    except FileNotFoundError:
            adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sc.pp.filter_cells(adata, min_counts=6000)
            sc.pp.filter_genes(adata, min_cells=10)

            ch.detect_svgs(adata)

            adata.write_h5ad(f'temp/V1_Human_Lymph_Node_ss.h5ad')

    return adata


def preprocess_multisample():
    # preprocessing samples
    print('Preprocessing...')
    samples = ['V1_Mouse_Brain_Sagittal_Anterior_Section_2', 'V1_Mouse_Brain_Sagittal_Posterior_Section_2']
    adatas = []

    if not os.path.isdir('temp/'):
        os.makedirs('temp/', exist_ok=True)

    files = glob('temp/*_ms.h5ad')
    adatas = [sc.read_h5ad(x) for x in files]

    if len(adatas) == 0:
        for sample in samples:
            ad = sc.datasets.visium_sge(sample_id=sample)
            ad.var_names_make_unique()
            sc.pp.calculate_qc_metrics(ad, inplace=True)
            sc.pp.filter_cells(ad, min_counts=1000)
            sc.pp.filter_genes(ad, min_cells=10)
            sc.pp.normalize_total(ad, inplace=True)
            sc.pp.log1p(ad)

            ch.detect_svgs(ad, min_morans=0.05, min_spots=0.05)
            ad.write_h5ad(f'temp/{sample}_ms.h5ad')

            adatas.append(ad)

    return adatas


def save_plot(plot_save, name=None):
    if plot_save:
        # plt.show()
        if isinstance(name, str):
            if not os.path.isdir('temp/plots/'):
                os.makedirs('temp/plots/', exist_ok=True)
            plt.savefig(f'temp/plots/{name}.png')
        else:
            raise ValueError('No plot name specified.')
    else:
        plt.clf()


def test_single_sample(save=True):

    adata = preprocess_sample()

    # normalization
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    ch.pca(adata)

    ch.plot_svgs(adata)
    print(os.getcwd())
    save_plot(save, name='singleplot_svg')

    ch.plot_explained_variance(adata)
    save_plot(save, name='singleplot_evr')

    ch.aa(adata, n_pcs=20, n_archetypes=8)

    ch.plot(adata)
    save_plot(save, name='singleplot_plot')

    ch.plot_compartments(adata)
    save_plot(save, name='singleplot_comps')

    ch.plot_heatmap(adata)
    save_plot(save, name='singleplot_heatmap')

    ch.plot_weights(adata)
    save_plot(save, name='singleplot_weights')


def test_multi_sample_harmony(save=True):

    adatas = preprocess_multisample()

    # concatenate samples
    adata = ch.integrate_adatas(adatas, sample_col='sample')
    # replace ENSEMBL IDs with the gene symbols and make them unique
    adata.var_names = adata.var['gene_symbols']
    adata.var_names_make_unique()
    # harmony
    ch.pca(adata, n_pcs=50)
    ch.harmony_integration(adata, 'sample', random_state=42, block_size=0.05)

    ch.aa(adata, n_pcs=20, n_archetypes=10)

    ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', spot_size=4.5)
    save_plot(save, name='multiplot_mip_harmony')

    ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', selected_comp=0)
    save_plot(save, name='multiplot_single_harmony')


def test_multi_sample_scanorama(save=True):

    adatas = preprocess_multisample()

    # scanorama
    adatas_cor = scanorama.correct_scanpy(adatas, return_dimred=True)
    # concatenate samples
    adata = ch.integrate_adatas(adatas_cor, sample_col='sample')
    # replace ENSEMBL IDs with the gene symbols and make them unique
    adata.var_names = adata.var['gene_symbols']
    adata.var_names_make_unique()

    ch.pca(adata, n_pcs=50)
    ch.aa(adata, n_pcs=20, n_archetypes=10)

    ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', spot_size=4.5)
    save_plot(save, name='multiplot_mip_scanorama')

    ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', selected_comp=0)
    save_plot(save, name='multiplot_single_scanorama')

def test_multi_sample_plots(save=True):
    adatas = preprocess_multisample()

    # concatenate samples
    adata = ch.integrate_adatas(adatas, sample_col='sample')
    # replace ENSEMBL IDs with the gene symbols and make them unique
    adata.var_names = adata.var['gene_symbols']
    adata.var_names_make_unique()

    ch.pca(adata, n_pcs=50)
    ch.aa(adata, n_pcs=20, n_archetypes=10)

    ch.plot_svg_matrix(adatas, figsize=(8, 7), obs_name='sample', cluster=True)
    save_plot(save, name='multiplot_svg_matrix')

    ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', spot_size=4.5)
    save_plot(save, name='multiplot_mip')

    ch.plot_samples(adata, 1, 2, dim=10, suptitle='test', selected_comp=0)
    save_plot(save, name='multiplot_single')


if __name__ == '__main__':
    save=True

    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)

    test_single_sample(save=True)
    test_multi_sample_harmony(save=True)
    test_multi_sample_scanorama(save=True)
    test_multi_sample_plots(save=True)