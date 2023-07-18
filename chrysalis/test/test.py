import scanpy as sc
import chrysalis as ch
import matplotlib.pyplot as plt


def test_chrysalis(show=False):

    def show_plot(s):
        if s:
            plt.show()
        else:
            plt.clf()

    adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.filter_cells(adata, min_counts=6000)
    sc.pp.filter_genes(adata, min_cells=10)

    ch.detect_svgs(adata)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    ch.pca(adata)

    ch.plot_svgs(adata)
    show_plot(show)

    ch.plot_explained_variance(adata)
    show_plot(show)

    ch.aa(adata, n_pcs=20, n_archetypes=8)

    ch.plot(adata)
    show_plot(show)

    ch.plot_compartments(adata)
    show_plot(show)

    ch.plot_heatmap(adata)
    show_plot(show)

    ch.plot_weights(adata)
    show_plot(show)
