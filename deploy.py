import scanpy as sc
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import chrysalis_calculate, chrysalis_plot


linux = False
if linux:
    plots_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/plots/'
    data_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/processed/'
else:
    plots_path = 'C:/Users/Deme/PycharmProjects/chrysalis/plots/'
    data_path = 'C:/Users/Deme/PycharmProjects/chrysalis/data/processed/'

samples = ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior",
           'V1_Human_Lymph_Node', 'V1_Mouse_Kidney']

for sample in tqdm(samples):
    adata = sc.datasets.visium_sge(sample_id=sample)

    chrysalis_calculate(adata)
    adata.write_h5ad(data_path + f'{sample}.h5ad')
    chrysalis_plot(adata, dim=8, mode='aa')
    plt.show()


for sample in tqdm(samples):
    adata = sc.read_h5ad(data_path + f'{sample}.h5ad')
    # chrysalis_plot(adata, pcs=8)
    chrysalis_plot(adata, dim=8, mode='pca')
    # plt.savefig(plots_path + f'{sample}_aa.svg')
    plt.show()

extended_samples = ['V1_Breast_Cancer_Block_A_Section_1', 'V1_Breast_Cancer_Block_A_Section_2', 'V1_Human_Heart',
                    'V1_Adult_Mouse_Brain', 'V1_Mouse_Brain_Sagittal_Posterior_Section_2',
                    'V1_Mouse_Brain_Sagittal_Anterior_Section_2', 'V1_Adult_Mouse_Brain_Coronal_Section_1',
                    'V1_Adult_Mouse_Brain_Coronal_Section_2',
                    'Parent_Visium_Human_Cerebellum',
                    'Parent_Visium_Human_Glioblastoma',
                    'Parent_Visium_Human_BreastCancer',
                    'Parent_Visium_Human_OvarianCancer',
                    'Parent_Visium_Human_ColorectalCancer']

for sample in tqdm(extended_samples):
    adata = sc.datasets.visium_sge(sample_id=sample)

    chrysalis_calculate(adata)
    adata.write_h5ad(data_path + f'{sample}.h5ad')

    chrysalis_plot(adata, dim=8, mode='aa')
    plt.show()

for sample in tqdm(samples + extended_samples):
    adata = sc.read_h5ad(data_path + f'{sample}.h5ad')
    # chrysalis_plot(adata, pcs=8)
    chrysalis_plot(adata)
    plt.title(f'{sample}\n', fontsize=10)
    plt.savefig(plots_path + f'gallery/{sample}.svg')
    plt.savefig(plots_path + f'gallery/{sample}.png')
    plt.show()

for sample in tqdm(samples):
    adata = sc.read_h5ad(data_path + f'{sample}.h5ad')
    # chrysalis_plot(adata, pcs=8)
    chrysalis_plot(adata)
    # plt.title(f'{sample}\n', fontsize=10)
    plt.savefig(plots_path + f'{sample}.svg')
    plt.savefig(plots_path + f'{sample}.png')
    plt.show()
