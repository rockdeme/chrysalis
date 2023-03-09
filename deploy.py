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

samples = ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior", 'V1_Human_Brain_Section_1',
           'V1_Human_Brain_Section_2', 'V1_Human_Lymph_Node', 'V1_Mouse_Kidney']

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
                    'V1_Adult_Mouse_Brain_Coronal_Section_2', 'Targeted_Visium_Human_Cerebellum_Neuroscience',
                    'Parent_Visium_Human_Cerebellum', 'Targeted_Visium_Human_SpinalCord_Neuroscience',
                    'Parent_Visium_Human_SpinalCord', 'Targeted_Visium_Human_Glioblastoma_Pan_Cancer',
                    'Parent_Visium_Human_Glioblastoma', 'Targeted_Visium_Human_BreastCancer_Immunology',
                    'Parent_Visium_Human_BreastCancer', 'Targeted_Visium_Human_OvarianCancer_Pan_Cancer',
                    'Targeted_Visium_Human_OvarianCancer_Immunology', 'Parent_Visium_Human_OvarianCancer',
                    'Targeted_Visium_Human_ColorectalCancer_GeneSignature', 'Parent_Visium_Human_ColorectalCancer']

for sample in tqdm(extended_samples):
    adata = sc.datasets.visium_sge(sample_id=sample)

    chrysalis_calculate(adata)
    adata.write_h5ad(data_path + f'{sample}.h5ad')
    chrysalis_plot(adata, dim=8, mode='aa')
    plt.show()