import scanpy as sc
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import chrysalis_calculate, chrysalis_plot, chrysalis_plot_aa


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

for sample in tqdm(samples):
    adata = sc.read_h5ad(data_path + f'{sample}.h5ad')
    # chrysalis_plot(adata, pcs=8)
    chrysalis_plot_aa(adata, pcs=8)
    # plt.savefig(plots_path + f'{sample}_aa.svg')
    plt.show()

from scipy.spatial.distance import cdist

distances = cdist(np.column_stack((row, col)), np.column_stack((row, col)))
np.fill_diagonal(distances, np.inf)
width, height = fig.get_size_inches() * fig.dpi
min_distance = np.min(min_distances) * np.sqrt(width**2 + height**2)
