import scanpy as sc
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.pyplot as plt
from functions import chrysalis_calculate, chrysalis_plot
import numpy as np


plots_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/plots/'
data_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/processed/'

samples = ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior", 'V1_Human_Brain_Section_1',
           'V1_Human_Brain_Section_2', 'V1_Human_Lymph_Node', 'V1_Mouse_Kidney']

for sample in tqdm(samples):
    adata = sc.datasets.visium_sge(sample_id=sample)
    chrysalis_calculate(adata)
    adata.write_h5ad(data_path + f'{sample}.h5ad')


for sample in tqdm(samples):
    adata = sc.read_h5ad(data_path + f'{sample}.h5ad')
    chrysalis_plot(adata, pcs=8)
    # plt.savefig(plots_path + f'L2210915.svg')
    plt.show()
