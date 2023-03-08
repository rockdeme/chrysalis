import scanpy as sc
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import chrysalis_calculate, chrysalis_plot


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

hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']

np.random.seed(len(adata))
np.random.shuffle(hexcodes)

loadings = pd.DataFrame(adata.varm['PCs'][:, :20], index=adata.var_names)
sl = loadings[[0]].sort_values(ascending=False, by=0)[:10]

fig, ax = plt.subplots(2, 4, figsize=(3 * 4, 4 * 2))
ax = ax.flatten()
for i in range(8):
    sl = loadings[[i]].sort_values(ascending=False, by=i)[:10]
    ax[i].axis('on')
    ax[i].grid(axis='x')
    ax[i].set_axisbelow(True)
    ax[i].barh(list(sl.index)[::-1], list(sl[i].values)[::-1], color=hexcodes[i])
    ax[i].set_xlabel('Loading')
    ax[i].set_title(f'PC {i}')
    # ax.set_aspect('equal')
plt.tight_layout()
plt.show()


plt.grid(axis='y', zorder=0)
ax.set_axisbelow(True)
ax.set_ylabel('Raw intensity')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel('ROI')
plt.title('Sum Pt Intensity')