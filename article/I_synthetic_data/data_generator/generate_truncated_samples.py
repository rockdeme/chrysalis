import os
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob


filepath = 'data/tabula_sapiens_immune_contamination'
savepath = 'data/tabula_sapiens_immune_size'
adatas = glob(filepath + '/*/*.h5ad')

for idx, adp in tqdm(enumerate(adatas), total=len(adatas)):
    print(adp)
    sample_folder = '/'.join(adp.split('/')[:-1]) + '/'
    adata = sc.read_h5ad(adp)

    sample_id = sample_folder.split('/')[-2]

    if (adata.uns['parameters']['mu_contamination'] == 0.03) & (adata.uns['parameters']['mu_depth_exp'] == 1):

        # 50 x 50
        os.makedirs(f'{savepath}/{sample_id}-5050/')
        adata.write_h5ad(f'{savepath}/{sample_id}-5050/{sample_id}-5050.h5ad')
        tissue_zones = adata.obsm['tissue_zones']
        n1, n2 = (50, 50)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        img_data = tissue_zones['tissue_zone_0'].values.reshape(n1, n2).T
        im = ax.imshow(img_data, cmap='mako_r', vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

        # 25 x 50
        adata = adata[adata.obsm['spatial'][:, 0] < 50]
        os.makedirs(f'{savepath}/{sample_id}-2550/')
        adata.write_h5ad(f'{savepath}/{sample_id}-2550/{sample_id}-2550.h5ad')
        tissue_zones = adata.obsm['tissue_zones']
        n1, n2 = (25, 50)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        img_data = tissue_zones['tissue_zone_0'].values.reshape(n1, n2).T
        im = ax.imshow(img_data, cmap='mako_r', vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

        # 25 x 25
        adata = adata[adata.obsm['spatial'][:, 1] < 50]
        os.makedirs(f'{savepath}/{sample_id}-2525/')
        adata.write_h5ad(f'{savepath}/{sample_id}-2525/{sample_id}-2525.h5ad')
        tissue_zones = adata.obsm['tissue_zones']
        n1, n2 = (25, 25)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        img_data = tissue_zones['tissue_zone_0'].values.reshape(n1, n2).T
        im = ax.imshow(img_data, cmap='mako_r', vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()
