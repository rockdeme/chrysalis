import time
import torch
import pandas as pd
import scanpy as sc
from GraphST import GraphST


start_time = time.time()

data_path =  '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/cell2loc_human_lymph_node/'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

adata = sc.read_h5ad(data_path + 'chr.h5ad')
adata.var_names_make_unique()

# define model
model = GraphST.GraphST(adata, device=torch.device('cpu'))

# train model
adata = model.train()

pd.DataFrame(adata.obsm['emb']).to_csv(data_path + 'graphst_lymph_node.csv')

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
