import torch
import pandas as pd
import scanpy as sc
from GraphST import GraphST


data_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/xenium_human_breast_cancer/'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

adata = sc.read_h5ad(data_path + 'visium_sample.h5ad')
adata.var_names_make_unique()

# define model
model = GraphST.GraphST(adata, device=torch.device('cpu'))
# train model
adata = model.train()

pd.DataFrame(adata.obsm['emb']).to_csv(data_path + 'graphst_breast_cancer.csv')
