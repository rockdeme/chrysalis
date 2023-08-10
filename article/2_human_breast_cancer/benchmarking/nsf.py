import time
import pickle
import random
import numpy as np
import scanpy as sc
from os import path
import matplotlib.pyplot as plt
from tensorflow_probability import math as tm
from nsf.models import cf, sf, sfh
from nsf.utils import preprocess, training, misc, visualize, postprocess


tfk = tm.psd_kernels

data_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/xenium_human_breast_cancer/'

ad = sc.read_h5ad(data_path + 'visium_sample.h5ad')
ad.var_names_make_unique()

sc.pp.calculate_qc_metrics(ad, inplace=True)
sc.pp.filter_cells(ad, min_counts=1000)
sc.pp.filter_genes(ad, min_cells=10)

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
sc.pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
sc.pp.log1p(ad)

# normalization, feature selection and train/test split
ad.var['deviance_poisson'] = preprocess.deviancePoisson(ad.layers["counts"])
o = np.argsort(-ad.var['deviance_poisson'])
idx = list(range(ad.shape[0]))
random.shuffle(idx)
ad = ad[idx,o]
ad = ad[:,:2000]

Dtr,Dval = preprocess.anndata_to_train_val(ad,layer="counts",sz="scanpy")
Dtr_n,Dval_n = preprocess.anndata_to_train_val(ad) #normalized data
fmeans,Dtr_c,Dval_c = preprocess.center_data(Dtr_n,Dval_n) #centered features
Xtr = Dtr["X"] #note this should be identical to Dtr_n["X"]
Ntr = Xtr.shape[0]
Dtf = preprocess.prepare_datasets_tf(Dtr,Dval=Dval,shuffle=False)
Dtf_n = preprocess.prepare_datasets_tf(Dtr_n,Dval=Dval_n,shuffle=False)
Dtf_c = preprocess.prepare_datasets_tf(Dtr_c,Dval=Dval_c,shuffle=False)
visualize.heatmap(Xtr,Dtr["Y"][:,0],marker="D",s=15)
plt.show()

# Visualize raw data
plt.imshow(np.log1p(Dtr["Y"])[:50,:100],cmap="Blues")
plt.show()

# Visualize inducing points
Z = misc.kmeans_inducing_pts(Xtr,500)
fig,ax=plt.subplots(figsize=(12,10))
ax.scatter(Xtr[:,0],Xtr[:,1],marker="D",s=50,)
ax.scatter(Z[:,0],Z[:,1],c="red",s=30)
plt.show()

# initialize inducing points and tuning parameters
Z = misc.kmeans_inducing_pts(Xtr, 2363)
M = Z.shape[0]
ker = tfk.MaternThreeHalves
S = 3 #samples for elbo approximation
# NSF: Spatial only with non-negative factors
L = 8 #number of latent factors, ideally divisible by 2
J = 2000

mpth = path.join("/mnt/c/Users/demeter_turos/PycharmProjects/deep_learning/nsf/models/V6")

fit = sf.SpatialFactorization(J,L,Z,psd_kernel=ker,nonneg=True,lik="poi")
fit.elbo_avg(Xtr,Dtr["Y"],sz=Dtr["sz"])
fit.init_loadings(Dtr["Y"],X=Xtr,sz=Dtr["sz"])
fit.elbo_avg(Xtr,Dtr["Y"],sz=Dtr["sz"])
pp = fit.generate_pickle_path("scanpy",base=mpth)
tro = training.ModelTrainer(fit,pickle_path=pp)
tro.train_model(*Dtf)
ttl = "NSF: spatial, non-negative factors, Poisson likelihood"
visualize.plot_loss(tro.loss,title=ttl)#,ss=range(2000,4000))
plt.show()

# Postprocessing

hmkw = {"figsize":(4,4), "s":0.3, "marker":"D", "subplot_space":0,
        "spinecolor":"white"}
insf = postprocess.interpret_nsf(fit,Xtr,S=10,lda_mode=False)
tgnames = [str(i) for i in range(1,L+1)]
fig,axes=visualize.multiheatmap(Xtr, np.sqrt(insf["factors"]), (4,3), **hmkw)
visualize.set_titles(fig, tgnames, x=0.05, y=.85, fontsize="medium", c="white",
                     ha="left", va="top")
plt.show()

file = open(mpth + '/human_breast_cancer_nsf.pkl', 'wb')
pickle.dump({'factors': insf, 'positions': Xtr}, file)
file.close()

file = open(mpth + '/human_breast_cancer_nsf.pkl', 'rb')
data = pickle.load(file)
file.close()