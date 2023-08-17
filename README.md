<p align="center">
   <img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/misc/banner.png" width="850">
</p>

**chrysalis** is a spatial domain detection and visualization tool that generates neat and meaningful visual 
representations of spatial transcriptomics datasets. It achieves this by leveraging archetypal analysis and 
spatially variable gene detection. Moreover, it seamlessly integrates into `scanpy` based pipelines.

<p align="center">
   <img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/misc/panel_1.png" width=800">

</p>

**chrysalis** can define distinct tissue compartments and cellular niches, which can be 
highlighted with specific colors. For instance, in the `V1_Human_Lymph_Node` dataset, **chrysalis** can identify and 
highlight various regions, such as germinal centers (yellow), B cell follicles (dark orange), and T cell compartments 
(lime). You can find more examples in the [gallery](https://github.com/rockdeme/chrysalis#gallery) section.

<p align="center">
   <img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/misc/panel_2.png" width="850">
</p>

## Package
**chrysalis** can be used with any pre-existing `anndata` snapshot of 10X Visium, Slide-seqV2 and Stereo-seq datasets 
generated with `scanpy`, and on new samples without the need of preprocessing. It is designed to be as lightweight as 
possible, however currently it relies on `libpysal` for its fast implementation of Moran's I.

**chrysalis** requires the following packages:
- numpy
- pandas
- matplotlib
- scanpy
- pysal
- archetypes
- scikit_learn
- scipy
- tqdm
- seaborn

To install **chrysalis**:
```terminal
pip install chrysalis-st
```

## Usage

```python
import chrysalis as ch
import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

ch.detect_svgs(adata)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

ch.pca(adata)

ch.aa(adata, n_pcs=20, n_archetypes=8)

ch.plot(adata)
plt.show()
```

## Documentation and API details

User documentation is available at: https://chrysalis.readthedocs.io/

## Gallery

<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Mouse_Brain_Sagittal_Anterior.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Mouse_Brain_Sagittal_Posterior.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Human_Lymph_Node.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Mouse_Kidney.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Breast_Cancer_Block_A_Section_1.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Breast_Cancer_Block_A_Section_2.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Human_Heart.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Adult_Mouse_Brain.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Mouse_Brain_Sagittal_Posterior_Section_2.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Mouse_Brain_Sagittal_Anterior_Section_2.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Adult_Mouse_Brain_Coronal_Section_1.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/V1_Adult_Mouse_Brain_Coronal_Section_2.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/Parent_Visium_Human_Cerebellum.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/Parent_Visium_Human_Glioblastoma.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/Parent_Visium_Human_BreastCancer.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/Parent_Visium_Human_OvarianCancer.png" width="500">
<img src="https://raw.githubusercontent.com/rockdeme/chrysalis/master/plots/gallery/Parent_Visium_Human_ColorectalCancer.png" width="500">
