# Chrysalis Article Readme


The collection of notebooks and scripts found in this folder was utilized for our research article. 
It serves the purpose of demonstrating the data analysis we performed. To replicate the analysis, 
supplementary actions are necessary, such as acquiring the raw data, modifying fixed directory paths, 
, downloading supplementary files from Zenodo (https://doi.org/10.5281/zenodo.8247780), and so on. 
```
.
├── 1_human_lymph_node
│   ├── SVG_detection_methods
│   │   ├── 1_bsp_spatialde_sepal.ipynb
│   │   ├── 2_spark.R
│   │   └── 3_method_comparison.ipynb
│   ├── benchmarking
│   │   ├── graphst.py
│   │   ├── mefisto.py
│   │   ├── nsf.py
│   │   ├── spatialpca.R
│   │   └── stagate.py
│   ├── chrysalis_analysis_and_validation.ipynb
│   └── morans_i.ipynb
├── 2_human_breast_cancer
│   ├── benchmarking
│   │   ├── graphst.py
│   │   ├── mefisto.py
│   │   ├── nsf.py
│   │   ├── spatialpca.R
│   │   └── stagate.py
│   ├── benchmarking.ipynb
│   ├── chrysalis_analysis_and_validation.ipynb
│   └── morphology_integration
│       ├── 1_extract_image_tiles.ipynb
│       ├── 2_autoencoder_training.py
│       └── 3_integrate_morphology.ipynb
├── 3_mouse_brain
│   ├── mouse_brain_analysis.ipynb
│   └── mouse_brain_integration.ipynb
├── 4_slide_seqv2
│   └── slide_seqv2_analysis.ipynb
├── 5_stereo_seq
│   └── stereo_seq_analysis.ipynb
└── readme.md
```

## Chrysalis: decoding tissue compartments in spatial transcriptomics with archetypal analysis

**Authors**: Demeter Túrós, Jelica Vasiljevic, Kerstin Hahn, Sven Rottenberg, and Alberto Valdeolivas

**Abstract**: Dissecting tissue compartments in spatial transcriptomics (ST) remains challenging due 
to limited spatial resolution and dependence on single-cell reference data. We present Chrysalis, a 
novel method to rapidly detect tissue compartments through spatially variable gene (SVG) detection 
and archetypal analysis without external references. We applied Chrysalis on ST datasets originating 
from various species, tissues and technologies and demonstrated state-of-the-art performance in 
identifying cellular niches.

