# Chrysalis Article Readme


This folder contains the notebooks and scripts used for our research article, showcasing the data analysis we conducted.
To recreate the analysis, you'll need to take some extra steps, such as getting the raw data, adjusting directory paths,
and downloading supplementary files from Zenodo (https://doi.org/10.5281/zenodo.8247780).
```
.
├── A1_synthetic_data
│   ├── array_size_benchmark.ipynb
│   ├── bm_functions.py
│   ├── chrysalis_example.ipynb
│   ├── contamination_benchmark.ipynb
│   ├── data_generator
│   │   ├── functions.py
│   │   ├── generate_synthetic_datasets.py
│   │   ├── generate_truncated_samples.py
│   │   ├── tissue_generator.py
│   │   └── tools.py
│   ├── main_synthetic_benchmark.ipynb
│   └── method_scripts
│       ├── array_size_benchmark
│       │   ├── chrysalis.py
│       │   ├── graphst.py
│       │   ├── mefisto.py
│       │   ├── nsf.py
│       │   └── stagate.py
│       ├── contamination_benchmark
│       │   ├── chrysalis.py
│       │   ├── graphst.py
│       │   ├── mefisto.py
│       │   ├── nsf.py
│       │   └── stagate.py
│       └── main_synthetic_benchmark
│           ├── chrysalis.py
│           ├── graphst.py
│           ├── mefisto.py
│           ├── nsf.py
│           └── stagate.py
├── A2_human_lymph_node
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
├── A3_human_breast_cancer
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
├── A4_mouse_brain
│   ├── ff
│   │   ├── mouse_brain_ff.ipynb
│   │   └── mouse_brain_integration.ipynb
│   └── ffpe
│       ├── benchmark
│       │   ├── graphst.py
│       │   ├── mefisto.py
│       │   ├── nsf.py
│       │   └── stagate.py
│       ├── map_annotations.py
│       ├── mouse_brain_ffpe.ipynb
│       └── mouse_brain_ffpe_benchmark.ipynb
├── A5_visium_hd
│   └── visium_hd_analysis.ipynb
├── A6_slide_seqv2
│   └── slide_seqv2_analysis.ipynb
├── A7_stereo_seq
│   └── stereo_seq_analysis.ipynb
└── readme.md
```

## Chrysalis: decoding tissue compartments in spatial transcriptomics with archetypal analysis

**Authors**: Demeter Túrós, Jelica Vasiljevic, Kerstin Hahn, Sven Rottenberg, and Alberto Valdeolivas

**Abstract**: Dissecting tissue compartments in spatial transcriptomics (ST) remains challenging due to 
limited spatial resolution and dependence on single-cell reference data. We present Chrysalis, a novel 
computational method that rapidly uncovers tissue compartments through spatially variable gene (SVG) 
detection and archetypal analysis without requiring external reference data. Additionally, it offers a 
unique visualisation approach for swift tissue characterization and provides access to gene expression 
signatures, enabling the identification of spatially and functionally distinct cellular niches. Chrysalis 
was evaluated through various benchmarks and validated against deconvolution, independently obtained cell 
type abundance data, and histopathological annotations, demonstrating superior performance compared to 
other algorithms on both in silico and real-world test examples. Furthermore, we underscored its versatility 
across different technologies, such as Visium, Visium HD, Slide-seq, and Stereo-seq.
