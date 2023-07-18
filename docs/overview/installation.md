# Installation

Create a new conda environment if required:

```shell
conda create -n chrysalis-env python=3.8
```

```shell
conda activate chrysalis-env
```

You can install **chrysalis** from PyPI using pip:

```shell
pip install chrysalis-st
```

This will install **chrysalis** and all dependencies including `scanpy`.

## Troubleshooting

If `rvlib` fails to install, you can try installing it with conda:
```shell
conda install -c conda-forge rvlib 
```
