[![Stars](https://img.shields.io/github/stars/rockdeme/chrysalis?logo=GitHub&color=yellow)](https://github.com/rockdeme/chrysalis/stargazers)
[![PyPI](https://img.shields.io/pypi/v/chrysalis-st?logo=PyPI)](https://pypi.org/project/chrysalis-st)

# Welcome to chrysalis!

```{include} ../README.md
:start-line: 4
:end-line: 7
```

* Discuss **chrysalis** on [GitHub].
* Get started by reading the {doc}`basic tutorial <tutorials/lymph_node_tutorial>`.
* You can also browse the {doc}`API <api>`.
* Consider citing our [bioRxiv preprint].

## Visual demonstration
### human lung cancer (FFPE)

[Squamous Cell Carcinoma](https://www.10xgenomics.com/resources/datasets/human-lung-cancer-ffpe-2-standard) sample by 10X Genomics.

Move the slider to reveal tissue compartments calculated by **chrysalis** or the associated tissue morphology.

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<div class='befaft_container'>
<div class='img background-img'></div>
<div class='img foreground-img'></div>
<input type="range" min="1" max="100" value="50" class="slider" name='slider' id="slider">
<div class='slider-button'></div>
</div>

```{toctree}
:hidden: true
:maxdepth: 1
:caption: chrysalis

overview/getting-started
overview/installation
```

```{toctree}
:hidden: true
:maxdepth: 2
:caption: Tutorials

tutorials/lymph_node_tutorial
tutorials/mouse_brain_integration_tutorial
```

```{toctree}
:hidden: true
:maxdepth: 2
:caption: API

api
```

[GitHub]: https://github.com/rockdeme/chrysalis
[bioRxiv preprint]: https://doi.org/10.1101/2023.08.17.553606
