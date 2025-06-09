import warnings
from numba.core.errors import NumbaDeprecationWarning

# core functions
from .core import detect_svgs
from .core import pca
from .core import aa
from .core import compute_svg_scores
from .core import select_svgs

# plotting functions
from .plots import plot
from .plots import plot_compartment
from .plots import plot_compartments
from .plots import plot_explained_variance
from .plots import plot_svgs  # deprecated
from .plots import plot_svg_ranks
from .plots import plot_rss
from .plots import plot_heatmap
from .plots import plot_weights  # deprecated
from .plots import plot_gene_weights
from .plots import plot_svg_matrix
from .plots import plot_samples

# utility functions
from .utils import get_compartment_df  #deprecated
from .utils import get_gene_weights_df
from .utils import get_coefficients_df
from .utils import integrate_adatas
from .utils import harmony_integration

# filter out specific warning categories
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
