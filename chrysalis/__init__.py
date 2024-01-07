import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# Filter out specific warning categories
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# core functions
from .core import detect_svgs
from .core import pca
from .core import aa

# plotting functions
from .plots import plot
from .plots import plot_compartment
from .plots import plot_compartments
from .plots import plot_explained_variance
from .plots import plot_svgs
from .plots import plot_rss
from .plots import plot_heatmap
from .plots import plot_weights
from .plots import plot_svg_matrix
from .plots import plot_samples

# utility functions
from .utils import get_compartment_df
from .utils import integrate_adatas
from .utils import harmony_integration
