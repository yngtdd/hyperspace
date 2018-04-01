from .space import space

from .rover.latin_hypercube_sampler import sample_latin_hypercube
from .rover.latin_hypercube_sampler import lhs_start
from .space.mapping_space import check_dimension
from .space.mapping_space import create_hyperspace
from .space import HyperInteger
from .space import HyperReal
from .space import HyperCategorical


__version__ = "0.2"

__all__ = (
    "check_dimension",
    "create_hyperspace",
    "HyperCategorical"
    "HyperInteger",
    "HyperReal"
    "lhs_start"
    "sample_latin_hypercube"
)
