from .hyperdrive import hyperband
from .hyperdrive import hyperbelt
from .hyperdrive import hyperdrive

from .rover.latin_hypercube_sampler import sample_latin_hypercube
from .rover.latin_hypercube_sampler import lhs_start

from .space.skopt.space import HyperInteger
from .space.skopt.space import HyperReal
from .space.skopt.space import HyperCategorical
from .space.skopt.mapping_space import check_dimension
from .space.skopt.mapping_space import create_hyperspace

from .space.robo.space import RoboInteger
from .space.robo.space import RoboReal
from .space.robo.mapping_space import check_robo_dimension
from .space.robo.mapping_space import create_robospace

__version__ = "0.2.3"

__all__ = [
    "check_dimension",
    "check_robo_dimension",
    "create_hyperspace",
    "dualdrive",
    "HyperCategorical",
    "hyperdrive",
    "HyperInteger",
    "HyperReal",
    "RoboInteger",
    "RoboReal",
    "lhs_start",
    "sample_latin_hypercube"
]

try:
    from .hyperdrive import robodrive
    __all__.append("robodrive")
except ImportError:
    print('To use RoBO methods, make sure to setup RoBO here: ' \
          'https://automl.github.io/RoBO/installation.html#manual-installation') 
