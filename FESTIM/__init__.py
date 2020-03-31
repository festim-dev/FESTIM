from . import helpers

from . import meshing
from . import functionspaces_and_functions
from . import initialise_solutions
from . import formulations

from . import boundary_conditions
from . import solving

from . import export
from . import post_processing

from . import generic_simulation

import sympy as sp
x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
R = 8.314  # Gas constant
k_B = 8.6e-5  # Boltzmann constant
