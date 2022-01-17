import sympy as sp
x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1

from .helpers import update_expressions, help_key, \
    parameters_helper

from .meshing import Mesh, Mesh1D, MeshFromVertices, MeshFromRefinements, MeshFromXDMF
from .initialising import check_no_duplicates
from .temperature import Temperature
from .boundary_conditions import BoundaryCondition, DirichletBC, FluxBC, \
    BoundaryConditionTheta, BoundaryConditionExpression

from .solving import solve_it, solve_once, adaptive_stepsize

from .export import treat_value, export_parameters, Export, Exports
from .exports.xdmf_export import XDMFExport, XDMFExports
from .exports.derived_quantities_export import DerivedQuantities
from .exports.txt_export import TXTExport, TXTExports

from .post_processing import run_post_processing, compute_error, \
    create_properties, \
    check_keys_derived_quantities

from .materials import Material, Materials
from .concentration import Concentration

from .mobile import Mobile

from .traps.trap import Trap
from .traps.traps import Traps
from .traps.extrinsic_trap import ExtrinsicTrap

from .formulations import formulation, formulation_extrinsic_traps

from .generic_simulation import Simulation, run
