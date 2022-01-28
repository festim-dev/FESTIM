import sympy as sp
x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1

from .helpers import update_expressions, help_key, \
    parameters_helper

from .meshing import Mesh, Mesh1D, MeshFromVertices, MeshFromRefinements, MeshFromXDMF
from .temperature import Temperature
from .boundary_conditions.boundary_conditions import BoundaryCondition, DirichletBC, FluxBC, \
    BoundaryConditionTheta, BoundaryConditionExpression
from .boundary_conditions.recombination_flux import RecombinationFlux
from .boundary_conditions.convective_flux import ConvectiveFlux

from .solving import solve_it, solve_once

from .export import Export, Exports
from .exports.error import Error
from .exports.xdmf_export import XDMFExport, XDMFExports
from .exports.derived_quantities_export import DerivedQuantities, \
    SurfaceFlux, AverageVolume, MinimumVolume, MaximumVolume, \
    TotalVolume, TotalSurface, ThermalFlux, HydrogenFlux
from .exports.txt_export import TXTExport, TXTExports


from .settings import Settings
from .stepsize import Stepsize

from .post_processing import is_export_derived_quantities, is_export_xdmf, \
    create_properties

from .source import Source

from .materials import Material, Materials
from .concentration import Concentration
from .initial_condition import InitialCondition
from .mobile import Mobile

from .traps.trap import Trap
from .traps.traps import Traps
from .traps.extrinsic_trap import ExtrinsicTrap

from .formulations import formulation, formulation_extrinsic_traps

from .generic_simulation import Simulation, run
