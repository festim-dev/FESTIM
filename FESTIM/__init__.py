import sympy as sp
x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1

from .helpers import update_expressions, help_key, \
    parameters_helper

from .meshing import Mesh, Mesh1D, MeshFromVertices, MeshFromRefinements, MeshFromXDMF
from .temperature import Temperature
from .temperature_solver import HeatTransferProblem

from .boundary_conditions.boundary_condition import BoundaryCondition
from .boundary_conditions.dirichlets.dirichlet_bc import DirichletBC, \
    BoundaryConditionTheta, BoundaryConditionExpression
from .boundary_conditions.dirichlets.dc_imp import ImplantationDirichlet
from .boundary_conditions.dirichlets.sieverts_bc import SievertsBC
from .boundary_conditions.dirichlets.custom_dc import CustomDirichlet

from .boundary_conditions.fluxes.flux_bc import FluxBC
from .boundary_conditions.fluxes.recombination_flux import RecombinationFlux
from .boundary_conditions.fluxes.convective_flux import ConvectiveFlux
from .boundary_conditions.fluxes.flux_custom import CustomFlux

from .solving import solve_it, solve_once

from .exports.exports import Exports
from .exports.export import Export
from .exports.error import Error
from .exports.xdmf_export import XDMFExport, XDMFExports

from .exports.derived_quantities.derived_quantity import DerivedQuantity
from .exports.derived_quantities.surface_flux import SurfaceFlux
from .exports.derived_quantities.hydrogen_flux import HydrogenFlux
from .exports.derived_quantities.thermal_flux import ThermalFlux
from .exports.derived_quantities.average_volume import AverageVolume
from .exports.derived_quantities.maximum_volume import MaximumVolume
from .exports.derived_quantities.minimum_volume import MinimumVolume
from .exports.derived_quantities.total_surface import TotalSurface
from .exports.derived_quantities.total_volume import TotalVolume

from .exports.derived_quantities.derived_quantities import DerivedQuantities

from .exports.txt_export import TXTExport, TXTExports


from .settings import Settings
from .stepsize import Stepsize

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
