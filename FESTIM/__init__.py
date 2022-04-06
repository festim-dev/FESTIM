from .generic_simulation import Simulation
from .h_transport_problem import HTransportProblem
from .concentration.traps.neutron_induced_trap import NeutronInducedTrap
from .concentration.traps.extrinsic_trap import ExtrinsicTrap
from .concentration.traps.extrinsic_trap import ExtrinsicTrapBase
from .concentration.traps.traps import Traps
from .concentration.traps.trap import Trap
from .concentration.theta import Theta
from .concentration.mobile import Mobile
from .initial_condition import InitialCondition
from .concentration.concentration import Concentration
from .materials.material import Material
from .materials.materials import Materials
from .sources.source_implantation_flux import ImplantationFlux
from .sources.source import Source
from .stepsize import Stepsize
from .settings import Settings
from .exports.txt_export import TXTExport, TXTExports
from .exports.derived_quantities.derived_quantities import DerivedQuantities
from .exports.derived_quantities.total_volume import TotalVolume
from .exports.derived_quantities.total_surface import TotalSurface
from .exports.derived_quantities.minimum_volume import MinimumVolume
from .exports.derived_quantities.maximum_volume import MaximumVolume
from .exports.derived_quantities.average_volume import AverageVolume
from .exports.derived_quantities.thermal_flux import ThermalFlux
from .exports.derived_quantities.hydrogen_flux import HydrogenFlux
from .exports.derived_quantities.surface_flux import SurfaceFlux
from .exports.derived_quantities.derived_quantity import DerivedQuantity
from .exports.xdmf_export import XDMFExport, XDMFExports
from .exports.export import Export
from .exports.exports import Exports
from .boundary_conditions.fluxes.flux_custom import CustomFlux
from .boundary_conditions.fluxes.convective_flux import ConvectiveFlux
from .boundary_conditions.fluxes.recombination_flux import RecombinationFlux
from .boundary_conditions.fluxes.flux_bc import FluxBC
from .boundary_conditions.dirichlets.custom_dc import CustomDirichlet
from .boundary_conditions.dirichlets.sieverts_bc import SievertsBC
from .boundary_conditions.dirichlets.dc_imp import ImplantationDirichlet
from .boundary_conditions.dirichlets.dirichlet_bc import DirichletBC, \
    BoundaryConditionTheta, BoundaryConditionExpression
from .boundary_conditions.boundary_condition import BoundaryCondition
from .temperature.temperature_solver import HeatTransferProblem
from .temperature.temperature import Temperature
from .meshing.mesh_from_xdmf import MeshFromXDMF
from .meshing.mesh_from_vertices import MeshFromVertices
from .meshing.mesh_from_refinements import MeshFromRefinements
from .meshing.mesh_1d import Mesh1D
from .meshing.mesh import Mesh
from .helpers import update_expressions, kJmol_to_eV, \
    extract_xdmf_labels, extract_xdmf_times, as_constant, as_expression, as_constant_or_expression
import sympy as sp
x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1
