try:
    # Python 3.8+
    from importlib import metadata
except ImportError:
    try:
        import importlib_metadata as metadata
    except ImportError:
        __version__ = "unknown"

try:
    __version__ = metadata.version("FESTIM")
except Exception:
    __version__ = "unknown"


import sympy as sp

x, y, z, t = sp.symbols("x[0] x[1] x[2] t")
R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1

from .helpers import (
    update_expressions,
    kJmol_to_eV,
    extract_xdmf_labels,
    extract_xdmf_times,
    as_constant,
    as_expression,
    as_constant_or_expression,
)

from .meshing.mesh import Mesh
from .meshing.mesh_1d import Mesh1D
from .meshing.mesh_from_refinements import MeshFromRefinements
from .meshing.mesh_from_vertices import MeshFromVertices
from .meshing.mesh_from_xdmf import MeshFromXDMF

from .temperature.temperature import Temperature
from .temperature.temperature_solver import HeatTransferProblem
from .temperature.temperature_from_xdmf import TemperatureFromXDMF

from .boundary_conditions.boundary_condition import BoundaryCondition
from .boundary_conditions.dirichlets.dirichlet_bc import (
    DirichletBC,
    BoundaryConditionTheta,
    BoundaryConditionExpression,
)
from .boundary_conditions.dirichlets.dc_imp import ImplantationDirichlet
from .boundary_conditions.dirichlets.sieverts_bc import SievertsBC
from .boundary_conditions.dirichlets.henrys_bc import HenrysBC
from .boundary_conditions.dirichlets.custom_dc import CustomDirichlet

from .boundary_conditions.fluxes.flux_bc import FluxBC
from .boundary_conditions.fluxes.recombination_flux import RecombinationFlux
from .boundary_conditions.fluxes.convective_flux import ConvectiveFlux
from .boundary_conditions.fluxes.flux_custom import CustomFlux

from .exports.exports import Exports
from .exports.export import Export
from .exports.xdmf_export import XDMFExport, XDMFExports
from .exports.trap_density_xdmf import TrapDensityXDMF

from .exports.derived_quantities.derived_quantity import (
    DerivedQuantity,
    VolumeQuantity,
    SurfaceQuantity,
)
from .exports.derived_quantities.surface_flux import SurfaceFlux
from .exports.derived_quantities.hydrogen_flux import HydrogenFlux
from .exports.derived_quantities.thermal_flux import ThermalFlux
from .exports.derived_quantities.average_volume import AverageVolume
from .exports.derived_quantities.maximum_volume import MaximumVolume
from .exports.derived_quantities.minimum_volume import MinimumVolume
from .exports.derived_quantities.minimum_surface import MinimumSurface
from .exports.derived_quantities.maximum_surface import MaximumSurface
from .exports.derived_quantities.total_surface import TotalSurface
from .exports.derived_quantities.total_volume import TotalVolume
from .exports.derived_quantities.average_surface import AverageSurface

from .exports.derived_quantities.derived_quantities import DerivedQuantities

from .exports.txt_export import TXTExport, TXTExports


from .settings import Settings
from .stepsize import Stepsize

from .sources.source import Source
from .sources.source_implantation_flux import ImplantationFlux

from .materials.material import Material
from .materials.materials import Materials

from .concentration.concentration import Concentration
from .initial_condition import InitialCondition
from .concentration.mobile import Mobile
from .concentration.theta import Theta

from .concentration.traps.trap import Trap
from .concentration.traps.traps import Traps
from .concentration.traps.extrinsic_trap import ExtrinsicTrapBase
from .concentration.traps.extrinsic_trap import ExtrinsicTrap
from .concentration.traps.neutron_induced_trap import NeutronInducedTrap

from .h_transport_problem import HTransportProblem

from .generic_simulation import Simulation
