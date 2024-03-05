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


R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1

from .helpers import as_fenics_constant

from .boundary_conditions.dirichlet_bc import DirichletBC
from .boundary_conditions.sieverts_bc import SievertsBC
from .boundary_conditions.henrys_bc import HenrysBC

from .material import Material

from .mesh.mesh import Mesh
from .mesh.mesh_1d import Mesh1D
from .mesh.mesh_from_xdmf import MeshFromXDMF

from .hydrogen_transport_problem import HydrogenTransportProblem

from .initial_condition import InitialCondition

from .settings import Settings

from .source import Source

from .species import Species, Trap, ImplicitSpecies, find_species_from_name

from .subdomain.surface_subdomain import SurfaceSubdomain, find_surface_from_id
from .subdomain.surface_subdomain_1d import SurfaceSubdomain1D
from .subdomain.volume_subdomain import VolumeSubdomain, find_volume_from_id
from .subdomain.volume_subdomain_1d import VolumeSubdomain1D

from .stepsize import Stepsize

from .exports.surface_quantity import SurfaceQuantity
from .exports.volume_quantity import VolumeQuantity
from .exports.total_volume import TotalVolume
from .exports.surface_flux import SurfaceFlux
from .exports.vtx import VTXExport
from .exports.xdmf import XDMFExport

from .reaction import Reaction
