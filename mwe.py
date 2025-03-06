import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, locate_entities
import basix
from dolfinx import fem
import festim as F
import ufl
from dolfinx import io


class AdvectionTerm:
    velocity: fem.Function

    def __init__(self, velocity, subdomain, species):
        self.velocity = velocity
        self.subdomain = subdomain
        self.species = species


mesh_2d = create_unit_square(MPI.COMM_WORLD, 20, 20)

# create velocity field
v_cg = basix.ufl.element(
    "Lagrange", mesh_2d.topology.cell_name(), 2, shape=(mesh_2d.geometry.dim,)
)
V_velocity = fem.functionspace(mesh_2d, v_cg)
u = fem.Function(V_velocity)


def velocity_func(x):
    values = np.zeros((2, x.shape[1]))  # Initialize with zeros
    scalar_value = -100 * x[1] * (x[1] - 1)  # Compute the scalar function
    values[0] = scalar_value  # Assign to first component
    values[1] = 0  # Second component remains zero
    return values


u.interpolate(velocity_func)


class LefSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        fdim = mesh.topology.dim - 1
        indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 0))
        return indices


class RightSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        fdim = mesh.topology.dim - 1
        indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1))
        return indices


class BottomSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        fdim = mesh.topology.dim - 1
        indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[1], 0))
        return indices


class TopSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        fdim = mesh.topology.dim - 1
        indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[1], 1))
        return indices


my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(mesh=mesh_2d)

my_mat = F.Material(name="mat", D_0=1, E_D=0)
vol = F.VolumeSubdomain(id=1, material=my_mat)
left = LefSurface(id=1)
right = RightSurface(id=2)
bottom = BottomSurface(id=3)
top = TopSurface(id=4)

my_model.subdomains = [vol, left, right, top, bottom]

H = F.Species("H", mobile=True, subdomains=[vol])
my_model.species = [H]


my_model.temperature = 500

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left, value=1, species=H),
    F.FixedConcentrationBC(subdomain=top, value=0, species=H),
    F.FixedConcentrationBC(subdomain=bottom, value=0, species=H),
]

my_model.exports = [
    # F.XDMFExport(filename="test_without_coupling.xdmf", field=H),
    F.XDMFExport(filename="test_with_coupling.xdmf", field=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)


my_model.advection_terms = [AdvectionTerm(velocity=u, subdomain=vol, species=[H])]


my_model.initialise()


# h_conc = H.solution
# h_test_function = H.test_function
# advection_term = ufl.inner(ufl.dot(ufl.grad(h_conc), u), h_test_function) * my_model.dx(
#     vol.id
# )
# my_model.formulation += advection_term

my_model.create_solver()


my_model.run()


# with io.XDMFFile(mesh_2d.comm, "test_with_coupling.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh_2d)
#     xdmf.write_function(H.post_processing_solution)
