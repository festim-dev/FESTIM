import festim as F
from problem import HTransportProblemDiscontinuous

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import festim as F
import ufl


# ---------------- Generate a mesh ----------------
def generate_mesh():
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def top_boundary(x):
        return np.isclose(x[1], 1.0)

    def half(x):
        return x[1] <= 0.5 + 1e-14

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 20, 20, dolfinx.mesh.CellType.triangle
    )

    # Split domain in half and set an interface tag of 5
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    fdim = tdim - 1
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
    num_facets_local = (
        mesh.topology.index_map(fdim).size_local
        + mesh.topology.index_map(fdim).num_ghosts
    )
    facets = np.arange(num_facets_local, dtype=np.int32)
    values = np.full_like(facets, 0, dtype=np.int32)
    values[top_facets] = 1
    values[bottom_facets] = 2

    bottom_cells = dolfinx.mesh.locate_entities(mesh, tdim, half)
    num_cells_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    cells = np.full(num_cells_local, 4, dtype=np.int32)
    cells[bottom_cells] = 3
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
    )
    all_b_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(3), tdim, fdim
    )
    all_t_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(4), tdim, fdim
    )
    interface = np.intersect1d(all_b_facets, all_t_facets)
    values[interface] = 5

    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, values)
    return mesh, mt, ct


mesh, mt, ct = generate_mesh()

my_model = HTransportProblemDiscontinuous()
my_model.mesh = F.Mesh(mesh)
my_model.volume_meshtags = ct
my_model.facet_meshtags = mt

material_bottom = F.Material(D_0=2.0, E_D=0.1)
material_top = F.Material(D_0=2.0, E_D=0.1)

material_bottom.K_S_0 = 2.0
material_bottom.E_K_S = 0 * 0.1
material_top.K_S_0 = 4.0
material_top.E_K_S = 0 * 0.12

top_domain = F.VolumeSubdomain(4, material=material_top)
bottom_domain = F.VolumeSubdomain(3, material=material_bottom)

# we should be able to automate this
my_model.interfaces = {5: [bottom_domain, top_domain]}

top_surface = F.SurfaceSubdomain(id=1)
bottom_surface = F.SurfaceSubdomain(id=2)

my_model.subdomains = [bottom_domain, top_domain, top_surface, bottom_surface]

H = F.Species("H", mobile=True)
trapped_H = F.Species("H_trapped", mobile=False)
empty_trap = F.ImplicitSpecies(n=0.5, others=[trapped_H])

my_model.species = [H, trapped_H]

for species in [H, trapped_H]:
    species.subdomains = [bottom_domain, top_domain]
    species.subdomain_to_solution = {}
    species.subdomain_to_prev_solution = {}
    species.subdomain_to_test_function = {}

my_model.reactions = [
    F.Reaction(
        reactant=[H, empty_trap],
        product=[trapped_H],
        k_0=2,
        E_k=0,
        p_0=0.1,
        E_p=0,
        volume=top_domain,
    ),
    F.Reaction(
        reactant=[H, empty_trap],
        product=[trapped_H],
        k_0=2,
        E_k=0,
        p_0=0.1,
        E_p=0,
        volume=bottom_domain,
    ),
]

my_model.boundary_conditions = [
    F.DirichletBC(top_surface, value=0.05, species=H),
    F.DirichletBC(bottom_surface, value=0.2, species=H),
]

my_model.surface_to_volume = {top_surface: top_domain, bottom_surface: bottom_domain}

my_model.temperature = lambda x: 300 + 10 * x[1] + 100 * x[0]

my_model.settings = F.Settings(atol=None, rtol=None, transient=False)


my_model.initialise()
my_model.run()

list_of_subdomains = my_model.volume_subdomains

for subdomain in list_of_subdomains:
    u_sub_0 = subdomain.u.sub(0).collapse()
    u_sub_0.name = "u_sub_0"

    u_sub_1 = subdomain.u.sub(1).collapse()
    u_sub_1.name = "u_sub_1"
    bp = dolfinx.io.VTXWriter(
        mesh.comm, f"u_{subdomain.id}.bp", [u_sub_0, u_sub_1], engine="BP4"
    )
    bp.write(0)
    bp.close()


# derived quantities
my_model.entity_maps[mesh] = bottom_domain.submesh_to_mesh

ds_b = ufl.Measure("ds", domain=bottom_domain.submesh, subdomain_data=bottom_domain.ft)
ds_t = ufl.Measure("ds", domain=top_domain.submesh, subdomain_data=top_domain.ft)
dx_b = ufl.Measure("dx", domain=bottom_domain.submesh)
dx = ufl.Measure("dx", domain=mesh)

n_b = ufl.FacetNormal(bottom_domain.submesh)
n_t = ufl.FacetNormal(top_domain.submesh)

form = dolfinx.fem.form(bottom_domain.u.sub(0) * dx_b)
print(dolfinx.fem.assemble_scalar(form))

form = dolfinx.fem.form(bottom_domain.u.sub(1) * dx_b)
print(dolfinx.fem.assemble_scalar(form))

form = dolfinx.fem.form(
    my_model.temperature_fenics * dx_b,
    entity_maps={mesh: bottom_domain.submesh_to_mesh},
)
print(dolfinx.fem.assemble_scalar(form))

D = subdomain.material.get_diffusion_coefficient(
    my_model.mesh.mesh, my_model.temperature_fenics, H
)
id_interface = 5
form = dolfinx.fem.form(
    ufl.dot(
        D * ufl.grad(bottom_domain.u.sub(0)),
        n_b,
    )
    * ds_b(id_interface),
    entity_maps={mesh: bottom_domain.submesh_to_mesh},
)
print(dolfinx.fem.assemble_scalar(form))
form = dolfinx.fem.form(
    ufl.dot(
        D * ufl.grad(top_domain.u.sub(0)),
        n_t,
    )
    * ds_t(id_interface),
    entity_maps={mesh: top_domain.submesh_to_mesh},
)
print(dolfinx.fem.assemble_scalar(form))