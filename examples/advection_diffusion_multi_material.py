from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import basix
import festim as F


# ---------------- Generate a mesh ----------------
def generate_mesh(n=20):
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def top_boundary(x):
        return np.isclose(x[1], 1.0)

    def bottom_left_boundary(x):
        return np.logical_and(np.isclose(x[0], 0.0), x[1] <= 0.5 + 1e-14)

    def half(x):
        return x[1] <= 0.5 + 1e-14

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle
    )

    # Split domain in half and set an interface tag of 5
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    fdim = tdim - 1
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
    bot_left_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, bottom_left_boundary
    )
    num_facets_local = (
        mesh.topology.index_map(fdim).size_local
        + mesh.topology.index_map(fdim).num_ghosts
    )
    facets = np.arange(num_facets_local, dtype=np.int32)
    values = np.full_like(facets, 0, dtype=np.int32)
    values[top_facets] = 1
    values[bottom_facets] = 2
    values[bot_left_facets] = 3

    bottom_cells = dolfinx.mesh.locate_entities(mesh, tdim, half)
    num_cells_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    cells = np.full(num_cells_local, 5, dtype=np.int32)
    cells[bottom_cells] = 4
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
    )
    all_b_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(4), tdim, fdim
    )
    all_t_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(5), tdim, fdim
    )
    interface = np.intersect1d(all_b_facets, all_t_facets)
    values[interface] = 6

    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, values)
    return mesh, mt, ct


mesh, mt, ct = generate_mesh()


# create velocity field
def create_velocity_field():
    def velocity_func(x):
        values = np.zeros((2, x.shape[1]))  # Initialize with zeros
        scalar_value = -500 * x[1] * (x[1] - 0.5)  # Compute the scalar function
        values[0] = scalar_value  # Assign to first component
        values[1] = 0  # Second component remains zero
        return values

    mesh2, mt2, ct2 = generate_mesh(n=10)
    submesh, submesh_to_mesh, v_map = dolfinx.mesh.create_submesh(
        mesh2, ct2.dim, ct2.find(4)
    )[0:3]
    v_cg = basix.ufl.element(
        "Lagrange", submesh.topology.cell_name(), 2, shape=(submesh.geometry.dim,)
    )
    V_velocity = dolfinx.fem.functionspace(submesh, v_cg)
    u = dolfinx.fem.Function(V_velocity)
    u.interpolate(velocity_func)
    return u


u = create_velocity_field()

writer = dolfinx.io.VTXWriter(MPI.COMM_WORLD, "velocity.bp", u, engine="BP5")
writer.write(t=0)

my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh(mesh)
my_model.volume_meshtags = ct
my_model.facet_meshtags = mt

material_bottom = F.Material(D_0=1, E_D=0 * 0.1)
material_top = F.Material(D_0=1, E_D=0 * 0.1)

material_bottom.K_S_0 = 2.0
material_bottom.E_K_S = 0
material_top.K_S_0 = 600
material_top.E_K_S = 0

top_domain = F.VolumeSubdomain(5, material=material_top)
bottom_domain = F.VolumeSubdomain(4, material=material_bottom)

top_surface = F.SurfaceSubdomain(id=1)
bottom_surface = F.SurfaceSubdomain(id=2)
inlet_surf = F.SurfaceSubdomain(id=3)
my_model.subdomains = [
    bottom_domain,
    top_domain,
    top_surface,
    bottom_surface,
    inlet_surf,
]

# we should be able to automate this
my_model.interfaces = [F.Interface(6, (bottom_domain, top_domain))]
my_model.surface_to_volume = {
    top_surface: top_domain,
    bottom_surface: bottom_domain,
    inlet_surf: bottom_domain,
}

H = F.Species("H", mobile=True)

my_model.species = [H]

for species in my_model.species:
    species.subdomains = [bottom_domain, top_domain]


my_model.boundary_conditions = [
    F.FixedConcentrationBC(top_surface, value=0.0, species=H),
    F.FixedConcentrationBC(inlet_surf, value=1.0, species=H),
    F.FixedConcentrationBC(bottom_surface, value=0.0, species=H),
]

my_model.advection_terms = [
    F.AdvectionTerm(velocity=u, species=[H], subdomain=bottom_domain)
]


my_model.temperature = 500.0

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.exports = [
    F.VTXSpeciesExport(f"u_{subdomain.id}.bp", field=H, subdomain=subdomain)
    for subdomain in my_model.volume_subdomains
]

my_model.initialise()
my_model.run()
