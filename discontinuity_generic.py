from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix
from festim.helpers_discontinuity import NewtonSolver, transfer_meshtags_to_submesh
import festim as F


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

top_domain = F.VolumeSubdomain(4, material=None)
bottom_domain = F.VolumeSubdomain(3, material=None)
list_of_subdomains = [bottom_domain, top_domain]
list_of_interfaces = {5: [bottom_domain, top_domain]}

gdim = mesh.geometry.dim
tdim = mesh.topology.dim
fdim = tdim - 1

num_facets_local = (
    mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
)

for subdomain in list_of_subdomains:
    subdomain.submesh, subdomain.submesh_to_mesh, subdomain.v_map = (
        dolfinx.mesh.create_submesh(mesh, tdim, ct.find(subdomain.id))[0:3]
    )

    subdomain.parent_to_submesh = np.full(num_facets_local, -1, dtype=np.int32)
    subdomain.parent_to_submesh[subdomain.submesh_to_mesh] = np.arange(
        len(subdomain.submesh_to_mesh), dtype=np.int32
    )

    # We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
    # We use the entity on the same side to fix this (as all restrictions are one-sided)

    # Transfer meshtags to submesh
    subdomain.ft, subdomain.facet_to_parent = transfer_meshtags_to_submesh(
        mesh, mt, subdomain.submesh, subdomain.v_map, subdomain.submesh_to_mesh
    )

# this seems to be not needed
# t_parent_to_facet = np.full(num_facets_local, -1)
# t_parent_to_facet[t_facet_to_parent] = np.arange(
#     len(t_facet_to_parent), dtype=np.int32
# )

# Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
# TODO ask Jorgen what this is for
mesh.topology.create_connectivity(fdim, tdim)
f_to_c = mesh.topology.connectivity(fdim, tdim)
for interface in list_of_interfaces:
    for facet in mt.find(interface):
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        for domain in list_of_interfaces[interface]:
            map = domain.parent_to_submesh[cells]
            domain.parent_to_submesh[cells] = max(map)

entity_maps = {
    subdomain.submesh: subdomain.parent_to_submesh for subdomain in list_of_subdomains
}
exit()


def D(T):
    k_B = 8.6173303e-5
    return 2 * ufl.exp(-0.1 / k_B / T)


def define_interior_eq(mesh, degree, submesh, submesh_to_mesh, value):
    # Compute map from parent entity to submesh cell
    codim = mesh.topology.dim - submesh.topology.dim
    ptdim = mesh.topology.dim - codim
    num_entities = (
        mesh.topology.index_map(ptdim).size_local
        + mesh.topology.index_map(ptdim).num_ghosts
    )
    mesh_to_submesh = np.full(num_entities, -1)
    mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh), dtype=np.int32)

    degree = 1
    element_CG = basix.ufl.element(
        basix.ElementFamily.P,
        submesh.basix_cell(),
        degree,
        basix.LagrangeVariant.equispaced,
    )
    element = basix.ufl.mixed_element([element_CG, element_CG])
    V = dolfinx.fem.functionspace(submesh, element)
    u = dolfinx.fem.Function(V)
    us = list(ufl.split(u))
    vs = list(ufl.TestFunctions(V))
    ct_r = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        submesh_to_mesh,
        np.full_like(submesh_to_mesh, 1, dtype=np.int32),
    )
    val = dolfinx.fem.Constant(submesh, value)
    dx_r = ufl.Measure("dx", domain=mesh, subdomain_data=ct_r, subdomain_id=1)
    F = ufl.inner(ufl.grad(us[0]), ufl.grad(vs[0])) * dx_r - val * vs[0] * dx_r
    k = 2
    p = 0.1
    n = 0.5
    F += k * us[0] * (n - us[1]) * vs[1] * dx_r - p * us[1] * vs[1] * dx_r
    return u, vs, F, mesh_to_submesh


# for each subdomain, define the interior equation
u_0, v_0s, F_00, m_to_b = define_interior_eq(mesh, 2, submesh_b, submesh_b_to_mesh, 0.0)
u_1, v_1s, F_11, m_to_t = define_interior_eq(mesh, 1, submesh_t, submesh_t_to_mesh, 0.0)
u_0.name = "u_b"
u_1.name = "u_t"


# Add coupling term to the interface
# Get interface markers on submesh b
dInterface = ufl.Measure("dS", domain=mesh, subdomain_data=mt, subdomain_id=5)
b_res = "+"
t_res = "-"

v_b = v_0s[0](b_res)
v_t = v_1s[0](t_res)

u_bs = list(ufl.split(u_0))
u_ts = list(ufl.split(u_1))
u_b = u_bs[0](b_res)
u_t = u_ts[0](t_res)


def mixed_term(u, v, n):
    return ufl.dot(ufl.grad(u), n) * v


n = ufl.FacetNormal(mesh)
n_b = n(b_res)
n_t = n(t_res)
cr = ufl.Circumradius(mesh)
h_b = 2 * cr(b_res)
h_t = 2 * cr(t_res)
gamma = 10.0

W_0 = dolfinx.fem.functionspace(submesh_b, ("DG", 0))
K_0 = dolfinx.fem.Function(W_0)
K_0.x.array[:] = 2
W_1 = dolfinx.fem.functionspace(submesh_t, ("DG", 0))
K_1 = dolfinx.fem.Function(W_1)
K_1.x.array[:] = 4

K_b = K_0(b_res)
K_t = K_1(t_res)


F_0 = (
    -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface
    - 0.5 * mixed_term(v_b, (u_b / K_b - u_t / K_t), n_b) * dInterface
)

F_1 = (
    +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface
    - 0.5 * mixed_term(v_t, (u_b / K_b - u_t / K_t), n_b) * dInterface
)
F_0 += 2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_b * dInterface
F_1 += -2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_t * dInterface

F_0 += F_00
F_1 += F_11

jac00 = ufl.derivative(F_0, u_0)

jac01 = ufl.derivative(F_0, u_1)

jac10 = ufl.derivative(F_1, u_0)
jac11 = ufl.derivative(F_1, u_1)

J00 = dolfinx.fem.form(jac00, entity_maps=entity_maps)
J01 = dolfinx.fem.form(jac01, entity_maps=entity_maps)
J10 = dolfinx.fem.form(jac10, entity_maps=entity_maps)
J11 = dolfinx.fem.form(jac11, entity_maps=entity_maps)
J = [[J00, J01], [J10, J11]]
F = [
    dolfinx.fem.form(F_0, entity_maps=entity_maps),
    dolfinx.fem.form(F_1, entity_maps=entity_maps),
]


# boundary conditions
b_bc = dolfinx.fem.Function(u_0.function_space)
b_bc.x.array[:] = 0.2
submesh_b.topology.create_connectivity(
    submesh_b.topology.dim - 1, submesh_b.topology.dim
)
bc_b = dolfinx.fem.dirichletbc(
    b_bc,
    dolfinx.fem.locate_dofs_topological(u_0.function_space.sub(0), fdim, ft_b.find(2)),
)


t_bc = dolfinx.fem.Function(u_1.function_space)
t_bc.x.array[:] = 0.05
submesh_t.topology.create_connectivity(
    submesh_t.topology.dim - 1, submesh_t.topology.dim
)
bc_t = dolfinx.fem.dirichletbc(
    t_bc,
    dolfinx.fem.locate_dofs_topological(u_1.function_space.sub(0), fdim, ft_t.find(1)),
)
bcs = [bc_b, bc_t]


solver = NewtonSolver(
    F,
    J,
    [u_0, u_1],
    bcs=bcs,
    max_iterations=10,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
solver.solve(1e-5)

# bp = dolfinx.io.VTXWriter(mesh.comm, "u_b.bp", [u_0.sub(0).collapse()], engine="BP4")
bp = dolfinx.io.VTXWriter(mesh.comm, "u_b_0.bp", [u_0.sub(0).collapse()], engine="BP4")
bp.write(0)
bp.close()
bp = dolfinx.io.VTXWriter(mesh.comm, "u_t_0.bp", [u_1.sub(0).collapse()], engine="BP4")
bp.write(0)
bp.close()
bp = dolfinx.io.VTXWriter(mesh.comm, "u_b_1.bp", [u_0.sub(1).collapse()], engine="BP4")
bp.write(0)
bp.close()
bp = dolfinx.io.VTXWriter(mesh.comm, "u_t_1.bp", [u_1.sub(1).collapse()], engine="BP4")
bp.write(0)
bp.close()


# derived quantities
V = dolfinx.fem.functionspace(mesh, ("CG", 1))
T = dolfinx.fem.Function(V)
T.interpolate(lambda x: 200 + x[1])


T_b = dolfinx.fem.Function(u_0.sub(0).collapse().function_space)
T_b.interpolate(T)

ds_b = ufl.Measure("ds", domain=submesh_b)
dx_b = ufl.Measure("dx", domain=submesh_b)
dx = ufl.Measure("dx", domain=mesh)

n_b = ufl.FacetNormal(submesh_b)

form = dolfinx.fem.form(u_0.sub(0) * dx_b)
print(dolfinx.fem.assemble_scalar(form))

form = dolfinx.fem.form(T_b * ufl.dot(ufl.grad(u_0.sub(0)), n_b) * ds_b)
print(dolfinx.fem.assemble_scalar(form))
