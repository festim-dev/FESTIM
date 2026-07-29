from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc_solver_for_manifold_derivatives import (
    custom_assemble_jacobian,
    custom_assemble_residual,
)
from ufl.algorithms import expand_derivatives

# Parameters
L = 4.0  # Total length of domain
x_int = L / 2  # Horizontal position of interface
D_left = 2.0  # diffusion coefficient of left domain
D_right = 1.5  # diffusion coefficient of right domain
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0
lam = 1.0
c_int_max = 1.0
D_int = 1.0  # diffusivity along the interface

dt = 0.1  # stepsize
T = 10.0  # final time

# Mesh, cell tags (left / right) and facet tags (interface)
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, 1.0])],
    [20, 10],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)


vdim = mesh.topology.dim
fdim = vdim - 1
mesh.topology.create_connectivity(fdim, vdim)

# Create mesh tags for the left and right subdomains
eps = 1e-10
LEFT_VOL_TAG, RIGHT_VOL_TAG = 1, 2
left_cells = dolfinx.mesh.locate_entities(mesh, vdim, lambda x: x[0] <= x_int + eps)
right_cells = dolfinx.mesh.locate_entities(mesh, vdim, lambda x: x[0] >= x_int - eps)
cell_indices = np.concatenate([left_cells, right_cells])
cell_values = np.concatenate(
    [np.full_like(left_cells, LEFT_VOL_TAG), np.full_like(right_cells, RIGHT_VOL_TAG)]
).astype(np.int32)
sort = np.argsort(cell_indices)
cell_tags = dolfinx.mesh.meshtags(mesh, vdim, cell_indices[sort], cell_values[sort])

# Create mesh tag for the interface
INT_TAG = 3
int_facets = dolfinx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], x_int))
int_facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, int_facets, np.full_like(int_facets, INT_TAG, dtype=np.int32)
)

# Create mesh tag for the left boundary (for a assigning a flux condition)
LEFT_BNDRY_TAG = 4
left_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[0], 0.0)
)
left_facet_tags = dolfinx.mesh.meshtags(
    mesh,
    fdim,
    left_facets,
    np.full_like(left_facets, LEFT_BNDRY_TAG, dtype=np.int32),
)


# Submeshes (two bulk + one interface) and their entity maps to parent
left_cells, left_emap, _, _ = dolfinx.mesh.create_submesh(
    mesh, vdim, cell_tags.find(LEFT_VOL_TAG)
)
right_cells, right_emap, _, _ = dolfinx.mesh.create_submesh(
    mesh, vdim, cell_tags.find(RIGHT_VOL_TAG)
)
mesh_int, int_emap, _, _ = dolfinx.mesh.create_submesh(
    mesh, fdim, int_facet_tags.find(INT_TAG)
)

# build interface integration entities ordered so "+" = left, "-" = right
imap = mesh.topology.index_map(vdim)
n_cells = imap.size_local + imap.num_ghosts
cell_marker = np.zeros(n_cells, dtype=np.int32)
cell_marker[cell_tags.indices] = cell_tags.values

f2c = mesh.topology.connectivity(fdim, vdim)
c2f = mesh.topology.connectivity(vdim, fdim)
ints = []
for f in int_facet_tags.find(INT_TAG):
    c0, c1 = f2c.links(f)
    if cell_marker[c0] == RIGHT_VOL_TAG:  # ensure left cell first
        c0, c1 = c1, c0
    lf0 = np.where(c2f.links(c0) == f)[0][0]
    lf1 = np.where(c2f.links(c1) == f)[0][0]
    ints += [c0, lf0, c1, lf1]
ints = np.array(ints, dtype=np.int32)


# Function spaces / functions
V_left = dolfinx.fem.functionspace(left_cells, ("CG", 1))
V_right = dolfinx.fem.functionspace(right_cells, ("CG", 1))
V_int = dolfinx.fem.functionspace(mesh_int, ("CG", 1))

W = ufl.MixedFunctionSpace(V_left, V_right, V_int)

c_left = dolfinx.fem.Function(V_left, name="c_left")
c_right = dolfinx.fem.Function(V_right, name="c_right")
c_int = dolfinx.fem.Function(V_int, name="c_int")

# previous time step (initial condition = 0)
c_left_n = dolfinx.fem.Function(V_left)
c_right_n = dolfinx.fem.Function(V_right)
c_int_n = dolfinx.fem.Function(V_int)

vh = ufl.TestFunctions(W)
v_left, v_right, v_int = vh

# Measures
dx_left = ufl.Measure(
    "dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=LEFT_VOL_TAG
)
dx_right = ufl.Measure(
    "dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=RIGHT_VOL_TAG
)
dx_int = ufl.Measure("dx", domain=mesh_int)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=int_facet_tags)  # internal facets
ds = ufl.Measure("ds", domain=mesh, subdomain_data=left_facet_tags)  # external facets

# Residual (backward Euler)
P, M = "+", "-"  # P = left side, M = right side

theta_P = c_int(P) / c_int_max
theta_M = c_int(M) / c_int_max

# --- group 1: integration domain = parent mesh ---------------------
F = 0
F += ((c_left - c_left_n) / dt) * v_left * dx_left
F += D_left * ufl.inner(ufl.grad(c_left), ufl.grad(v_left)) * dx_left
F += (k1 * c_left(P) * (1 - theta_P) - k2 * c_int(P)) * v_left(P) * dS(INT_TAG)

F += ((c_right - c_right_n) / dt) * v_right * dx_right
F += D_right * ufl.inner(ufl.grad(c_right), ufl.grad(v_right)) * dx_right
F += (k3 * c_right(M) * (1 - theta_M) - k4 * c_int(M)) * v_right(M) * dS(INT_TAG)

# --- group 2: integration domain = mesh_int ------------------------
F += lam * ((c_int - c_int_n) / dt) * v_int * dx_int  # trapping
F += D_int * ufl.inner(ufl.grad(c_int), ufl.grad(v_int)) * dx_int  # diffusion

# --- group 3: flux boundary condition -------------------------
J_in = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.5))
F += -J_in * v_left * ds(LEFT_BNDRY_TAG)

# --- group 4: interface coupling: stays on dS ----------------------
ci = c_int(P)
theta = ci / c_int_max
J_left = k1 * c_left(P) * (1 - theta) - k2 * ci
J_right = k3 * c_right(M) * (1 - theta) - k4 * ci

F_coupling = -(J_left + J_right) * v_int(P) * dS(INT_TAG)

emaps = [left_emap, right_emap, int_emap]

residual_1 = ufl.extract_blocks(F)
residual_2 = ufl.extract_blocks(F_coupling)

F_1 = dolfinx.fem.form(residual_1, entity_maps=emaps)
F_2 = dolfinx.fem.form(residual_2, entity_maps=emaps)
for i, f in enumerate(F_2):
    if f is None:
        F_2[i] = dolfinx.fem.form(ufl.ZeroBaseForm((vh[i],)))

du = ufl.TrialFunctions(W)


J_1 = ufl.extract_blocks(  # expand_derivative avoids fem.form assembly conflicts
    expand_derivatives(ufl.derivative(F, (c_left, c_right, c_int), du))
)
J_2 = ufl.extract_blocks(
    expand_derivatives(ufl.derivative(F_coupling, (c_left, c_right, c_int), du))
)

jacobian_1 = dolfinx.fem.form(J_1, entity_maps=emaps)
jacobian_2 = dolfinx.fem.form(J_2, entity_maps=emaps)
for i in range(len(jacobian_2)):
    if jacobian_2[i][i] is None:
        jacobian_2[i][i] = dolfinx.fem.form(ufl.ZeroBaseForm((du[i], vh[i])))

# Dirichlet BCs: right (BeO, x=L) c=0 ; int (int, x=L/2, y=0) c=0
right_dofs = dolfinx.fem.locate_dofs_geometrical(V_right, lambda x: np.isclose(x[0], L))
bc_right = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(0.0), right_dofs, V_right
)

# Interface field: c_int = 0 at the bottom endpoint of the interface (x = x_int, y = 0)
mesh_int.topology.create_connectivity(0, mesh_int.topology.dim)
bottom_vertex = dolfinx.mesh.locate_entities_boundary(
    mesh_int, 0, lambda x: np.isclose(x[1], 0.0)
)
int_dofs = dolfinx.fem.locate_dofs_topological(V_int, 0, bottom_vertex)
bc_int = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), int_dofs, V_int)

bcs = [bc_right, bc_int]

problem = dolfinx.fem.petsc.NonlinearProblem(
    residual_1,
    [c_left, c_right, c_int],
    J=J_1,
    bcs=bcs,
    petsc_options_prefix="interface_coupling",
    entity_maps=emaps,
)

_blocks = problem.b.getAttr("_blocks")
problem.solver.setFunction(
    custom_assemble_residual,
    problem.b,
    kargs={
        "u": (c_left, c_right, c_int),
        "residual": [F_1, F_2],
        "jacobian": [jacobian_1, jacobian_2],
        "bcs": bcs,
        "_blocks": _blocks,
    },
)
problem.solver.setJacobian(
    custom_assemble_jacobian,
    problem.A,
    problem.P_mat,
    kargs={
        "u": (c_left, c_right, c_int),
        "jacobian": [jacobian_1, jacobian_2],
        "preconditioner": None,
        "bcs": bcs,
    },
)

# Time loop
writer_left = dolfinx.io.VTXWriter(left_cells.comm, "results/c_left.bp", [c_left])
writer_right = dolfinx.io.VTXWriter(right_cells.comm, "results/c_right.bp", [c_right])
writer_int = dolfinx.io.VTXWriter(mesh_int.comm, "results/c_int.bp", [c_int])

t = 0.0
writer_left.write(t)
writer_right.write(t)
writer_int.write(t)

n_steps = round(T / dt)
for step in range(n_steps):
    t += dt
    problem.solve()

    c_left_n.x.array[:] = c_left.x.array
    c_right_n.x.array[:] = c_right.x.array
    c_int_n.x.array[:] = c_int.x.array

    writer_left.write(t)
    writer_right.write(t)
    writer_int.write(t)

    if mesh.comm.rank == 0:
        print(f"t = {t:.2f}")

writer_left.close()
writer_right.close()
writer_int.close()
print(c_int.x.array)
