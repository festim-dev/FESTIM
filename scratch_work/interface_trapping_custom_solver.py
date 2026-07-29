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
L = 10.0
x_int = 5.0
D_Be = 1.0
D_BeO = 1.0
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0
lam = 1.0
c_int_max = 1.0
D_int = 1.0  # diffusivity along the interface

dt = 0.1
T = 10.0

# Mesh, cell tags (Be / BeO) and facet tags (interface)
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, 1.0])],
    [20, 10],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)


vdim = mesh.topology.dim
fdim = vdim - 1
mesh.topology.create_connectivity(fdim, vdim)

eps = 1e-10
BE_TAG, BEO_TAG = 1, 2
be_cells = dolfinx.mesh.locate_entities(mesh, vdim, lambda x: x[0] <= x_int + eps)
beo_cells = dolfinx.mesh.locate_entities(mesh, vdim, lambda x: x[0] >= x_int - eps)
cell_indices = np.concatenate([be_cells, beo_cells])
cell_values = np.concatenate(
    [np.full_like(be_cells, BE_TAG), np.full_like(beo_cells, BEO_TAG)]
).astype(np.int32)
sort = np.argsort(cell_indices)
cell_tags = dolfinx.mesh.meshtags(mesh, vdim, cell_indices[sort], cell_values[sort])

INT_TAG = 3
int_facets = dolfinx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], x_int))
facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, int_facets, np.full_like(int_facets, INT_TAG, dtype=np.int32)
)


# Submeshes (two bulk + one interface) and their entity maps to parent
mesh_Be, Be_emap, _, _ = dolfinx.mesh.create_submesh(mesh, vdim, cell_tags.find(BE_TAG))
mesh_BeO, BeO_emap, _, _ = dolfinx.mesh.create_submesh(
    mesh, vdim, cell_tags.find(BEO_TAG)
)
mesh_int, int_emap, _, _ = dolfinx.mesh.create_submesh(
    mesh, fdim, facet_tags.find(INT_TAG)
)

# ----- build interface integration entities ordered so "+" = Be, "-" = BeO -----
imap = mesh.topology.index_map(vdim)
n_cells = imap.size_local + imap.num_ghosts
cell_marker = np.zeros(n_cells, dtype=np.int32)
cell_marker[cell_tags.indices] = cell_tags.values

f2c = mesh.topology.connectivity(fdim, vdim)
c2f = mesh.topology.connectivity(vdim, fdim)
ints = []
for f in facet_tags.find(INT_TAG):
    c0, c1 = f2c.links(f)
    if cell_marker[c0] == BEO_TAG:  # ensure Be cell first
        c0, c1 = c1, c0
    lf0 = np.where(c2f.links(c0) == f)[0][0]
    lf1 = np.where(c2f.links(c1) == f)[0][0]
    ints += [c0, lf0, c1, lf1]
ints = np.array(ints, dtype=np.int32)


# Function spaces / functions
V_Be = dolfinx.fem.functionspace(mesh_Be, ("CG", 1))
V_BeO = dolfinx.fem.functionspace(mesh_BeO, ("CG", 1))
V_int = dolfinx.fem.functionspace(mesh_int, ("CG", 1))

W = ufl.MixedFunctionSpace(V_Be, V_BeO, V_int)

c_Be = dolfinx.fem.Function(V_Be, name="c_Be")
c_BeO = dolfinx.fem.Function(V_BeO, name="c_BeO")
c_int = dolfinx.fem.Function(V_int, name="c_int")

# previous time step (initial condition = 0)
c_Be_n = dolfinx.fem.Function(V_Be)
c_BeO_n = dolfinx.fem.Function(V_BeO)
c_int_n = dolfinx.fem.Function(V_int)

vh = ufl.TestFunctions(W)
v_Be, v_BeO, v_int = vh

# Measures
dx_Be = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=BE_TAG)
dx_BeO = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=BEO_TAG)
dx_int = ufl.Measure("dx", domain=mesh_int)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facet_tags)

# Residual (backward Euler)
P, M = "+", "-"  # P = Be side, M = BeO side

theta_P = c_int(P) / c_int_max
theta_M = c_int(M) / c_int_max

# --- group 1: integration domain = parent mesh ---------------------
F = 0
F += ((c_Be - c_Be_n) / dt) * v_Be * dx_Be
F += D_Be * ufl.inner(ufl.grad(c_Be), ufl.grad(v_Be)) * dx_Be
F += (k1 * c_Be(P) * (1 - theta_P) - k2 * c_int(P)) * v_Be(P) * dS(INT_TAG)

F += ((c_BeO - c_BeO_n) / dt) * v_BeO * dx_BeO
F += D_BeO * ufl.inner(ufl.grad(c_BeO), ufl.grad(v_BeO)) * dx_BeO
F += (k3 * c_BeO(M) * (1 - theta_M) - k4 * c_int(M)) * v_BeO(M) * dS(INT_TAG)

# --- group 2: integration domain = mesh_int ------------------------
F += lam * ((c_int - c_int_n) / dt) * v_int * dx_int  # trapping
F += D_int * ufl.inner(ufl.grad(c_int), ufl.grad(v_int)) * dx_int  # diffusion

# interface coupling: stays on dS
F_coupling = (
    -(
        k1 * c_Be(P) * (1 - theta_P)
        - k2 * c_int(P)
        + k3 * c_BeO(M) * (1 - theta_M)
        - k4 * c_int(M)
    )
    * v_int(P)
    * dS(INT_TAG)
)

emaps = [Be_emap, BeO_emap, int_emap]

residual_1 = ufl.extract_blocks(F)
residual_2 = ufl.extract_blocks(F_coupling)

F_1 = dolfinx.fem.form(residual_1, entity_maps=emaps)
F_2 = dolfinx.fem.form(residual_2, entity_maps=emaps)
for i, f in enumerate(F_2):
    if f is None:
        F_2[i] = dolfinx.fem.form(ufl.ZeroBaseForm((vh[i],)))

du = ufl.TrialFunctions(W)


J_1 = ufl.extract_blocks(  # expand_derivative avoids fem.form assembly conflicts
    expand_derivatives(ufl.derivative(F, (c_Be, c_BeO, c_int), du))
)
J_2 = ufl.extract_blocks(
    expand_derivatives(ufl.derivative(F_coupling, (c_Be, c_BeO, c_int), du))
)

jacobian_1 = dolfinx.fem.form(J_1, entity_maps=emaps)
jacobian_2 = dolfinx.fem.form(J_2, entity_maps=emaps)
for i in range(len(jacobian_2)):
    if jacobian_2[i][i] is None:
        jacobian_2[i][i] = dolfinx.fem.form(ufl.ZeroBaseForm((du[i], vh[i])))


# Dirichlet BCs:  left (Be, x=0) c=1   ;   right (BeO, x=L) c=0
left_dofs = dolfinx.fem.locate_dofs_geometrical(V_Be, lambda x: np.isclose(x[0], 0.0))
bc_left = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(1.0), left_dofs, V_Be)

right_dofs = dolfinx.fem.locate_dofs_geometrical(V_BeO, lambda x: np.isclose(x[0], L))
bc_right = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), right_dofs, V_BeO)

problem = dolfinx.fem.petsc.NonlinearProblem(
    residual_1,
    [c_Be, c_BeO, c_int],
    J=J_1,
    bcs=[bc_left, bc_right],
    petsc_options_prefix="be_beo_",
    entity_maps=emaps,
)

_blocks = problem.b.getAttr("_blocks")
problem.solver.setFunction(
    custom_assemble_residual,
    problem.b,
    kargs={
        "u": (c_Be, c_BeO, c_int),
        "residual": [F_1, F_2],
        "jacobian": [jacobian_1, jacobian_2],
        "bcs": [bc_left, bc_right],
        "_blocks": _blocks,
    },
)
problem.solver.setJacobian(
    custom_assemble_jacobian,
    problem.A,
    problem.P_mat,
    kargs={
        "u": (c_Be, c_BeO, c_int),
        "jacobian": [jacobian_1, jacobian_2],
        "preconditioner": None,
        "bcs": [bc_left, bc_right],
    },
)

# Time loop
writer_Be = dolfinx.io.VTXWriter(mesh_Be.comm, "results/c_Be.bp", [c_Be])
writer_BeO = dolfinx.io.VTXWriter(mesh_BeO.comm, "results/c_BeO.bp", [c_BeO])
writer_int = dolfinx.io.VTXWriter(mesh_int.comm, "results/c_int.bp", [c_int])

t = 0.0
writer_Be.write(t)
writer_BeO.write(t)
writer_int.write(t)

n_steps = round(T / dt)
for step in range(n_steps):
    t += dt
    problem.solve()

    c_Be_n.x.array[:] = c_Be.x.array
    c_BeO_n.x.array[:] = c_BeO.x.array
    c_int_n.x.array[:] = c_int.x.array

    writer_Be.write(t)
    writer_BeO.write(t)
    writer_int.write(t)

    if mesh.comm.rank == 0:
        print(f"t = {t:.2f}")

writer_Be.close()
writer_BeO.close()
writer_int.close()
print(c_int.x.array)
