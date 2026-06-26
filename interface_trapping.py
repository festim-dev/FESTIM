from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl

from dolfinx.log import set_log_level, LogLevel

set_log_level(LogLevel.INFO)


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
def arrhenius(prefactor, activation_energy, temperature, mod=ufl):
    return prefactor * mod.exp(-activation_energy / (8.617e-5 * temperature))


T = 1250  # K
L = 2.0  # m
x_int = 1.0  # m
D_Be = 3.0e8  # m^2/s
D_BeO = 3.0e8  # m^2/s
lam_be = 2 * 1.577e-10  # m
nu_be_i = arrhenius(3.285e10, 0.231, T) * lam_be  # m/s
nu_i_be = arrhenius(4.104e9, 0.229, T)  # s-1

nu_beo_i = 0.0  # m/s
nu_i_beo = 0.0  # s-1
n_i = 1e20  # at
c_Be_ini = 0.5e28  # D/m3
dt = 0.1e-9  # s
t_f = 20e-9  # s

# ----------------------------------------------------------------------
# Mesh, cell tags (Be / BeO) and facet tags (interface)
# ----------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, 1.0])],
    [20, 10],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)

# mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 20, [0.0, L])

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


# ----------------------------------------------------------------------
# Submeshes (two bulk + one interface) and their entity maps to parent
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Function spaces / functions
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------
dx_Be = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=BE_TAG)
dx_BeO = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=BEO_TAG)
dx_int = ufl.Measure("dx", domain=mesh_int)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facet_tags)


# Initial conditions

c_Be_n.x.array[:] = c_Be_ini

# ----------------------------------------------------------------------
# Residual (backward Euler)
# ----------------------------------------------------------------------
P, M = "+", "-"  # P = Be side, M = BeO side
F = 0

theta_P = c_int(P) / n_i
theta_M = c_int(M) / n_i

# Be bulk:  dc/dt = div(D grad c)  + interface flux  -D grad c . n = k1 cA - k2 ci
F += ((c_Be - c_Be_n) / dt) * v_Be * dx_Be
F += D_Be * ufl.inner(ufl.grad(c_Be), ufl.grad(v_Be)) * dx_Be
F += (nu_be_i * c_Be(P) * (1 - theta_P) - nu_i_be * c_int(P)) * v_Be(P) * dS(INT_TAG)

# BeO bulk:  -D grad c . n = k3 cB - k4 ci
F += ((c_BeO - c_BeO_n) / dt) * v_BeO * dx_BeO
F += D_BeO * ufl.inner(ufl.grad(c_BeO), ufl.grad(v_BeO)) * dx_BeO
F += (
    (nu_beo_i * c_BeO(M) * (1 - theta_M) - nu_i_beo * c_int(P)) * v_BeO(M) * dS(INT_TAG)
)

# Interface:  lambda dci/dt = (k1 cA - k2 ci) + (k3 cB - k4 ci)
F += ((c_int(P) - c_int_n(P)) / dt) * v_int(P) * dS(INT_TAG)
F += (
    -(
        nu_be_i * c_Be(P) * (1 - theta_P)
        - nu_i_be * c_int(P)
        + nu_beo_i * c_BeO(M) * (1 - theta_M)
        - nu_i_beo * c_int(P)
    )
    * v_int(P)
    * dS(INT_TAG)
)

# ----------------------------------------------------------------------
# Blocks + Jacobian
# ----------------------------------------------------------------------
residual = ufl.extract_blocks(F)

du = ufl.TrialFunctions(W)
J = ufl.extract_blocks(ufl.derivative(F, (c_Be, c_BeO, c_int), du))

for i in range(len(J)):
    for j in range(len(J)):
        if J[i][j] is None:
            J[i][j] = ufl.ZeroBaseForm((du[j], vh[i]))

# ----------------------------------------------------------------------
# Dirichlet BCs:  left (Be, x=0) c=1   ;   right (BeO, x=L) c=0
# ----------------------------------------------------------------------
left_dofs = dolfinx.fem.locate_dofs_geometrical(V_Be, lambda x: np.isclose(x[0], 0.0))
bc_left = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(1.0), left_dofs, V_Be)

right_dofs = dolfinx.fem.locate_dofs_geometrical(V_BeO, lambda x: np.isclose(x[0], L))
bc_right = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), right_dofs, V_BeO)

# ----------------------------------------------------------------------
# Nonlinear problem (entity maps relate all submeshes through the parent)
# ----------------------------------------------------------------------
problem = dolfinx.fem.petsc.NonlinearProblem(
    residual,
    [c_Be, c_BeO, c_int],
    J=J,
    # bcs=[bc_left, bc_right],
    petsc_options_prefix="be_beo_",
    entity_maps=[Be_emap, BeO_emap, int_emap],
    petsc_options={"snes_monitor": None, "snes_atol": 1e-10, "snes_rtol": 1e-10},
)

# ----------------------------------------------------------------------
# Time loop
# ----------------------------------------------------------------------
writer_Be = dolfinx.io.VTXWriter(mesh_Be.comm, "results/c_Be.bp", [c_Be])
writer_BeO = dolfinx.io.VTXWriter(mesh_BeO.comm, "results/c_BeO.bp", [c_BeO])
writer_int = dolfinx.io.VTXWriter(mesh_int.comm, "results/c_int.bp", [c_int])

t = 0.0
writer_Be.write(t)
writer_BeO.write(t)
writer_int.write(t)


all_ci = [0.0]
all_t = [t]

n_steps = int(round(t_f / dt))
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

    all_ci.append(c_int.x.array.mean())
    all_t.append(t)

writer_Be.close()
writer_BeO.close()
writer_int.close()
print(c_int.x.array)

import matplotlib.pyplot as plt

plt.plot(all_t, all_ci)
plt.xlabel("t")
plt.ylabel("c_int (mean)")
plt.show()
