"""
Left / interface / right coupled diffusion problem with

  1) a prescribed incoming flux at x = 0,
  2) a reaction-based flux balance at the three boundaries that exchange mass
     with the outside world (inlet, right wall, interface bottom vertex), and
  3) a sweep of the interface diffusivity D_int = 0, 1, ..., 10 with a plot of
     the three fluxes at t = T.

Targets dolfinx 0.10.x. Requires petsc_solver_for_manifold_derivatives.py
(custom SNES callbacks for the split residual/Jacobian).
"""

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

# ---------------------------------------------------------------- parameters
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

dt = 0.1  # stepsize
T = 100.0  # final time

# --- sweep / output control ------------------------------------------------
D_INT_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
WRITE_FIELDS = False  # VTX output for every sweep point (slow)
VERBOSE_STEPS = False  # per-timestep balance table
PLOT_FILE = "flux_vs_D_int.png"

# ---------------------------------------------------------------- mesh / tags
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

# Create mesh tag for the left boundary (flux condition)
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

# Submeshes (two bulk + one interface) and their entity maps to parent.
# NOTE: left_cells / right_cells are rebound here from index arrays to meshes.
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

# ---------------------------------------------------------------- spaces
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

# ---------------------------------------------------------------- measures
dx_left = ufl.Measure(
    "dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=LEFT_VOL_TAG
)
dx_right = ufl.Measure(
    "dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=RIGHT_VOL_TAG
)
dx_int = ufl.Measure("dx", domain=mesh_int)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=int_facet_tags)  # internal facets
ds = ufl.Measure("ds", domain=mesh, subdomain_data=left_facet_tags)  # external facets

# submesh-local cell measures, used only for the inventory forms
dx_left_sub = ufl.Measure("dx", domain=left_cells)
dx_right_sub = ufl.Measure("dx", domain=right_cells)

# --- SWEEP PARAMETER: a Constant, so the forms are compiled only once. It
#     lives on mesh_int because that is the integration domain of its term.
D_int = dolfinx.fem.Constant(mesh_int, dolfinx.default_scalar_type(1.0))

# ---------------------------------------------------------------- residual
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

# --- group 3: flux boundary condition ------------------------------
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

# ---------------------------------------------------------------- BCs
right_dofs = dolfinx.fem.locate_dofs_geometrical(V_right, lambda x: np.isclose(x[0], L))
bc_right = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(0.0), right_dofs, V_right
)

# Interface field: c_int = 0 at the bottom endpoint of the interface
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

# ============================================================ flux balance
# Reaction (residual) method: after Newton convergence the residual vanishes
# for every test function that vanishes on the constrained dofs, so evaluating
# a residual block with the *indicator of the constrained dofs* as test
# function returns the omitted boundary term, i.e. the discretely conservative
# outward flux through that constraint.
#
# IMPORTANT vs. the single-form version: the c_int equation is split across
# residual_1[2] (dx_int: storage + interface diffusion) and residual_2[2]
# (dS: coupling). BOTH must be evaluated, or Phi_int is wrong.

BE, RI, IN = 0, 1, 2  # block order follows MixedFunctionSpace(V_left, V_right, V_int)


def _reaction_form(block, v, w, with_emaps):
    """Residual block with its test function replaced by the indicator w.

    ufl.action() cannot be used: extract_blocks keeps the `part` numbering on
    the arguments and compute_form_action then tries to index the coefficient
    by part(). ufl.replace handles it, restrictions included.
    """
    replaced = ufl.replace(block, {v: w})
    if with_emaps:
        return dolfinx.fem.form(replaced, entity_maps=emaps)
    return dolfinx.fem.form(replaced)


# indicators of the constrained dofs
w_right = dolfinx.fem.Function(V_right)
w_right.x.array[right_dofs] = 1.0
w_right.x.scatter_forward()

w_int = dolfinx.fem.Function(V_int)
w_int.x.array[int_dofs] = 1.0
w_int.x.scatter_forward()

# right wall (x = L): only the parent-mesh block, needs entity maps
form_flux_right = _reaction_form(residual_1[RI], v_right, w_right, with_emaps=True)

# interface bottom vertex: two contributions
form_flux_int_dx = _reaction_form(residual_1[IN], v_int, w_int, with_emaps=False)
form_flux_int_dS = (
    _reaction_form(residual_2[IN], v_int, w_int, with_emaps=True)
    if residual_2[IN] is not None
    else None
)

# inlet (x = 0): the prescribed flux, integrated over the boundary
form_flux_left = dolfinx.fem.form(J_in * ds(LEFT_BNDRY_TAG))

# storage terms with v = 1, using the SAME backward-Euler difference as the
# residual so the balance closes algebraically rather than approximately
form_dm_left = dolfinx.fem.form((c_left - c_left_n) / dt * dx_left_sub)
form_dm_right = dolfinx.fem.form((c_right - c_right_n) / dt * dx_right_sub)
form_dm_int = dolfinx.fem.form(lam * (c_int - c_int_n) / dt * dx_int)

# total inventory, for the cumulative check
form_m_left = dolfinx.fem.form(c_left * dx_left_sub)
form_m_right = dolfinx.fem.form(c_right * dx_right_sub)
form_m_int = dolfinx.fem.form(lam * c_int * dx_int)


def gather(compiled_form):
    """Assemble a rank-0 form and sum over ranks."""
    if compiled_form is None:
        return 0.0
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(compiled_form), op=MPI.SUM)


def fluxes():
    """(inlet, right wall, interface bottom vertex), positive = leaving."""
    phi_in = gather(form_flux_left)
    phi_right = -gather(form_flux_right)
    phi_int = -(gather(form_flux_int_dx) + gather(form_flux_int_dS))
    return phi_in, phi_right, phi_int


def d_inventory_dt():
    return gather(form_dm_left) + gather(form_dm_right) + gather(form_dm_int)


def inventory():
    return gather(form_m_left) + gather(form_m_right) + gather(form_m_int)


def reset_state():
    for f in (c_left, c_right, c_int, c_left_n, c_right_n, c_int_n):
        f.x.array[:] = 0.0
        f.x.scatter_forward()


# ============================================================ sweep
def run_transient(d_int_value):
    """Run 0 -> T for one D_int and return the diagnostics at t = T."""
    D_int.value = dolfinx.default_scalar_type(d_int_value)
    reset_state()

    writers = []
    if WRITE_FIELDS:
        tag = f"{d_int_value:g}".replace(".", "p")
        writers = [
            dolfinx.io.VTXWriter(
                left_cells.comm, f"results/D_int_{tag}/c_left.bp", [c_left]
            ),
            dolfinx.io.VTXWriter(
                right_cells.comm, f"results/D_int_{tag}/c_right.bp", [c_right]
            ),
            dolfinx.io.VTXWriter(
                mesh_int.comm, f"results/D_int_{tag}/c_int.bp", [c_int]
            ),
        ]

    t = 0.0
    m0 = inventory()
    cum_in = cum_out = 0.0
    for w in writers:
        w.write(t)

    if VERBOSE_STEPS and mesh.comm.rank == 0:
        print(
            f"{'t':>6} {'Phi_in':>11} {'Phi_right':>11} {'Phi_int':>11} "
            f"{'dInv/dt':>11} {'imbalance':>11}"
        )

    n_steps = round(T / dt)
    for _ in range(n_steps):
        t += dt
        problem.solve()
        reason = problem.solver.getConvergedReason()
        if reason <= 0 and mesh.comm.rank == 0:
            print(f"  ! SNES diverged at t = {t:.2f} (reason {reason})")

        # must be evaluated BEFORE overwriting the *_n functions
        phi_in, phi_right, phi_int = fluxes()
        dinv = d_inventory_dt()
        imbalance = phi_in - phi_right - phi_int - dinv
        cum_in += phi_in * dt
        cum_out += (phi_right + phi_int) * dt

        c_left_n.x.array[:] = c_left.x.array
        c_right_n.x.array[:] = c_right.x.array
        c_int_n.x.array[:] = c_int.x.array

        for w in writers:
            w.write(t)

        if VERBOSE_STEPS and mesh.comm.rank == 0:
            print(
                f"{t:6.2f} {phi_in:11.4e} {phi_right:11.4e} {phi_int:11.4e} "
                f"{dinv:11.4e} {imbalance:11.3e}"
            )

    for w in writers:
        w.close()

    return {
        "D_int": d_int_value,
        "phi_in": phi_in,
        "phi_right": phi_right,
        "phi_int": phi_int,
        "dinv_dt": dinv,
        "imbalance": imbalance,
        "cum_imbalance": cum_in - cum_out - (inventory() - m0),
    }


results = []
if mesh.comm.rank == 0:
    print(
        f"{'D_int':>6} {'Phi_in':>11} {'Phi_right':>11} {'Phi_int':>11} "
        f"{'dInv/dt':>11} {'imbal':>10} {'cum imbal':>10}"
    )
for d in D_INT_VALUES:
    r = run_transient(float(d))
    results.append(r)
    if mesh.comm.rank == 0:
        print(
            f"{r['D_int']:6.1f} {r['phi_in']:11.4e} {r['phi_right']:11.4e} "
            f"{r['phi_int']:11.4e} {r['dinv_dt']:11.4e} "
            f"{r['imbalance']:10.2e} {r['cum_imbalance']:10.2e}"
        )

# ============================================================ output
if mesh.comm.rank == 0:
    d_vals = np.array([r["D_int"] for r in results])
    phi_in = np.array([r["phi_in"] for r in results])
    phi_right = np.array([r["phi_right"] for r in results])
    phi_int = np.array([r["phi_int"] for r in results])
    dinv = np.array([r["dinv_dt"] for r in results])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.semilogx(d_vals, phi_in, "o-", label=r"inlet, $x=0$ (imposed)")
    ax.semilogx(d_vals, phi_right, "s-", label=r"right wall, $x=L$")
    ax.semilogx(d_vals, phi_int, "^-", label="interface bottom vertex")

    ax.set_xlabel(r"interface diffusivity $D_{int}$")
    ax.set_ylabel(f"flux at t = {T:g}   (positive = leaving)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=200)
