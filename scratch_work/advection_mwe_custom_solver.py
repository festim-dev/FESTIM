from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl
from dolfinx import plot
from petsc_solver_for_manifold_derivatives import (
    custom_assemble_jacobian,
    custom_assemble_residual,
)

dx = 1 / 5
L = 100

nx = int(L / dx)
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([L, 1])],
    [nx, 10],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)
vdim = mesh.topology.dim
fdim = mesh.topology.dim - 1

mesh.topology.create_connectivity(fdim, vdim)

# facet meshtags top and bottom
tag_to_marker = {
    1: lambda x: np.isclose(x[1], 0),  # bottom
    2: lambda x: np.isclose(x[1], 1),  # top
    3: lambda x: np.isclose(x[0], 0),  # left
    4: lambda x: np.isclose(x[0], L),  # right
}

facets = np.array([], dtype=np.int64)
tags = np.array([], dtype=np.int32)
for tag, marker in tag_to_marker.items():
    facet_indices = dolfinx.mesh.locate_entities(mesh, fdim, marker)
    facets = np.concatenate((facets, facet_indices))
    tags = np.concatenate((tags, np.full_like(facet_indices, tag, dtype=np.int32)))
facet_tags = dolfinx.mesh.meshtags(mesh, fdim, facets, tags)

cell_tags = dolfinx.mesh.meshtags(
    mesh,
    vdim,
    np.arange(mesh.topology.index_map(vdim).size_local),
    np.ones(mesh.topology.index_map(vdim).size_local, dtype=np.int32),
)

with dolfinx.io.XDMFFile(mesh.comm, "results/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tags, x=mesh.geometry)

with dolfinx.io.XDMFFile(mesh.comm, "results/cell_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_tags, x=mesh.geometry)

# make submesh of the bottom boundary
submesh, cmap, vmap, nmap = dolfinx.mesh.create_submesh(
    mesh, dim=fdim, entities=facet_tags.find(1)
)
submesh.topology.create_connectivity(0, 1)

# Function spaces and functions
V_bulk = dolfinx.fem.functionspace(mesh, ("CG", 1))
V_sub = dolfinx.fem.functionspace(submesh, ("CG", 1))

W = ufl.MixedFunctionSpace(V_bulk, V_sub)

u = dolfinx.fem.Function(V_bulk)
u.name = "u"
u_sub = dolfinx.fem.Function(V_sub)
u_sub.name = "u_sub"

# Jorgen adds in this new intermediate function for easy callback later
# in the problem assembly
vh = ufl.TestFunctions(W)
v, v_sub = vh

# Formulation
dx = ufl.dx(domain=mesh, subdomain_data=cell_tags)
ds = ufl.ds(domain=mesh, subdomain_data=facet_tags)
dx_sub = ufl.dx(domain=submesh)

F = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
F += ufl.inner(ufl.grad(u_sub), ufl.grad(v_sub)) * dx_sub  # <-- CHANGED

vel_x = 10

# 2D velocity defined at the manifold (in this case y is zero)
vel = dolfinx.fem.Constant(
    submesh,
    PETSc.ScalarType([vel_x, vel_x]),  # <-- note remi had vel_x, vel_x
)
F += ufl.inner(ufl.dot(ufl.grad(u_sub), vel), v_sub) * dx_sub  # <-- CHANGED

# coupling term
h_l = dolfinx.fem.Constant(mesh, 0.4)
flux = h_l * (u - u_sub)

F += flux * v * ds(1)

# NOTE Jorgen said that only gradient terms need the submesh, hence ds(1) below.
# Also he didn't add this to F but created a separate F_coupling term?
F_coupling = -flux * v_sub * ds(1)

# Instead of just forms = ufl.extract_blocks(F), we now have:
residual_1 = ufl.extract_blocks(F)
residual_2 = ufl.extract_blocks(F_coupling)
F_1 = dolfinx.fem.form(residual_1, entity_maps=[cmap])
F_2 = dolfinx.fem.form(residual_2, entity_maps=[cmap])
for i, f in enumerate(F_2):
    if f is None:
        F_2[i] = dolfinx.fem.form(ufl.ZeroBaseForm((vh[i],)))

du = ufl.TrialFunctions(W)
J_1 = ufl.extract_blocks(ufl.derivative(F, (u, u_sub), du))
J_2 = ufl.extract_blocks(ufl.derivative(F_coupling, (u, u_sub), du))
jacobian_1 = dolfinx.fem.form(J_1, entity_maps=[cmap])
jacobian_2 = dolfinx.fem.form(J_2, entity_maps=[cmap])
for i in range(len(jacobian_2)):
    if jacobian_2[i][i] is None:
        jacobian_2[i][i] = dolfinx.fem.form(ufl.ZeroBaseForm((du[i], vh[i])))
# (end of new assembly block)

# Dirichlet BC left
bc_top_dofs = dolfinx.fem.locate_dofs_topological(
    V_bulk,
    mesh.topology.dim - 1,
    dolfinx.mesh.locate_entities(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1)
    ),
)
bc_top = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(0.0),
    bc_top_dofs,
    V_bulk,
)

bc_left_dofs = dolfinx.fem.locate_dofs_topological(
    V_sub, 0, dolfinx.mesh.locate_entities(submesh, 0, lambda x: np.isclose(x[0], 0))
)
bc_left = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(1.0),
    bc_left_dofs,
    V_sub,
)
# Nonlinear problem

problem = dolfinx.fem.petsc.NonlinearProblem(
    residual_1,  # < -- now replaced with residual_1
    [u, u_sub],
    J=J_1,  # now need to specify this explicitly I guess?
    bcs=[
        bc_top,
        bc_left,
    ],
    petsc_options_prefix="codim1_prob",
    entity_maps=[cmap],
)

# NOTE now invoke the customized versions of assemble_residual and assemble_jacobian
_blocks = problem.b.getAttr("_blocks")
cntx = {
    "u": (u, u_sub),
    "residual": [F_1, F_2],
    "jacobian": [jacobian_1, jacobian_2],
    "bcs": [bc_top, bc_left],
    "_blocks": _blocks,
}
problem.solver.setFunction(custom_assemble_residual, problem.b, kargs=cntx)
cntx = {
    "u": (u, u_sub),
    "jacobian": [jacobian_1, jacobian_2],
    "preconditioner": None,
    "bcs": [bc_top, bc_left],
}
problem.solver.setJacobian(
    custom_assemble_jacobian, problem.A, problem.P_mat, kargs=cntx
)

problem.solve()

# Post processing
with dolfinx.io.VTXWriter(mesh.comm, "results/u.bp", [u]) as writer:
    writer.write(0.0)

with dolfinx.io.VTXWriter(submesh.comm, "results/u_sub.bp", [u_sub]) as writer:
    writer.write(0.0)

topology, cell_types, geometry = plot.vtk_mesh(u.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["c"] = u.x.array
grid.set_active_scalars("c")

plotter = pyvista.Plotter()

plotter.add_mesh(grid)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("u.png")

topology, cell_types, geometry = plot.vtk_mesh(u_sub.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["c"] = u_sub.x.array
grid.set_active_scalars("c")

# Make two points to construct the line between
a = [0, 0, 0]
b = [L, 0, 0]
sample = grid.sample_over_line(a, b, resolution=100)

plt.plot(sample["Distance"], sample["c"])
plt.ylim(0, 1)
plt.xlabel("x")
plt.ylabel("u_sub")
plt.show()
