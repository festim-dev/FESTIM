import typing

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl


def custom_assemble_residual(
    _snes: PETSc.SNES,  # type: ignore[name-defined]
    x: PETSc.Vec,  # type: ignore[name-defined]
    b: PETSc.Vec,  # type: ignore[name-defined]
    u: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function],
    residual: typing.Sequence[typing.Sequence[dolfinx.fem.Form]],
    jacobian: typing.Sequence[typing.Sequence[typing.Sequence[dolfinx.fem.Form]]],
    bcs: typing.Sequence[dolfinx.fem.DirichletBC],
    _blocks: tuple[tuple[int, int, int], ...] | None = None,
):
    """Assemble the residual at ``x`` into the vector ``b``.
    A function conforming to the interface expected by ``SNES.setFunction``
    by setting all arguments except `snes`, `x` and `b` through the `kargs`
    keyword argument.
    Example::
        snes = PETSc.SNES().create(mesh.comm)
        cntx = {"u": u, "residual": residual, "jacobian": jacobian,
            "bcs": bcs}
        snes.setFunction(assemble_residual, b, kargs=cntx)
    Args:
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        b: Vector to assemble the residual into.
        u: Function(s) tied to the solution vector within the residual and
           Jacobian.
        residual: Form of the residual. It can be a sequence of forms.
        jacobian: Form of the Jacobian. It can be a nested sequence of
            forms.
        bcs: List of Dirichlet boundary conditions to lift the residual.
        _blocks: If block assembly is requested this should contain the
            ownership layout for each block.
            See :func:`dolfinx.la.create_vector` for more details on the
            format of this argument.
    """
    # Update input vector before assigning
    dolfinx.la.petsc._ghost_update(
        x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )  # type: ignore[attr-defined]

    # Assign the input vector to the unknowns
    dolfinx.fem.petsc.assign(x, u)

    # Assign block data if block assembly is requested
    if (
        isinstance(residual, typing.Sequence)
        and isinstance(residual[0], typing.Sequence)
        and b.getType() != PETSc.Vec.Type.NEST
    ):  # type: ignore[attr-defined]
        assert _blocks is not None, "Block data must be provided for block assembly."
        b.setAttr("_blocks", _blocks)  # type: ignore[attr-defined]
        x.setAttr("_blocks", _blocks)  # type: ignore[attr-defined]

    # Assemble the residual
    dolfinx.la.petsc._zero_vector(b)
    for res in residual:
        dolfinx.fem.petsc.assemble_vector(b, res)  # type: ignore[arg-type]

    # Lift vector
    if isinstance(jacobian[0], typing.Sequence):
        for jac in jacobian:
            # Nest and blocked lifting
            bcs1 = dolfinx.fem.bcs.bcs_by_block(
                dolfinx.fem.forms.extract_function_spaces(jac, 1), bcs
            )  # type: ignore[arg-type]
            dolfinx.fem.petsc.apply_lifting(b, jac, bcs=bcs1, x0=x, alpha=-1.0)
        dolfinx.la.petsc._ghost_update(
            b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE
        )  # type: ignore[attr-defined]
        bcs0 = dolfinx.fem.bcs.bcs_by_block(
            dolfinx.fem.forms.extract_function_spaces(residual[0]), bcs
        )  # type: ignore[arg-type]
        dolfinx.fem.petsc.set_bc(b, bcs0, x0=x, alpha=-1.0)
    else:
        # Single form lifting
        dolfinx.fem.petsc.apply_lifting(b, [jacobian], bcs=[bcs], x0=[x], alpha=-1.0)
        dolfinx.la.petsc._ghost_update(
            b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE
        )  # type: ignore[attr-defined]
        dolfinx.fem.petsc.set_bc(b, bcs, x0=x, alpha=-1.0)
    dolfinx.la.petsc._ghost_update(
        b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )  # type: ignore[attr-defined]


def custom_assemble_jacobian(
    _snes: PETSc.SNES,  # type: ignore[name-defined]
    x: PETSc.Vec,  # type: ignore[name-defined]
    J: PETSc.Mat,  # type: ignore[name-defined]
    P_mat: PETSc.Mat,  # type: ignore[name-defined]
    u: typing.Sequence[dolfinx.fem.Function] | dolfinx.fem.Function,
    jacobian: dolfinx.fem.Form
    | typing.Sequence[typing.Sequence[typing.Sequence[dolfinx.fem.Form]]],
    preconditioner: dolfinx.fem.Form
    | typing.Sequence[typing.Sequence[typing.Sequence[dolfinx.fem.Form]]]
    | None,
    bcs: typing.Sequence[dolfinx.fem.DirichletBC],
):
    """Assemble the Jacobian and preconditioner matrices.
    A function conforming to the interface expected by
    ``SNES.setJacobian`` can be created by fixing the first four
    A function conforming to the interface expected by
    ``SNES.setJacobian`` can be created by setting all
    arguments except `_snes`, `x`, `J` and `P_mat`
    through the `kargs` argument e.g.:
    Example::
        snes = PETSc.SNES().create(mesh.comm)
        cntx = {"u": u, "jacobian": jacobian,
            "preconditioner": preconditioner, "bcs": bcs}
        snes.setJacobian(assemble_jacobian, A, P_mat, kargs=cntx)
    Args:
        _snes: The solver instance.
        x: Vector containing the point to evaluate at.
        J: Matrix to assemble the Jacobian into.
        P_mat: Matrix to assemble the preconditioner into.
        u: Function tied to the solution vector within the residual and
            Jacobian.
        jacobian: Compiled form of the Jacobian.
        preconditioner: Compiled form of the preconditioner.
        bcs: List of Dirichlet boundary conditions to apply to the Jacobian
            and preconditioner matrices.
    """
    # Copy existing soultion into the function used in the residual and
    # Jacobian
    dolfinx.la.petsc._ghost_update(
        x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )  # type: ignore[attr-defined]
    dolfinx.fem.petsc.assign(x, u)

    # Assemble Jacobian
    J.zeroEntries()
    if isinstance(jacobian[0][0], typing.Sequence):
        for jac in jacobian:
            dolfinx.fem.petsc.assemble_matrix(J, jac, bcs, diag=1.0)  # type: ignore[arg-type, misc]
    else:
        dolfinx.fem.petsc.assemble_matrix(J, jacobian, bcs, diag=1.0)  # type: ignore[arg-type, misc]
    J.assemble()
    if preconditioner is not None:
        P_mat.zeroEntries()
        if isinstance(preconditioner[0][0], typing.Sequence):
            for prec in preconditioner:
                dolfinx.fem.petsc.assemble_matrix(P_mat, prec, bcs, diag=1.0)  # type: ignore[arg-type, misc]
        else:
            dolfinx.fem.petsc.assemble_matrix(P_mat, preconditioner, bcs, diag=1.0)  # type: ignore[arg-type, misc]
        P_mat.assemble()


mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([10, 1])],
    [10, 10],
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
    4: lambda x: np.isclose(x[0], 10),  # right
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
u_sub = dolfinx.fem.Function(V_sub)

vh = ufl.TestFunctions(W)
v, v_sub = vh

# Formulation
dx = ufl.dx(domain=mesh, subdomain_data=cell_tags)
dx_sub = ufl.dx(domain=submesh)
ds = ufl.ds(domain=mesh, subdomain_data=facet_tags)

F = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
F += ufl.inner(ufl.grad(u_sub), ufl.grad(v_sub)) * dx_sub

vel_x = 0.1
vel = dolfinx.fem.Constant(submesh, dolfinx.default_scalar_type([vel_x, 0]))
F += ufl.inner(ufl.dot(ufl.grad(u_sub), vel), v_sub) * dx_sub

# coupling term
h_l = dolfinx.fem.Constant(submesh, 0.4)
flux = h_l * (u - u_sub)

F += flux * v * ds(1)
F_coupling = -flux * v_sub * ds(1)

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
    residual_1,
    [u, u_sub],
    J=J_1,
    bcs=[
        bc_top,
        bc_left,
    ],
    petsc_options_prefix="codim1_prob",
    entity_maps=[cmap],
)
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


with dolfinx.io.VTXWriter(mesh.comm, "results/u.bp", [u]) as writer:
    writer.write(0.0)

with dolfinx.io.VTXWriter(submesh.comm, "results/u_sub.bp", [u_sub]) as writer:
    writer.write(0.0)

print(u.x.array)
print(u_sub.x.array)
