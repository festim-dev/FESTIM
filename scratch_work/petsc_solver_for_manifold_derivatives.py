import typing

from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc


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
