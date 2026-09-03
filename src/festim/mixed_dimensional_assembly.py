"""Assembly of residuals and Jacobians spread over several integration domains.

When a codim-1 (manifold) volume subdomain is present, its residual cannot be written
as a single UFL form: the manifold gradient is only well defined on a measure whose
domain is the submesh, while the terms coupling the manifold field to the bulk field
must be integrated on the parent mesh (a bulk function cannot be resolved inside a
codim-1 integral).  DOLFINx compiles one integration domain per form, so the residual
is split into *groups* -- one blocked form list per integration domain -- and the
groups are accumulated into the same PETSc vector and matrix through custom SNES
callbacks.

The callbacks mirror ``dolfinx.fem.petsc.assemble_residual`` and
``assemble_jacobian``, and are adapted from a script by Jørgen S. Dokken
(https://gist.github.com/jorgensd/af848dd9ab825721ec6696eb84c06b43).  Four things
differ from the upstream implementation, each needed to tolerate a group whose block
structure is incomplete; they are flagged at the point of use below.
"""

from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl
from dolfinx.cpp.fem import petsc as _cpp_fem_petsc
from dolfinx.cpp.la import petsc as _cpp_la_petsc
from dolfinx.fem.assemble import (
    _assemble_vector_array,
    pack_coefficients,
    pack_constants,
)
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.petsc import _extract_function_spaces, apply_lifting, assign, set_bc
from packaging.version import Version

__all__ = [
    "custom_assemble_jacobian",
    "custom_assemble_residual",
    "make_index_sets",
    "prune_empty_blocks",
]


def prune_empty_blocks(blocks):
    """Replace structurally empty Jacobian blocks by ``None``.

    ``ufl.derivative`` of a form with respect to an unknown it does not depend on is
    not automatically simplified away.  Left in place, such a block makes DOLFINx look
    for an entity map relating two sibling submeshes -- which does not exist -- and
    raise ``RuntimeError: Incompatible mesh. argument entity_maps must be provided.``

    Args:
        blocks: nested list of UFL forms (or ``None``)

    Returns:
        The same nested list with empty blocks replaced by ``None``
    """
    out = []
    for row in blocks:
        new_row = []
        for blk in row:
            if blk is None:
                new_row.append(None)
                continue
            blk = ufl.algorithms.expand_derivatives(blk)
            new_row.append(
                None if isinstance(blk, ufl.ZeroBaseForm) or blk.empty() else blk
            )
        out.append(new_row)
    return out


def _get_dofmap(V):
    try:
        # dolfinx >= 0.11
        return V.dofmaps[0]
    except TypeError:
        return V.dofmaps(0)


def make_index_sets(a):
    """PETSc index sets of the blocked layout.

    Built once from the *complete* block structure (the group holding the parent-mesh
    integrals), so that a group with an entirely-``None`` row or column can still be
    assembled into the same matrix.  ``dolfinx.fem.petsc.assemble_matrix`` derives them
    from the group's own function spaces and raises on an all-``None`` row.

    Args:
        a: nested list of compiled bilinear forms

    Returns:
        A ``(is0, is1)`` tuple of row and column index sets
    """
    is0 = _cpp_la_petsc.create_index_sets(
        [
            (_get_dofmap(V).index_map, _get_dofmap(V).index_map_bs)
            for V in _extract_function_spaces(a, 0)
        ]
    )
    is1 = _cpp_la_petsc.create_index_sets(
        [
            (_get_dofmap(V).index_map, _get_dofmap(V).index_map_bs)
            for V in _extract_function_spaces(a, 1)
        ]
    )
    return is0, is1


def _assemble_blocked_vector(b, L):
    """Accumulate one group of blocked linear forms into ``b``.

    ``dolfinx.fem.petsc.assemble_vector`` packs constants and coefficients
    unconditionally and so crashes on a ``None`` block, which a group covering only
    some of the blocks necessarily has.
    """
    offset0, offset1 = b.getAttr("_blocks")
    with b.localForm() as b_l:
        for L_, off0, off1, offg0, offg1 in zip(
            L, offset0, offset0[1:], offset1, offset1[1:], strict=False
        ):
            if L_ is None:
                continue
            bx_ = np.zeros((off1 - off0) + (offg1 - offg0), dtype=PETSc.ScalarType)
            _assemble_vector_array(bx_, L_)
            size = off1 - off0
            b_l.array_w[off0:off1] += bx_[:size]
            b_l.array_w[offg0:offg1] += bx_[size:]
    return b


def _assemble_blocked_matrix(A, a, bcs, is0, is1):
    """Add one group of blocked bilinear forms into ``A`` in ADD mode.

    No flush and no diagonal insertion: those happen once, after every group has been
    accumulated.  Inserting the boundary-condition diagonal between two groups would
    switch the matrix to INSERT mode, and a subsequent group re-entering ADD mode
    silently corrupts the Jacobian.
    """
    consts = [pack_constants(row) for row in a]
    coeffs = [pack_coefficients(row) for row in a]
    _bcs = [bc._cpp_object for bc in bcs]
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is None:
                continue
            Asub = A.getLocalSubMatrix(is0[i], is1[j])
            _cpp_fem_petsc.assemble_matrix(
                Asub, a_sub._cpp_object, consts[i][j], coeffs[i][j], _bcs, True
            )
            A.restoreLocalSubMatrix(is0[i], is1[j], Asub)


def _insert_blocked_diagonal(A, a, bcs, diag, is0, is1):
    """Insert ``diag`` on the rows of ``A`` constrained by ``bcs``.

    Uses INSERT, so it must follow a FLUSH assembly.
    """
    _bcs = [bc._cpp_object for bc in bcs]
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is None:
                continue
            Asub = A.getLocalSubMatrix(is0[i], is1[j])
            if a_sub.function_spaces[0] is a_sub.function_spaces[1]:
                _cpp_fem_petsc.insert_diagonal(
                    Asub, a_sub.function_spaces[0], _bcs, diag
                )
            A.restoreLocalSubMatrix(is0[i], is1[j], Asub)


def custom_assemble_residual(
    _snes, x, b, u, residual_groups, jacobian_groups, bcs, _blocks=None
):
    """Assemble the residual at ``x`` into ``b``, summing over all groups.

    Conforms to the interface expected by ``PETSc.SNES.setFunction`` once the arguments
    after ``b`` are bound through ``kargs``.

    Args:
        _snes: the solver instance (unused)
        x: vector containing the point to evaluate the residual at
        b: vector to assemble the residual into
        u: the functions tied to the solution vector
        residual_groups: one blocked list of compiled residual forms per integration
            domain
        jacobian_groups: the matching Jacobian groups, used for lifting
        bcs: the Dirichlet boundary conditions
        _blocks: ownership layout of the blocked vector
    """
    dolfinx.la.petsc._ghost_update(
        x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )
    assign(x, u)
    if _blocks is not None:
        b.setAttr("_blocks", _blocks)
        x.setAttr("_blocks", _blocks)

    dolfinx.la.petsc._zero_vector(b)
    for residual in residual_groups:
        _assemble_blocked_vector(b, residual)

    for jacobian in jacobian_groups:
        cols = _extract_function_spaces(jacobian, 1)
        if all(V is None for V in cols):
            continue
        bcs1 = _bcs_by_block(cols, bcs)
        if not any(bcs1):
            continue
        apply_lifting(b, jacobian, bcs=bcs1, x0=x, alpha=-1.0)

    dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    bcs0 = _bcs_by_block(_extract_function_spaces(residual_groups[0]), bcs)
    set_bc(b, bcs0, x0=x, alpha=-1.0)
    dolfinx.la.petsc._ghost_update(
        b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )


def custom_assemble_jacobian(
    _snes, x, J, P_mat, u, jacobian_groups, preconditioner, bcs, index_sets
):
    """Assemble the Jacobian into ``J``, summing over all groups.

    Conforms to the interface expected by ``PETSc.SNES.setJacobian`` once the arguments
    after ``P_mat`` are bound through ``kargs``.

    Args:
        _snes: the solver instance (unused)
        x: vector containing the point to evaluate at
        J: matrix to assemble the Jacobian into
        P_mat: matrix to assemble the preconditioner into
        u: the functions tied to the solution vector
        jacobian_groups: one blocked list of compiled Jacobian forms per integration
            domain
        preconditioner: compiled preconditioner groups, or ``None``
        bcs: the Dirichlet boundary conditions
        index_sets: the ``(is0, is1)`` returned by :func:`make_index_sets`
    """
    dolfinx.la.petsc._ghost_update(
        x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD
    )
    assign(x, u)
    is0, is1 = index_sets

    J.zeroEntries()
    for jacobian in jacobian_groups:
        _assemble_blocked_matrix(J, jacobian, bcs, is0, is1)
    # a single flush allows the switch to INSERT mode for the boundary-condition rows
    J.assemble(PETSc.Mat.AssemblyType.FLUSH)
    _insert_blocked_diagonal(J, jacobian_groups[0], bcs, 1.0, is0, is1)
    J.assemble()

    if preconditioner is not None:
        P_mat.zeroEntries()
        for prec in preconditioner:
            _assemble_blocked_matrix(P_mat, prec, bcs, is0, is1)
        P_mat.assemble(PETSc.Mat.AssemblyType.FLUSH)
        _insert_blocked_diagonal(P_mat, preconditioner[0], bcs, 1.0, is0, is1)
        P_mat.assemble()
