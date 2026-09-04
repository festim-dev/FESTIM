"""Cell-wise error indicators and Dorfler marking for the AMR prototype.

The indicator follows the shape used in the DOLFINx AMR tutorial
(https://jsdokken.com/dolfinx-tutorial/chapter2/amr.html): a DG0 assembly of a cell
term plus an interior-facet jump term. For a reaction-dominated trapping front there
is no cheap residual estimator worth trusting, so what is used here is a *gradient*
indicator -- it marks where the solution varies, which is exactly where a moving front
needs resolution.
"""

from collections.abc import Callable

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.cpp.mesh import cell_num_entities


def cell_indicator(u: fem.Function, kind: str = "gradient_jump") -> np.ndarray:
    """Cell-wise error indicator for a scalar field.

    Args:
        u: the field to measure, a real collapsed ``fem.Function`` (not a ``ufl.split``
            view of a mixed function -- those cannot be assembled on their own).
        kind: ``"gradient_jump"`` for ``h^2|grad u|^2 + (h/2)|[grad u . n]|^2``, or
            ``"gradient"`` for the cell term alone (cheaper, no ``dS`` assembly).

    Returns:
        one non-negative value per cell of ``u``'s mesh, indexed by cell (owned cells
        first, as DG0 dofs are).
    """
    mesh = u.function_space.mesh
    W = fem.functionspace(mesh, ("DG", 0))
    w = ufl.TestFunction(W)

    h = fem.Function(W)
    h.x.array[:] = mesh.h(mesh.topology.dim, np.arange(len(h.x.array), dtype=np.int32))
    n = ufl.FacetNormal(mesh)

    G = ufl.inner(h**2 * ufl.inner(ufl.grad(u), ufl.grad(u)), w) * ufl.dx
    if kind == "gradient_jump":
        # a 1D interval mesh has no interior facets in the ufl sense that dS can use
        # them cheaply, but the term is still well defined and assembles fine
        jump = ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(u), n))
        G += ufl.inner(h("+") / 2 * jump, w("+")) * ufl.dS
        G += ufl.inner(h("-") / 2 * jump, w("-")) * ufl.dS
    elif kind != "gradient":
        raise ValueError(f"unknown indicator kind {kind!r}")

    eta_squared = fem.assemble_vector(fem.form(G))
    eta_squared.scatter_reverse(dolfinx.la.InsertMode.add)
    return np.sqrt(np.maximum(eta_squared.array, 0.0))


def variation_indicator(u: fem.Function, scale: float) -> np.ndarray:
    """Relative change of ``u`` across each cell, as a fraction of ``scale``.

    Unlike :func:`cell_indicator` this is an *absolute*, mesh-independent measure with
    a physical reading -- "the concentration changes by 5% across this cell" -- so it
    can be thresholded directly. A purely relative criterion such as Dorfler marking
    always has a top 15% to refine, so it never stops; that is fine for a steady-state
    AMR loop driven to a tolerance, and wrong for a transient one whose mesh has to
    reach a steady size behind a moving front.

    For a P1 field the spread of the vertex values over a cell *is* ``h_K |grad u|``,
    so this reads straight off the dofmap. That matters: assembling a DG0 form instead
    cost 60 ms per adapt in the trapping demo, several times the cost of the timestep
    it was deciding about.

    Args:
        u: a real collapsed ``fem.Function`` in a P1 space.
        scale: the concentration scale to measure the change against, normally the
            global maximum of ``u``.

    Returns:
        one value per cell (owned cells first, as cell-indexed arrays are).
    """
    V = u.function_space
    mesh = V.mesh
    cmap = mesh.topology.index_map(mesh.topology.dim)
    n_cells = cmap.size_local + cmap.num_ghosts

    if scale <= 0:
        return np.zeros(n_cells)

    n_vertices = cell_num_entities(mesh.topology.cell_type, 0)
    dofs_per_cell = V.dofmap.list.shape[1]
    if dofs_per_cell != n_vertices:
        raise ValueError(
            "variation_indicator assumes a P1 space (one dof per vertex); got "
            f"{dofs_per_cell} dofs on a cell with {n_vertices} vertices"
        )

    values = u.x.array[V.dofmap.list[:n_cells]]
    return (values.max(axis=1) - values.min(axis=1)) / scale


def _submesh_to_parent(
    subdomain, values: np.ndarray, n_parent_cells: int
) -> np.ndarray:
    """Scatter a per-submesh-cell array onto the parent mesh's cells."""
    n_sub = len(values)
    sub_cells = np.arange(n_sub, dtype=np.int32)
    parent_cells = subdomain.cell_map.sub_topology_to_topology(sub_cells, False)
    out = np.zeros(n_parent_cells, dtype=np.float64)
    # a parent cell belongs to exactly one codim-0 subdomain, so no accumulation needed
    out[parent_cells] = values
    return out


def problem_indicator(
    problem,
    species_names: list[str] | None = None,
    kind: str = "gradient_jump",
    normalise: bool = True,
) -> np.ndarray:
    """Indicator for a ``HydrogenTransportProblemDiscontinuous``, on the parent mesh.

    The problem's fields live on per-subdomain submeshes, so the indicator is computed
    on each submesh and scattered back onto the parent cells.

    Species concentrations differ by orders of magnitude (mobile vs trapped), so each
    species' indicator is normalised by its own maximum before they are combined, or
    a single trap would decide the whole mesh.

    Args:
        problem: an initialised ``HydrogenTransportProblemDiscontinuous``.
        species_names: which species to look at. Defaults to all of them.
        kind: passed to :func:`cell_indicator`.
        normalise: divide each species' indicator by its global max before combining.

    Returns:
        one value per cell of the parent mesh.
    """
    comm = problem.mesh.mesh.comm
    vdim = problem.mesh.vdim
    cmap = problem.mesh.mesh.topology.index_map(vdim)
    n_parent = cmap.size_local + cmap.num_ghosts

    species = problem.species
    if species_names is not None:
        species = [s for s in species if s.name in species_names]

    combined = np.zeros(n_parent, dtype=np.float64)
    for spe in species:
        per_species = np.zeros(n_parent, dtype=np.float64)
        scale = _species_scale(problem, spe, comm) if kind == "variation" else 0.0

        for subdomain in problem.volume_subdomains:
            u = spe.subdomain_to_post_processing_solution.get(subdomain)
            if u is None:
                continue
            if kind == "variation":
                eta = variation_indicator(u, scale)
            else:
                eta = cell_indicator(u, kind=kind)
            per_species += _submesh_to_parent(subdomain, eta, n_parent)

        if normalise and kind != "variation":
            # "variation" is already on an absolute scale; normalising would throw
            # away exactly the property it exists for
            local_max = per_species.max() if per_species.size else 0.0
            global_max = comm.allreduce(local_max, op=MPI.MAX)
            if global_max > 0:
                per_species /= global_max

        combined = np.maximum(combined, per_species)

    return combined


def _species_scale(problem, species, comm) -> float:
    """Global maximum of a species' concentration, across every subdomain."""
    local = 0.0
    for subdomain in problem.volume_subdomains:
        u = species.subdomain_to_post_processing_solution.get(subdomain)
        if u is not None and u.x.array.size:
            local = max(local, float(np.abs(u.x.array).max()))
    return comm.allreduce(local, op=MPI.MAX)


def dorfler_mark(
    eta: np.ndarray,
    comm,
    theta_refine: float = 0.3,
    theta_coarsen: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark cells for refinement and coarsening by thresholding on the global maximum.

    This is the cheap "maximum strategy" rather than true bulk (Dorfler) marking: it
    needs a single ``allreduce`` instead of a global sort, and for a front-tracking
    indicator the two agree closely.

    Args:
        eta: per-cell indicator.
        comm: the MPI communicator to reduce the maximum over.
        theta_refine: refine cells with ``eta > theta_refine * max(eta)``.
        theta_coarsen: coarsen cells with ``eta < theta_coarsen * max(eta)``.

    Returns:
        ``(refine_cells, coarsen_cells)``, both int32 index arrays.
    """
    if theta_coarsen >= theta_refine:
        raise ValueError("theta_coarsen must be smaller than theta_refine")

    eta_max = comm.allreduce(eta.max() if eta.size else 0.0, op=MPI.MAX)
    if eta_max <= 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, np.arange(len(eta), dtype=np.int32)

    refine = np.flatnonzero(eta > theta_refine * eta_max).astype(np.int32)
    coarsen = np.flatnonzero(eta < theta_coarsen * eta_max).astype(np.int32)
    return refine, coarsen


def threshold_mark(
    eta: np.ndarray,
    refine_above: float = 0.05,
    coarsen_below: float = 0.01,
    max_splits: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mark against absolute thresholds, for use with the ``"variation"`` indicator.

    With ``refine_above=0.05`` a cell is refined when the concentration changes by more
    than 5% across it, and coarsened when it changes by less than ``coarsen_below``.
    Because the criterion does not reference the current maximum, the mesh reaches a
    steady size once the front is resolved instead of growing for as long as the run
    lasts.

    A cell may need more than one bisection to come under the threshold. Since the
    indicator is close to linear in ``h``, the number needed is
    ``log2(eta / refine_above)``, and doing them all at once lets the mesh catch up
    with a front in one adapt rather than over the next several dozen timesteps.

    Returns:
        ``(refine_cells, coarsen_cells, splits)`` where ``splits[i]`` is how many times
        ``refine_cells[i]`` should be bisected.
    """
    if coarsen_below >= refine_above:
        raise ValueError("coarsen_below must be smaller than refine_above")

    refine = np.flatnonzero(eta > refine_above).astype(np.int32)
    coarsen = np.flatnonzero(eta < coarsen_below).astype(np.int32)
    splits = np.ceil(np.log2(eta[refine] / refine_above)).astype(np.int32)
    np.clip(splits, 1, max_splits, out=splits)
    return refine, coarsen, splits


IndicatorFn = Callable[[object], np.ndarray]
