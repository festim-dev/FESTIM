"""Mesh adaptation backends for the AMR prototype.

Two backends, one interface. Both answer the same question -- "given cells to refine
and cells to coarsen, what is the new mesh?" -- but they get there differently, because
**DOLFINx cannot coarsen**: ``dolfinx.mesh.refine`` (Plaza) is one-way and there is no
inverse operation. Unrefinement has to come from rebuilding.

- :class:`Mesh1DAdapter` rebuilds the vertex array outright. Exact refinement *and*
  coarsening, no parent map needed, 1D only.
- :class:`BaseMeshRefiner` keeps the original mesh forever and re-refines it from
  scratch each generation against a per-base-cell *level* field. Coarsening emerges
  from lowering a level. Works in any dimension.

Both return ``(festim_mesh, facet_meshtags, volume_meshtags)`` where the meshtags are
``None`` when the problem's tags are locator-defined and FESTIM should re-derive them.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
from dolfinx import mesh as dmesh

import festim as F

AdaptResult = tuple[F.Mesh, dolfinx.mesh.MeshTags | None, dolfinx.mesh.MeshTags | None]


class Mesh1DAdapter:
    """Rebuild a 1D mesh from its vertices.

    Refinement bisects a cell; coarsening merges an adjacent pair. Because the mesh is
    rebuilt rather than derived, there is no parent map, so user-supplied meshtags are
    carried by *coordinate provenance* instead: refinement only adds vertices and
    coarsening only removes them, so a surviving vertex keeps its facet tag, and every
    new cell lies inside exactly one old cell.

    Two rules make that exact, and they are what protects the mesh from becoming
    invalid:

    - a *protected vertex* is never removed: subdomain borders (which
      ``Mesh1D.check_borders`` requires to be mesh vertices), block boundaries, and any
      vertex carrying a facet tag;
    - two cells are never merged across a cell-tag boundary.

    Args:
        problem: the problem being adapted, read for its subdomain borders.
        hmin: never bisect a cell that would produce cells smaller than this.
        hmax: never merge a pair that would produce a cell larger than this.
        max_cells: stop refining once the mesh has this many cells.
        buffer: also refine (and never coarsen) within this distance of a marked cell.
            A front moves between two adapts, and without a buffer it walks straight
            out of the refined band -- in the trapping demo it travelled 1.4 um per
            adapt at late times, many times the refined cell size, and the resolved
            band was left behind it.
    """

    def __init__(
        self,
        problem,
        hmin: float,
        hmax: float,
        max_cells: int = 100_000,
        buffer: float = 0.0,
    ):
        self.hmin = hmin
        self.hmax = hmax
        self.max_cells = max_cells
        self.buffer = buffer

        self.protected = set()
        for vol in problem.volume_subdomains:
            borders = getattr(vol, "borders", None)
            if borders is not None:
                self.protected.update(float(b) for b in borders)
        for block in problem.mesh.vertex_blocks:
            self.protected.add(float(block[0]))
            self.protected.add(float(block[-1]))

    def _is_protected(self, x: float, tol: float = 1e-14) -> bool:
        return any(abs(x - p) <= tol * max(1.0, abs(p)) for p in self.protected)

    def adapt(self, problem, refine_cells, coarsen_cells, splits=None) -> AdaptResult:
        """Return the new mesh.

        Args:
            problem: the problem whose mesh is being adapted.
            refine_cells / coarsen_cells: DOLFINx cell indices to bisect and to merge.
            splits: how many times to bisect each entry of ``refine_cells``. Defaults
                to one. More than one lets the mesh catch up with a front in a single
                adapt instead of over the following dozens of timesteps.

        Returns:
            ``(mesh, None, None)``. The tags are ``None`` because 1D subdomains are
            located geometrically, so FESTIM re-derives them on the new mesh.
        """
        blocks = problem.mesh.vertex_blocks
        midpoints = np.concatenate(
            [0.5 * (b[:-1] + b[1:]) for b in blocks]
        )  # real cells only -- the gap between two blocks is not a cell
        order = self._geometric_cell_order(problem.mesh.mesh, midpoints)

        n_cells = len(midpoints)
        refine_cells = np.asarray(refine_cells, dtype=np.int32)
        if splits is None:
            splits = np.ones(len(refine_cells), dtype=np.int32)

        n_splits = np.zeros(n_cells, dtype=np.int32)
        n_splits[order[refine_cells]] = np.asarray(splits, dtype=np.int32)
        to_coarsen = np.zeros(n_cells, dtype=bool)
        to_coarsen[order[np.asarray(coarsen_cells, dtype=np.int32)]] = True

        budget = self.max_cells - n_cells
        new_blocks = []
        offset = 0
        for block in blocks:
            n = len(block) - 1
            new_block, budget = self._adapt_block(
                block,
                n_splits[offset : offset + n],
                to_coarsen[offset : offset + n],
                budget,
            )
            new_blocks.append(new_block)
            offset += n

        vertices = new_blocks if len(new_blocks) > 1 else new_blocks[0]
        return F.Mesh1D(vertices), None, None

    def _adapt_block(self, vertices, n_splits, to_coarsen, budget):
        """Bisect and merge the cells of one contiguous block of vertices."""
        widths = np.diff(vertices)
        n = len(widths)
        n_splits, to_coarsen = self._apply_buffer(vertices, n_splits, to_coarsen)
        new = [vertices[0]]
        i = 0
        while i < n:
            a, b, h = vertices[i], vertices[i + 1], widths[i]

            if n_splits[i] > 0 and h / 2 >= self.hmin and budget > 0:
                # bisect as many times as asked, subject to hmin and the cell budget
                pieces = 2 ** int(n_splits[i])
                while pieces > 2 and (h / pieces < self.hmin or pieces - 1 > budget):
                    pieces //= 2
                new += list(a + (b - a) * np.arange(1, pieces + 1) / pieces)
                budget -= pieces - 1
                i += 1
                continue

            can_merge = (
                i + 1 < n
                and to_coarsen[i]
                and to_coarsen[i + 1]
                and not self._is_protected(float(b))
                and (vertices[i + 2] - a) <= self.hmax
            )
            if can_merge:
                new.append(vertices[i + 2])  # b is dropped: the two cells become one
                budget += 1
                i += 2
                continue

            new.append(b)
            i += 1

        return np.array(new), budget

    def _apply_buffer(self, vertices, n_splits, to_coarsen):
        """Spread the refinement marks over ``self.buffer`` either side.

        Done in physical distance rather than in cells, because the cells near a front
        are precisely the small ones: a fixed number of cells of buffer shrinks exactly
        where the front is about to move.
        """
        if self.buffer <= 0 or not n_splits.any():
            return n_splits, to_coarsen

        midpoints = 0.5 * (vertices[:-1] + vertices[1:])
        marked = np.flatnonzero(n_splits > 0)
        buffered = n_splits.copy()
        for cell in marked:
            lo = np.searchsorted(midpoints, midpoints[cell] - self.buffer)
            hi = np.searchsorted(midpoints, midpoints[cell] + self.buffer)
            np.maximum(buffered[lo:hi], n_splits[cell], out=buffered[lo:hi])

        return buffered, to_coarsen & (buffered == 0)

    @staticmethod
    def _geometric_cell_order(mesh, midpoints) -> np.ndarray:
        """``order[k]`` is the position of DOLFINx cell ``k`` in geometric order."""
        cmap = mesh.topology.index_map(1)
        n = cmap.size_local + cmap.num_ghosts
        mids = dmesh.compute_midpoints(mesh, 1, np.arange(n, dtype=np.int32))[:, 0]
        # every cell midpoint is in ``midpoints``; find which one it is
        pos = np.searchsorted(midpoints, mids)
        # searchsorted can land one past the match through floating point, so snap
        left = np.clip(pos - 1, 0, len(midpoints) - 1)
        pos = np.clip(pos, 0, len(midpoints) - 1)
        take_left = np.abs(midpoints[left] - mids) < np.abs(midpoints[pos] - mids)
        return np.where(take_left, left, pos).astype(np.int32)


class BaseMeshRefiner:
    """Re-refine a persistent base mesh against a per-base-cell level field.

    This is how coarsening is faked in nD: the base mesh is never thrown away, and each
    generation is built by applying ``dolfinx.mesh.refine`` to it ``max(level)`` times,
    marking at pass ``j`` every cell whose target level exceeds ``j``. Dropping a level
    returns that region to base resolution *exactly*.

    User-supplied meshtags are carried by transferring them at each refine pass, which
    keeps the transfer chain rooted at the base mesh rather than compounding
    generation over generation.

    Not usable in 1D with user-supplied facet tags: DOLFINx raises
    ``Parent facet computation not yet supported!`` for interval meshes.

    Args:
        base_mesh: the ``festim.Mesh`` to refine from. Kept for the whole run.
        max_level: cap on refinement depth.
        base_facet_meshtags / base_volume_meshtags: pass these only when the problem's
            tags are user-supplied; leave ``None`` to let FESTIM re-derive them from
            subdomain locators.
        buffer: also refine (and never coarsen) within this distance of a marked cell,
            so that a front cannot walk out of the refined band between two adapts.
            Measured in physical distance rather than cells, for the same reason as in
            :class:`Mesh1DAdapter`.
    """

    def __init__(
        self,
        base_mesh: F.Mesh,
        max_level: int = 4,
        base_facet_meshtags=None,
        base_volume_meshtags=None,
        buffer: float = 0.0,
    ):
        self.base_mesh = base_mesh
        self.max_level = max_level
        self.buffer = buffer
        self.base_ft = base_facet_meshtags
        self.base_ct = base_volume_meshtags

        vdim = base_mesh.mesh.topology.dim
        cmap = base_mesh.mesh.topology.index_map(vdim)
        self.n_base = cmap.size_local + cmap.num_ghosts
        self.levels = np.zeros(self.n_base, dtype=np.int32)
        # cell of the current mesh -> cell of the base mesh
        self.ancestry = np.arange(self.n_base, dtype=np.int32)

        if vdim == 1 and self.base_ft is not None:
            raise ValueError(
                "BaseMeshRefiner cannot carry user-supplied facet tags in 1D "
                "(DOLFINx has no parent-facet computation for intervals). Use "
                "Mesh1DAdapter, or define the subdomains with locators."
            )

    def adapt(self, problem, refine_cells, coarsen_cells, splits=None) -> AdaptResult:
        """Adjust the level field from the marks, then replay the refinement."""
        mesh = problem.mesh.mesh
        refine_cells = np.asarray(refine_cells, dtype=np.int32)
        coarsen_cells = np.asarray(coarsen_cells, dtype=np.int32)
        if splits is None:
            splits = np.ones(len(refine_cells), dtype=np.int32)
        splits = np.asarray(splits, dtype=np.int32)

        refine_cells, splits, coarsen_cells = self._apply_buffer(
            mesh, refine_cells, splits, coarsen_cells
        )

        target = self.levels[self.ancestry]
        np.maximum.at(target, refine_cells, target[refine_cells] + splits)
        target[coarsen_cells] -= 1
        np.clip(target, 0, self.max_level, out=target)

        # several current cells share a base cell: the finest request wins, so a front
        # crossing a base cell keeps it refined until the front has fully left
        new_levels = np.zeros(self.n_base, dtype=np.int32)
        np.maximum.at(new_levels, self.ancestry, target)
        self.levels = new_levels

        return self._build()

    def _build(self) -> AdaptResult:
        mesh = self.base_mesh.mesh
        vdim = mesh.topology.dim
        ft, ct = self.base_ft, self.base_ct

        # The ancestry is carried as a meshtag rather than by indexing the refined mesh
        # with the ``parent_cell`` array DOLFINx returns. That array is NOT a valid
        # index into the refined mesh: DOLFINx reorders cells after refining, so
        # ``ancestry[parent_cell]`` silently maps children to unrelated parents (in an
        # 8x8 square it put children a full domain width away from their claimed
        # parent). ``transfer_meshtag`` accounts for the reordering, so routing the
        # base-cell indices through it is the only reliable way to keep the map.
        ancestry_tag = dmesh.meshtags(
            mesh,
            vdim,
            np.arange(self.n_base, dtype=np.int32),
            np.arange(self.n_base, dtype=np.int32),
        )

        for j in range(int(self.levels.max())):
            n_current = self._n_cells(mesh)
            ancestry = self._ancestry_from(ancestry_tag, n_current)
            cells = np.flatnonzero(self.levels[ancestry] > j).astype(np.int32)
            if cells.size == 0:
                break

            mesh.topology.create_entities(1)
            edges = dmesh.compute_incident_entities(mesh.topology, cells, vdim, 1)
            option = (
                dmesh.RefinementOption.parent_cell_and_facet
                if ft is not None
                else dmesh.RefinementOption.parent_cell
            )
            refined, parent_cell, parent_facet = dmesh.refine(
                mesh, edges, option=option
            )

            # transfer_meshtag needs these on the *refined* mesh, otherwise it raises
            # "Refined mesh is missing cell-facet connectivity"
            refined.topology.create_entities(1)
            refined.topology.create_connectivity(vdim, vdim - 1)
            refined.topology.create_connectivity(vdim - 1, vdim)

            ancestry_tag = dmesh.transfer_meshtag(ancestry_tag, refined, parent_cell)
            if ct is not None:
                ct = dmesh.transfer_meshtag(ct, refined, parent_cell)
            if ft is not None:
                ft = dmesh.transfer_meshtag(ft, refined, parent_cell, parent_facet)
            mesh = refined

        self.ancestry = self._ancestry_from(ancestry_tag, self._n_cells(mesh))
        return F.Mesh(mesh=mesh), ft, ct

    def _apply_buffer(self, mesh, refine_cells, splits, coarsen_cells):
        """Spread the refinement marks over ``self.buffer`` in every direction.

        A topological dilation (cell -> vertices -> cells) would be the natural thing
        to reach for, but one layer of *refined* cells is exactly as thin as the
        refinement made it, so the buffer would shrink precisely where the front is
        about to move. Distance is the right unit.
        """
        if self.buffer <= 0 or refine_cells.size == 0:
            return refine_cells, splits, coarsen_cells

        from scipy.spatial import cKDTree

        vdim = mesh.topology.dim
        n = self._n_cells(mesh)
        midpoints = dmesh.compute_midpoints(mesh, vdim, np.arange(n, dtype=np.int32))

        tree = cKDTree(midpoints[refine_cells])
        neighbours = tree.query_ball_point(midpoints, r=self.buffer)

        buffered = np.zeros(n, dtype=np.int32)
        for cell, marks in enumerate(neighbours):
            if marks:
                buffered[cell] = splits[marks].max()

        refine = np.flatnonzero(buffered > 0).astype(np.int32)
        return refine, buffered[refine], coarsen_cells[buffered[coarsen_cells] == 0]

    @staticmethod
    def _n_cells(mesh) -> int:
        imap = mesh.topology.index_map(mesh.topology.dim)
        return imap.size_local + imap.num_ghosts

    @staticmethod
    def _ancestry_from(tag, n_cells: int) -> np.ndarray:
        """Unpack an ancestry meshtag into a dense cell-indexed array."""
        ancestry = np.zeros(n_cells, dtype=np.int32)
        ancestry[tag.indices] = tag.values
        return ancestry


def uniform_mesh_1d(x_min: float, x_max: float, n: int) -> F.Mesh1D:
    """Convenience for the demos and reference runs."""
    return F.Mesh1D(np.linspace(x_min, x_max, n + 1))


def n_cells(problem) -> int:
    """Global number of cells in a problem's parent mesh."""
    mesh = problem.mesh.mesh
    imap = mesh.topology.index_map(mesh.topology.dim)
    return mesh.comm.allreduce(imap.size_local, op=MPI.SUM)
